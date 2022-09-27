import importlib.util
import math
import sys
import torch

from PIL import Image, ImageOps
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import k_diffusion as K
import numpy as np
import torch.nn as nn

from contextlib import nullcontext
from einops import rearrange, repeat
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from transformers import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .exceptions import StableDiffusionInferenceValueError
from .util import (
    cat_self_with_repeat_interleaved,
    load_img,
    load_model_from_config,
    preprocess_mask,
    prompt_inject_custom_concepts,
    repeat_interleave_along_dim_0,
    split_weighted_subprompts_and_return_cond_latents,
    sum_along_slices_of_dim_0
)

# Deal with "ldm" module collisions.
try:
    from ldm.modules.encoders.modules import FrozenCLIPEmbedder
except ImportError:
    spec = importlib.util.spec_from_file_location('ldm',
        str(Path(__file__).resolve().parent.parent.parent / 'ldm' / '__init__.py'))
    ldm = importlib.util.module_from_spec(spec)
    sys.modules['ldm'] = ldm
    spec.loader.exec_module(ldm)


# Make transformers stop screaming.
logging.set_verbosity_error()


VALID_SAMPLERS = {'k_lms', 'dpm2', 'dpm2_ancestral', 'heun', 'euler',
    'euler_ancestral'}


MAX_STEPS = 250
MIN_HEIGHT = 384
MIN_WIDTH = 384


def k_forward_multiconditionable(
    inner_model: Callable,
    x: torch.Tensor,
    sigma: torch.Tensor,
    uncond: torch.Tensor,
    cond: torch.Tensor,
    cond_scale: float,
    cond_arities: Optional[Iterable[int]],
    cond_weights: Optional[Iterable[float]],
    use_half: bool=False,
) -> torch.Tensor:
    '''
    Magicool k-sampler prompt positive/negative weighting from birch-san.

    https://github.com/Birch-san/stable-diffusion/blob/birch-mps-waifu/scripts/txt2img_fork.py
    '''
    uncond_count = uncond.size(dim=0)
    cond_count = cond.size(dim=0)
    cond_in = torch.cat((uncond, cond)).to(x.device)
    del uncond, cond
    cond_arities_tensor = torch.tensor(cond_arities, device=cond_in.device)
    if use_half and (x.dtype == torch.float32 or x.dtype == torch.float64):
        x = x.half()
    x_in = cat_self_with_repeat_interleaved(t=x,
        factors_tensor=cond_arities_tensor, factors=cond_arities,
        output_size=cond_count)
    del x
    sigma_in = cat_self_with_repeat_interleaved(t=sigma,
        factors_tensor=cond_arities_tensor, factors=cond_arities,
        output_size=cond_count)
    del sigma
    uncond_out, conds_out = inner_model(x_in, sigma_in, cond=cond_in) \
        .split([uncond_count, cond_count])
    del x_in, sigma_in, cond_in
    unconds = repeat_interleave_along_dim_0(t=uncond_out,
        factors_tensor=cond_arities_tensor, factors=cond_arities,
        output_size=cond_count)
    del cond_arities_tensor
    # transform
    #   tensor([0.5, 0.1])
    # into:
    #   tensor([[[[0.5000]]],
    #           [[[0.1000]]]])
    weight_tensor = torch.tensor(list(cond_weights),
        device=uncond_out.device, dtype=uncond_out.dtype) * cond_scale
    weight_tensor = weight_tensor.reshape(len(list(cond_weights)), 1, 1, 1)
    deltas: torch.Tensor = (conds_out-unconds) * weight_tensor
    del conds_out, unconds, weight_tensor
    cond = sum_along_slices_of_dim_0(deltas, arities=cond_arities)
    del deltas
    return uncond_out + cond


class StableDiffusionConfig:
    '''
    Configuration for Stable Diffusion.
    '''
    C = 4 # latent channels
    ckpt = '' # model checkpoint path
    config = '' # model configuration file path
    ddim_eta = 0.0
    ddim_steps = 50
    f = 8 # downsampling factor
    fixed_code = False
    height = 512
    n_iter = 1 # number of times to sample
    n_samples = 1 # batch size, GPU memory use scales quadratically with this but it makes it sample faster!
    precision = 'autocast'
    scale = 7.5 # unconditional guidance scale
    seed = 1
    width = 512


class KCFGDenoiser(nn.Module):
    '''
    k-diffusion sampling with multi-conditionable denoising.
    '''
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        uncond: torch.Tensor,
        cond: torch.Tensor,
        cond_scale: float,
        cond_arities: Optional[Iterable[int]]=None,
        cond_weights: Optional[Iterable[float]]=None,
        use_half: bool=False,
    ) -> torch.Tensor:
        return k_forward_multiconditionable(
            self.inner_model,
            x,
            sigma,
            uncond,
            cond,
            cond_scale,
            cond_arities=cond_arities,
            cond_weights=cond_weights,
            use_half=use_half,
        )


class KCFGDenoiserMasked(nn.Module):
    '''
    k-diffusion sampling with multi-conditionable denoising and masking.
    '''
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        uncond: torch.Tensor,
        cond: torch.Tensor,
        cond_scale: float,
        cond_arities: Optional[Iterable[int]]=None,
        cond_weights: Optional[Iterable[float]]=None,
        use_half: bool=False,
        mask: Optional[torch.Tensor]=None,
        x_frozen: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        denoised = k_forward_multiconditionable(
            self.inner_model,
            x,
            sigma,
            uncond,
            cond,
            cond_scale,
            cond_arities=cond_arities,
            cond_weights=cond_weights,
            use_half=use_half,
        )

        if mask is not None:
            assert x_frozen is not None
            img_orig = x_frozen
            mask_inv = 1. - mask
            denoised = (img_orig * mask) + (denoised * mask_inv)

        return denoised


class StableDiffusionInference:
    '''
    Inference calss for stable diffusion.

    config: OmegaConf containing the SD configuration, loaded from
      configs/stable-diffusion/v1-inference.yaml.
    device: Device we are running on, typically CUDA.
    input_path: Where to play temporary files or stores.
    max_n_subprompts: Maximum number of subprompts for the user to be able to
      use, as more subprompts mean slower inference.
    max_resolution: Max resolution in pixels for generation.
    model: Stable diffusion model instance.
    model_k_wrapped: k-diffusion wrapped model instance.
    model_k_config: Default configuration for the k-diffusion wrapper.
    model_k_config_masked: k-diffusion configuration for masked sampling
      (inpainting/outpainting).
    use_half: Whether or not to use fp16 instead of fp32.
    '''
    opt: StableDiffusionConfig = StableDiffusionConfig()

    config = None
    device = None
    input_path = ''
    max_n_subprompts = None
    max_resolution = None
    model = None
    model_k_wrapped = None
    model_k_config = None
    model_k_config_masked = None
    use_half = True

    def __init__(self,
        checkpoint_loc: Optional[str]=None,
        config_loc: Optional[str]=None,
        height: int=512,
        max_n_subprompts=8,
        max_resolution=589824,
        n_iter: int=1,
        stable_path: str=str(Path(__file__).resolve().parent.parent),
        use_half: bool=True,
        width: int=512,
        **kwargs,
    ):
        '''
        @checkpoint_loc: Location of the weights.
        @config_loc: Location of the OmegaConf configuration file.
        @height: Default height of image in pixels.
        @max_n_subprompts: Maximum number of subprompts you can add to an image
          in the denoising step. More subprompts = slower denoising.
        @max_resolution: The maximum resolution for images in pixels, to keep
          your GPU from OOMing in server applications.
        @n_iter: Default number of iterations for sampler.
        @stable_path: Path for this library.
        @use_half: Sample with FP16 instead of FP32.
        @width: Default width of image in pixels.
        '''
        super().__init__(**kwargs)
        self.input_path = stable_path
        if config_loc is not None:
            self.opt.config = config_loc
        else:
            self.opt.config = f'{stable_path}/configs/stable-diffusion/v1-inference.yaml'
        if checkpoint_loc is not None:
            self.opt.ckpt = checkpoint_loc
        else:
            self.opt.ckpt = f'{stable_path}/models/ldm/stable-diffusion-v1/model.ckpt'

        self.opt.height = height
        self.opt.width = width
        self.opt.n_iter = n_iter

        self.max_n_subprompts = max_n_subprompts
        self.max_resolution = max_resolution

        self.config = OmegaConf.load(f"{self.opt.config}")
        self.model = load_model_from_config(self.config, f"{self.opt.ckpt}",
            use_half=use_half)
        self.use_half = use_half

        self.device = torch.device("cuda") \
            if torch.cuda.is_available() \
            else torch.device("cpu")
        self.model = self.model.to(self.device)

        self.model_k_wrapped = K.external.CompVisDenoiser(self.model)
        self.model_k_config = KCFGDenoiser(self.model_k_wrapped)
        self.model_k_config_masked = KCFGDenoiserMasked(self.model_k_wrapped)

    def _height_and_width_check(self, height, width):
        if height * width > self.max_resolution:
            raise StableDiffusionInferenceValueError(
                f'height {height} and width {width} produce too ' +
                f'many pixels ({height * width}). Max pixels {self.max_resolution}')
        if height % 32 != 0:
            raise StableDiffusionInferenceValueError(
                f'height must be a multiple of 32 (got {height})')
        if width % 32 != 0:
            raise StableDiffusionInferenceValueError(
                f'width must be a multiple of 32 (got {width})')
        if height < MIN_HEIGHT:
            raise StableDiffusionInferenceValueError(
                f'width must be >= {MIN_HEIGHT} (got {height})')
        if width < MIN_WIDTH:
            raise StableDiffusionInferenceValueError(
                f'width must be >= {MIN_WIDTH} (got {width})')

    def precision_cast_and_freeze_model(func: Callable):
        '''
        Decorator for proper usage of the model. Casts floats correctly and
        freezes it for greater precision.
        '''
        def wrapper(self, *args, **kwargs):
            nonlocal func
            precision_scope = autocast \
                if self.opt.precision == "autocast" \
                else nullcontext
            with torch.no_grad():
                with precision_scope("cuda"):
                    with self.model.ema_scope():
                        return func(self, *args, **kwargs)
        return wrapper

    @precision_cast_and_freeze_model
    def compute_conditioning_and_weights(
        self,
        prompt: str,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple], Any]:
        '''
        Get conditioning, a weighted subprompt, and an embedding manager from
        a prompt.

        @prompt: Text prompt to sample. Can have positive or negative weights
          in the form "foo:1 bar:-2". sd-diffusers concepts can be used like:
          '<mycat>', where 'mycat' is the URL name of the diffusers concept.
        @batch_size: Batch size.

        Returns tuple[
            conditioning tensor,
            unconditioning tensor,
            weights prompts list in the form of tuples of str, float (foo, 1.)
            an embedding manager
        ]
        '''
        prompt, embedding_manager = prompt_inject_custom_concepts(prompt,
            self.input_path, self.use_half)

        unconditioning = self.model.get_learned_conditioning(batch_size * [''])
        conditioning, weighted_subprompts = split_weighted_subprompts_and_return_cond_latents(
            prompt,
            self.model.get_learned_conditioning,
            embedding_manager,
            unconditioning,
            max_n_subprompts=self.max_n_subprompts)

        return (
            conditioning,
            unconditioning,
            weighted_subprompts,
            embedding_manager,
        )

    @precision_cast_and_freeze_model
    def sample(
        self,
        prompt: str,
        batch_size: int,
        sampler: str,
        seed: int,
        steps: int,

        conditioning: torch.Tensor=None,
        decode_first_stage: bool=True,
        height: int=None,
        init_latent: torch.Tensor=None,
        init_pil_image: Image=None,
        init_pil_image_as_random_latent: bool=False,
        k_sampler_callback: Callable=None,
        k_sampler_config: nn.Module=None,
        k_sampler_extra_args: Dict=None,
        prompt_concept_injection_required: bool=True,
        return_pil_images: bool=True,
        scale: float=7.5,
        strength: float=0.75,
        unconditioning: torch.Tensor=None,
        weighted_subprompts: List[Tuple]=None,
        width: int=None,
    ) -> Tuple[torch.Tensor, Dict]:
        '''
        Create image(s) from text or (optionally) images.

        If init_pil_image is not None, it will initialize from an image and
        steps will be scaled according to strength.

        Mandatory arguments
        @prompt: Text prompt to sample. Can have positive or negative weights
          in the form "foo:1 bar:-2". sd-diffusers concepts can be used like:
          '<mycat>', where 'mycat' is the URL name of the diffusers concept.
        @batch_size: Batch size.
        @seed: Deterministic seed to use to generate images.
        @steps: Number of steps.

        kwarguments
        @conditioning: Tensor for the conditioning (conditioning embeddings).
        @decode_first_stage: Whether or not to decode the samples produced and
          store them in extra_data. Will be ignored if return_pil_images is
          True.
        @height: Height of the image to produce in pixels.
        @init_latent: A tensor representing the latent space image to begin
          diffusion from. If this is set, the number of steps is scaled by the
          strength such that actual_steps = math.floor(steps * strength).
        @init_pil_image: A PIL image to use to initialize an image2image
          conversion. If this is set, the number of steps is scaled by the
          strength such that actual_steps = math.floor(steps * strength).
          The image can be RGB or RGBA. If the image has an alpha layer, that
          alpha layer is used to mask the image and perform inpainting/
          outpainting.
        @init_pil_image_as_random_latent: Use a random image instead of an
          actual image for image2image.
        @k_sampler_callback: Function for the callback on the k-diffusion
          sampler.
        @k_sampler_config: Use this k-diffusion sampler instead of the default
          one.
        @k_sampler_extra_args: Extra arguments to inject into the k_diffusion
          sampler's extra args. These are merged such that these arguments
          will take precedence.
        @prompt_concept_injection_required: Whether or not prompt has already
          been processed by prompt_inject_custom_concepts, which handles pushing
          new sd-diffusers concepts into the EmbeddingManager. Ignored if
          conditioning is provided because it assumes that when computing the
          conditioning you already altered the embedding manager.
        @return_pil_images: Whether or not to return PIL images.
        @scale: Conditioning scale, multiplier for the conditioning.
        @strength: Strength for when doing "img2img". Adjusts steps by this
          amount such that a strength of 0.5 will halve the number of steps.
          Also is used to condition the amount of noise added to the starting
          latent image.
        @unconditioning: Tensor for unconditioning (unconditioned embeddings).
        @weighted_subprompts: List of tuples in the form (foo, 1.) where "foo"
          is the subprompt and "1." is the float for the weight of that
          subprompt.
        @width: Width of image to produce in pixels.

        Returns:
        tuple[
            samples: latent space image representations after denoising
            extra_data: {
                'cond_arities': arities for conditioning (number of subprompt
                  conditionings)
                'cond_weights': weights for conditioning (weights for each
                  subprompt conditioning)
                'conditioning': conditioning embeddings
                'images': optional; a list of PIL images generated
                'samples': samples, # Latent space representations, undecoded
                'unconditioning': unconditioning embeddings
                'x_noised': the input tensor for diffusion, latent space
                  representation
                'x_samples': Tensor for the image representations after
                  denoising and decoding from latent space
            }
        ]
        '''
        seed_everything(seed)

        _height = self.opt.height if height is None else height
        _width = self.opt.width if width is None else width

        if isinstance(prompt, tuple) or isinstance(prompt, list):
            prompt = prompt[0]

        embedding_manager = None
        if conditioning is None or prompt_concept_injection_required is True:
            prompt, embedding_manager = prompt_inject_custom_concepts(prompt,
                self.input_path, self.use_half)

        if unconditioning is None:
            unconditioning = self.model.get_learned_conditioning(
                batch_size * [''])
        if conditioning is None:
            conditioning, weighted_subprompts = split_weighted_subprompts_and_return_cond_latents(
                prompt,
                self.model.get_learned_conditioning,
                embedding_manager,
                unconditioning,
                max_n_subprompts=self.max_n_subprompts,
            )

        mask_image = None
        mask_latent = None
        mask_tensor_from_image = None
        t_enc = None
        if init_latent is None and init_pil_image is not None:
            # We have an alpha channel so we are inpainting, initialize all the
            # components of masking.
            if init_pil_image.mode == 'RGBA':
                mask_image = init_pil_image.split()[-1]
                mask_image = ImageOps.invert(mask_image)

                temp_tensor, _ = load_img(img=mask_image)
                mask_tensor_from_image = repeat(
                    temp_tensor.to(self.device),
                    '1 ... -> b ...',
                    b=batch_size,
                )
                mask = preprocess_mask(mask_image).to(self.device)
                mask_latent = torch.cat([mask] * batch_size)

            # Load image converts to RGB mode by default, stripping any alpha
            # channels that might exist.
            init_image, (_width, _height) = load_img(img=init_pil_image)
            init_image = init_image.to(self.device)
            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
            if init_pil_image_as_random_latent is False:
                init_latent = self.model.get_first_stage_encoding(
                    self.model.encode_first_stage(init_image))
            else:
                init_latent = torch.zeros(
                    batch_size,
                    4,
                    _height >> 3,
                    _width >> 3,
                ).cuda()
        if init_latent is not None:
            assert 0. < strength < 1., 'can only work with strength in (0.0, 1.0)'
            t_enc = math.floor(strength * steps)

        self._height_and_width_check(_height, _width)
        shape = [self.opt.C, _height // self.opt.f, _width // self.opt.f]

        samples = None

        # k_lms is the fallthrough
        sampling_fn = K.sampling.sample_lms
        if sampler == 'dpm2':
            sampling_fn = K.sampling.sample_dpm_2
        if sampler == 'dpm2_ancestral':
            sampling_fn = K.sampling.sample_dpm_2_ancestral
        if sampler == 'heun':
            sampling_fn = K.sampling.sample_heun
        if sampler == 'euler':
            sampling_fn = K.sampling.sample_euler
        if sampler == 'euler_ancestral':
            sampling_fn = K.sampling.sample_euler_ancestral

        sigmas = self.model_k_wrapped.get_sigmas(steps)

        x_noised = None
        if init_latent is not None:
            x_0 = init_latent
            noise = torch.randn_like(x_0) * sigmas[steps - t_enc - 1]
            x_noised = x_0 + noise
            sigmas = sigmas[steps - t_enc - 1:]
        else:
            x_noised = torch.randn([batch_size, *shape], device=self.device) * \
                sigmas[0] # for GPU draw

        was_masked = all(val is not None for val in
            [mask_image, mask_latent, mask_tensor_from_image])
        extra_args = {
            'cond': conditioning,
            'uncond': unconditioning,
            'cond_scale': scale,
            'cond_arities': [len(weighted_subprompts),] * batch_size,
            'cond_weights': [pr[1] for pr in weighted_subprompts] * batch_size,
            'use_half': self.use_half,
        }
        if was_masked:
            extra_args['mask'] = mask_latent
            extra_args['x_frozen'] = init_latent
        if k_sampler_extra_args is not None:
            extra_args = {
                **extra_args,
                **k_sampler_extra_args,
            }

        _k_sampler_config = k_sampler_config
        if _k_sampler_config is None:
            _k_sampler_config = self.model_k_config
        if mask_image  is not None:
            _k_sampler_config = self.model_k_config_masked
        samples = sampling_fn(
            _k_sampler_config,
            x_noised,
            sigmas,
            callback=k_sampler_callback,
            extra_args=extra_args)

        x_samples = None
        if decode_first_stage is True or return_pil_images is True:
            x_samples = self.model.decode_first_stage(samples)

        images: List = []
        if return_pil_images is True and not was_masked:
            x_samples_clamped = torch.clamp(
                (x_samples + 1.0) / 2.0,
                min=0.0,
                max=1.0)

            for x_sample_c in x_samples_clamped:
                x_sample_c = 255. * rearrange(x_sample_c.cpu().numpy(),
                    'c h w -> h w c')
                img = Image.fromarray(x_sample_c.astype(np.uint8))
                buffered = BytesIO()
                img.save(buffered, format='PNG')

                images.append(img)
        if return_pil_images is True and was_masked:
            image = torch.clamp(
                (init_image + 1.0) / 2.0,
                min=0.0, max=1.0)
            mask = torch.clamp((mask_tensor_from_image + 1.0) / 2.0,
                min=0.0, max=1.0)

            x_samples_clamped = torch.clamp(
                (x_samples + 1.0) / 2.0,
                min=0.0,
                max=1.0)

            for x_sample_c in x_samples_clamped:
                inpainted = (1 - mask) * image + mask * x_sample_c
                inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0] * 255
                img = Image.fromarray(inpainted.astype(np.uint8))
                images.append(img)

        torch.cuda.empty_cache()
        extra_data = {
            'cond_arities': extra_args['cond_arities'],
            'cond_weights': extra_args['cond_weights'],
            'conditioning': conditioning,
            'unconditioning': unconditioning,
            'x_noised': x_noised,
            'x_samples': x_samples, # Decoded samples
        }

        if return_pil_images:
            extra_data['images'] = images

        return samples, extra_data
