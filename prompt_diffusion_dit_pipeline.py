# Copyright 2024 Stability AI, The HuggingFace Team and InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.models.embeddings import ImageProjection
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3Pipeline, StableDiffusion3PipelineOutput

from prompt_diffusion_dit_controlnet import (
    PromptDiffusionDiTControlNetModel,
    PromptDiffusionMultiDiTControlNetModel
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class PromptDiffusionDiTControlNetPipeline(StableDiffusion3Pipeline):
    """
    Pipeline for text-to-image generation with Stable Diffusion 3 and PromptDiffusionDiTControlNet.
    
    This pipeline extends the StableDiffusion3Pipeline to support the PromptDiffusionDiTControlNetModel which
    accepts two conditional inputs: a primary condition and a query condition.

    Args:
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder (`CLIPTextModel`):
            Frozen CLIP text-encoder. StableDiffusion3 uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 (`CLIPTextModelWithProjection`):
            Second frozen text encoder. StableDiffusion3 uses the text projection layers of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        transformer (`SD3Transformer2DModel`): Diffusion Transformer model.
        controlnet (`PromptDiffusionDiTControlNetModel`, *optional*):
            Conditional control model for the diffusion transformer. If set, the pipeline will use the conditional mechanism of
            ControlNet to condition the denoising process on a control image.
        scheduler (`KarrasDiffusionSchedulers`):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        feature_extractor (`CLIPImageProcessor`):
            Model that extracts features from generated images to be used as inputs for the `image_encoder` by
            resizing and normalizing.
        image_processor (`VaeImageProcessor`):
            A image_processor to process and prepare batched images for VAE encoding/decoding.
    """
    
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_projection->transformer->controlnet->vae"

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        transformer: SD3Transformer2DModel,
        controlnet: Optional[Union[PromptDiffusionDiTControlNetModel, PromptDiffusionMultiDiTControlNetModel]] = None,
        scheduler: KarrasDiffusionSchedulers = None,
        feature_extractor: Optional[CLIPImageProcessor] = None,
        image_processor: Optional[VaeImageProcessor] = None,
        image_projection: Optional[ImageProjection] = None,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            image_projection=image_projection,
        )
        self.register_modules(
            controlnet=controlnet,
        )

    def prepare_image(
        self, image, width, height, batch_size, num_images_per_prompt, device, dtype, do_classifier_free_guidance
    ):
        """
        Prepares a conditon image for use with a ControlNet.
        """
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=dtype)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    def prepare_control_images(
        self,
        image,
        query_image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
    ):
        """
        Preprocess a control image and an optional query image for PromptDiffusionControlNet.
        """
        if not isinstance(image, list):
            image = [image]

        if not isinstance(query_image, list):
            query_image = [query_image]

        if len(image) != len(query_image):
            raise ValueError(
                f"Length of primary control images {len(image)} must match length of query control images {len(query_image)}"
            )

        # Get required image size
        image_processor = self.image_processor

        processed_images = []
        processed_query_images = []

        for i, (img, q_img) in enumerate(zip(image, query_image)):
            # Process the primary control image
            if isinstance(img, PIL.Image.Image):
                img = [img]

            if isinstance(img, list):
                if not all(isinstance(i, PIL.Image.Image) for i in img):
                    raise ValueError(f"Incorrect format of control image {i}")

                img = [image_processor.preprocess(i, height=height, width=width) for i in img]
                img = torch.cat(img, dim=0)

            # Process the query control image
            if isinstance(q_img, PIL.Image.Image):
                q_img = [q_img]

            if isinstance(q_img, list):
                if not all(isinstance(i, PIL.Image.Image) for i in q_img):
                    raise ValueError(f"Incorrect format of query control image {i}")

                q_img = [image_processor.preprocess(i, height=height, width=width) for i in q_img]
                q_img = torch.cat(q_img, dim=0)

            # Repeat for batch
            img_batch_size = img.shape[0]
            q_img_batch_size = q_img.shape[0]

            if img_batch_size == 1:
                img = img.repeat(batch_size * num_images_per_prompt, 1, 1, 1)
            else:
                img = img.repeat_interleave(num_images_per_prompt, dim=0)

            if q_img_batch_size == 1:
                q_img = q_img.repeat(batch_size * num_images_per_prompt, 1, 1, 1)
            else:
                q_img = q_img.repeat_interleave(num_images_per_prompt, dim=0)

            # Move to device and convert dtype
            img = img.to(device=device, dtype=dtype)
            q_img = q_img.to(device=device, dtype=dtype)

            # Handle classifier-free guidance
            if do_classifier_free_guidance:
                img = torch.cat([img] * 2)
                q_img = torch.cat([q_img] * 2)

            processed_images.append(img)
            processed_query_images.append(q_img)

        return processed_images, processed_query_images

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 30,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        controller: Optional[Any] = None,
        control_image: Union[
            torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]
        ] = None,
        query_control_image: Union[
            torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]
        ] = None,
        conditioning_scale: Union[float, List[float]] = 1.0,
        controlnet_guess_mode: bool = False,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        skip_config_check: bool = False,
        **kwargs,
    ):
        """
        Main method for text-to-image generation with PromptDiffusionDiTControlNet.
        
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation.
            height (`int`, *optional*, defaults to `self.transformer.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.transformer.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 30):
                The number of denoising steps. More steps lead to a higher-quality image at the expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for denoising. If not defined, equal-length timesteps are used.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide image generation.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [schedulers.DDIMScheduler], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                Seed generator(s) for controllable generation.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting).
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting).
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs (prompt weighting).
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose from `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] instead of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.transformer.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Rescale the guidance by `guidance_scale` prior to merging with negative prompt, thereby
                creating a dedicated negative branch.
            original_size (`Tuple[int, int]`, *optional*):
                Original size of the image (height, width) before any resizing.
            crops_coords_top_left (`Tuple[int, int]`, *optional*, defaults to `(0, 0)`):
                Coordinates of the top-left corner of the crop area (top, left).
            target_size (`Tuple[int, int]`, *optional*):
                Target size of the image (height, width) after resizing.
            controller (`na`, *optional*):
                Extension to be run at each callback step.
            control_image (`torch.Tensor`, `PIL.Image.Image`, `List[torch.Tensor]` or `List[PIL.Image.Image]`):
                The primary control image(s) for the PromptDiffusionControlNet model.
            query_control_image (`torch.Tensor`, `PIL.Image.Image`, `List[torch.Tensor]` or `List[PIL.Image.Image]`):
                The query control image(s) for the PromptDiffusionControlNet model.
            conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The size of control image conditioning.
            controlnet_guess_mode (`bool`, *optional*, defaults to `False`):
                When enabled, controlnet will try to recognize control signal automatically.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called.
            joint_attention_kwargs (`dict`, *optional*):
                A keyword args dictionary passed along to `transformer.forward` for the joint cross-attention mechanism.
            skip_config_check (`bool`, *optional*, defaults to `False`):
                Whether to skip checking the config compatibility between `transformer`, `text_encoder` and others.
        
        Returns:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
                A [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or a plain `tuple` if `return_dict` is not True.
                When return a tuple, the first element is a list with processed image(s).
        """
        controlnet = self.controlnet

        # Make sure both control images are provided
        if control_image is None or query_control_image is None:
            raise ValueError("Both control_image and query_control_image must be provided.")

        # Check if controlnet is available
        if controlnet is None:
            raise ValueError("ControlNet must be loaded for this pipeline")

        if not skip_config_check:
            for attr in ["sample_size"]:
                if hasattr(controlnet.config, attr):
                    if getattr(controlnet.config, attr) != getattr(self.transformer.config, attr):
                        raise ValueError(
                            f"`controlnet.config.{attr}` != `transformer.config.{attr}` "
                            f"({getattr(controlnet.config, attr)!r} != {getattr(self.transformer.config, attr)!r})  "
                        )

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Default height and width to transformer
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 3. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 4. Encode input prompt
        lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
        )

        # 5. Prepare timesteps
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, timesteps, device=device)
        latent_timestep = timesteps[:1]

        # 6. Prepare latents
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.transformer.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare control images
        if isinstance(controlnet, PromptDiffusionMultiDiTControlNetModel):
            # Multiple controlnets
            if isinstance(conditioning_scale, float):
                conditioning_scale = [conditioning_scale] * len(controlnet.nets)

            prepared_images, prepared_query_images = self.prepare_control_images(
                control_image,
                query_control_image,
                width,
                height,
                batch_size,
                num_images_per_prompt,
                device,
                prompt_embeds.dtype,
                self.do_classifier_free_guidance,
            )
        else:
            # Single controlnet
            if isinstance(conditioning_scale, list):
                if len(conditioning_scale) != 1:
                    raise ValueError(
                        f"`conditioning_scale` must contain exactly 1 element when using a single controlnet, but got {len(conditioning_scale)}"
                    )
                conditioning_scale = conditioning_scale[0]
            
            prepared_images, prepared_query_images = self.prepare_control_images(
                [control_image],
                [query_control_image],
                width,
                height,
                batch_size,
                num_images_per_prompt,
                device,
                prompt_embeds.dtype,
                self.do_classifier_free_guidance,
            )
            prepared_images = prepared_images[0]
            prepared_query_images = prepared_query_images[0]

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # 8.1 Process controlnet(s) output at timestep t
                if isinstance(controlnet, PromptDiffusionMultiDiTControlNetModel):
                    # Process with multiple controlnets
                    control_model_input = latents
                    controlnet_prompt_embeds = prompt_embeds
                    controlnet_pooled_embeds = pooled_prompt_embeds

                    if self.do_classifier_free_guidance:
                        control_model_input = torch.cat([latents] * 2)
                        controlnet_prompt_embeds = torch.cat([prompt_embeds] * 2)
                        controlnet_pooled_embeds = torch.cat([pooled_prompt_embeds] * 2)

                    # Multiple ControlNet inference
                    control_model_output = controlnet(
                        hidden_states=control_model_input,
                        timestep=t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        pooled_projections=controlnet_pooled_embeds,
                        controlnet_cond=prepared_images,
                        controlnet_query_cond=prepared_query_images,
                        conditioning_scale=conditioning_scale,
                        joint_attention_kwargs=joint_attention_kwargs,
                        return_dict=False,
                    )
                else:
                    # Process with a single controlnet
                    control_model_input = latents
                    controlnet_prompt_embeds = prompt_embeds
                    controlnet_pooled_embeds = pooled_prompt_embeds

                    if self.do_classifier_free_guidance:
                        control_model_input = torch.cat([latents] * 2)
                        controlnet_prompt_embeds = torch.cat([prompt_embeds] * 2)
                        controlnet_pooled_embeds = torch.cat([pooled_prompt_embeds] * 2)

                    # ControlNet inference
                    control_model_output = controlnet(
                        hidden_states=control_model_input,
                        timestep=t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        pooled_projections=controlnet_pooled_embeds,
                        controlnet_cond=prepared_images,
                        controlnet_query_cond=prepared_query_images,
                        conditioning_scale=conditioning_scale,
                        joint_attention_kwargs=joint_attention_kwargs,
                        return_dict=False,
                    )

                # 8.2 Get the transformer output
                transformer_outputs = self.transformer(
                    hidden_states=latents,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"pooled_projections": pooled_prompt_embeds},
                    down_block_additional_residuals=control_model_output[0],
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=True,
                )
                # Get noise predicition
                noise_pred = transformer_outputs.sample

                # 8.3 Handle classifier-free guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # Perform guidance rescale
                    if guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # 8.4 Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **kwargs).prev_sample

                # 8.5 Call callback
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
                
                # 8.6 Update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 9. Post-processing
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        # 10. Convert to output format
        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # 11. Offload all models
        self.maybe_free_model_hooks()

        # 12. Return result
        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)