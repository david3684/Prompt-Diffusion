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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, logging, scale_lora_layers, unscale_lora_layers, zero_module
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.attention_processor import Attention, AttentionProcessor, FusedJointAttnProcessor2_0
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_sd3 import SD3SingleTransformerBlock

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class PromptDiffusionDiTControlNetOutput(BaseOutput):
    controlnet_block_samples: Tuple[torch.Tensor]


class PromptDiffusionDiTControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    PromptDiffusionControlNet model adapted for DiT (Diffusion Transformer) architecture from SD3.
    
    This class extends the SD3ControlNetModel to support the dual conditioning inputs of PromptDiffusion.

    Parameters:
        sample_size (`int`, defaults to `128`): 
            The width/height of the latents.
        patch_size (`int`, defaults to `2`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `16`):
            The number of latent channels in the input.
        num_layers (`int`, defaults to `18`):
            The number of layers of transformer blocks to use.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        num_attention_heads (`int`, defaults to `18`):
            The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, defaults to `4096`):
            The embedding dimension to use for joint text-image attention.
        caption_projection_dim (`int`, defaults to `1152`):
            The embedding dimension of caption embeddings.
        pooled_projection_dim (`int`, defaults to `2048`):
            The embedding dimension of pooled text projections.
        out_channels (`int`, defaults to `16`):
            The number of latent channels in the output.
        pos_embed_max_size (`int`, defaults to `96`):
            The maximum latent height/width of positional embeddings.
        extra_conditioning_channels (`int`, defaults to `0`):
            The number of extra channels to use for conditioning for patch embedding.
        dual_attention_layers (`Tuple[int, ...]`, defaults to `()`):
            The number of dual-stream transformer blocks to use.
        qk_norm (`str`, *optional*, defaults to `None`):
            The normalization to use for query and key in the attention layer.
        pos_embed_type (`str`, defaults to `"sincos"`):
            The type of positional embedding to use.
        use_pos_embed (`bool`, defaults to `True`):
            Whether to use positional embeddings.
        force_zeros_for_pooled_projection (`bool`, defaults to `True`):
            Whether to force zeros for pooled projection embeddings.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        extra_conditioning_channels: int = 0,
        dual_attention_layers: Tuple[int, ...] = (),
        qk_norm: Optional[str] = None,
        pos_embed_type: Optional[str] = "sincos",
        use_pos_embed: bool = True,
        force_zeros_for_pooled_projection: bool = True,
    ):
        super().__init__()
        default_out_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        if use_pos_embed:
            self.pos_embed = PatchEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=self.inner_dim,
                pos_embed_max_size=pos_embed_max_size,
                pos_embed_type=pos_embed_type,
            )
        else:
            self.pos_embed = None
            
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )
        
        if joint_attention_dim is not None:
            self.context_embedder = nn.Linear(joint_attention_dim, caption_projection_dim)

            # `attention_head_dim` is doubled to account for the mixing.
            # It needs to crafted when we get the actual checkpoints.
            self.transformer_blocks = nn.ModuleList(
                [
                    JointTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        context_pre_only=False,
                        qk_norm=qk_norm,
                        use_dual_attention=True if i in dual_attention_layers else False,
                    )
                    for i in range(num_layers)
                ]
            )
        else:
            self.context_embedder = None
            self.transformer_blocks = nn.ModuleList(
                [
                    SD3SingleTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                    )
                    for _ in range(num_layers)
                ]
            )

        # controlnet_blocks - standard controlnet processing blocks
        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(len(self.transformer_blocks)):
            controlnet_block = nn.Linear(self.inner_dim, self.inner_dim)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_blocks.append(controlnet_block)
            
        # Primary condition embedding
        cond_embed_input = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels + extra_conditioning_channels,
            embed_dim=self.inner_dim,
            pos_embed_type=None,
        )
        self.cond_embed_input = zero_module(cond_embed_input)
        
        # Query condition embedding - PromptDiffusion specific
        query_cond_embed_input = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels + extra_conditioning_channels,
            embed_dim=self.inner_dim, 
            pos_embed_type=None,
        )
        self.query_cond_embed_input = zero_module(query_cond_embed_input)

        self.gradient_checkpointing = False

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def fuse_qkv_projections(self):
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedJointAttnProcessor2_0())

    def unfuse_qkv_projections(self):
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)
    
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    @classmethod
    def from_transformer(
        cls, transformer, num_layers=12, num_extra_conditioning_channels=1, load_weights_from_transformer=True
    ):
        config = transformer.config
        config["num_layers"] = num_layers or config.num_layers
        config["extra_conditioning_channels"] = num_extra_conditioning_channels
        controlnet = cls.from_config(config)

        if load_weights_from_transformer:
            controlnet.pos_embed.load_state_dict(transformer.pos_embed.state_dict())
            controlnet.time_text_embed.load_state_dict(transformer.time_text_embed.state_dict())
            controlnet.context_embedder.load_state_dict(transformer.context_embedder.state_dict())
            controlnet.transformer_blocks.load_state_dict(transformer.transformer_blocks.state_dict(), strict=False)

            controlnet.cond_embed_input = zero_module(controlnet.cond_embed_input)
            controlnet.query_cond_embed_input = zero_module(controlnet.query_cond_embed_input)

        return controlnet

    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        controlnet_query_cond: torch.Tensor,  # PromptDiffusion specific: query condition
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, PromptDiffusionDiTControlNetOutput]:
        """
        The forward method for PromptDiffusionDiTControlNetModel.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            controlnet_cond (`torch.Tensor`):
                The primary conditional input tensor.
            controlnet_query_cond (`torch.Tensor`):
                The query conditional input tensor (PromptDiffusion specific).
            conditioning_scale (`float`, defaults to `1.0`):
                The scale factor for ControlNet outputs.
            encoder_hidden_states (`torch.Tensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): 
                Embeddings projected from the embeddings of input conditions.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a `PromptDiffusionDiTControlNetOutput` instead of a plain tuple.

        Returns:
            If `return_dict` is True, a `PromptDiffusionDiTControlNetOutput` is returned, otherwise a
            `tuple` where the first element contains the controlnet block samples.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        if self.pos_embed is not None and hidden_states.ndim != 4:
            raise ValueError("hidden_states must be 4D when pos_embed is used")
        elif self.pos_embed is None and hidden_states.ndim != 3:
            raise ValueError("hidden_states must be 3D when pos_embed is not used")

        if self.context_embedder is not None and encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be provided when context_embedder is used")
        elif self.context_embedder is None and encoder_hidden_states is not None:
            raise ValueError("encoder_hidden_states should not be provided when context_embedder is not used")

        if self.pos_embed is not None:
            hidden_states = self.pos_embed(hidden_states)

        temb = self.time_text_embed(timestep, pooled_projections)

        if self.context_embedder is not None:
            encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # Add primary and query conditions - PromptDiffusion specific
        hidden_states = hidden_states + self.cond_embed_input(controlnet_cond) + self.query_cond_embed_input(controlnet_query_cond)

        block_res_samples = ()

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                if self.context_embedder is not None:
                    encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                    )
                else:
                    hidden_states = self._gradient_checkpointing_func(block, hidden_states, temb)
            else:
                if self.context_embedder is not None:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
                    )
                else:
                    hidden_states = block(hidden_states, temb)

            block_res_samples = block_res_samples + (hidden_states,)

        controlnet_block_res_samples = ()
        for block_res_sample, controlnet_block in zip(block_res_samples, self.controlnet_blocks):
            block_res_sample = controlnet_block(block_res_sample)
            controlnet_block_res_samples = controlnet_block_res_samples + (block_res_sample,)

        # Apply conditioning scale
        controlnet_block_res_samples = [sample * conditioning_scale for sample in controlnet_block_res_samples]

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (controlnet_block_res_samples,)

        return PromptDiffusionDiTControlNetOutput(controlnet_block_samples=controlnet_block_res_samples)


class PromptDiffusionMultiDiTControlNetModel(ModelMixin):
    """
    PromptDiffusionDiTControlNetModel wrapper class for Multi-PromptDiffusionDiTControlNet.

    This module is a wrapper for multiple instances of the PromptDiffusionDiTControlNetModel.
    The forward() API is designed to be compatible with PromptDiffusionDiTControlNetModel.

    Args:
        controlnets (List[PromptDiffusionDiTControlNetModel]):
            Provides additional conditioning to the transformer during the denoising process.
            You must set multiple PromptDiffusionDiTControlNetModel as a list.
    """

    def __init__(self, controlnets):
        super().__init__()
        self.nets = nn.ModuleList(controlnets)

    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: List[torch.tensor],
        controlnet_query_cond: List[torch.tensor],  # PromptDiffusion specific
        conditioning_scale: List[float],
        pooled_projections: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[PromptDiffusionDiTControlNetOutput, Tuple]:
        for i, (cond, query_cond, scale, controlnet) in enumerate(zip(controlnet_cond, controlnet_query_cond, conditioning_scale, self.nets)):
            block_samples = controlnet(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                controlnet_cond=cond,
                controlnet_query_cond=query_cond,
                conditioning_scale=scale,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=return_dict,
            )

            # Merge samples
            if i == 0:
                control_block_samples = block_samples
            else:
                control_block_samples = [
                    control_block_sample + block_sample
                    for control_block_sample, block_sample in zip(control_block_samples[0], block_samples[0])
                ]
                control_block_samples = (tuple(control_block_samples),)

        return control_block_samples