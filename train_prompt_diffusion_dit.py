#!/usr/bin/env python
# coding=utf-8

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusion3Pipeline
from prompt_diffusion_dit_controlnet import (
    PromptDiffusionDiTControlNetModel,
    PromptDiffusionMultiDiTControlNetModel
)


class PromptDiffusionDiTTrainer(pl.LightningModule):
    """PyTorch Lightning module for training PromptDiffusionDiTControlNetModel."""
    
    def __init__(
        self,
        controlnet: PromptDiffusionDiTControlNetModel,
        transformer: torch.nn.Module,  # DiT model
        vae: torch.nn.Module = None,
        text_encoder: torch.nn.Module = None,
        text_encoder_2: torch.nn.Module = None,
        noise_scheduler: Optional[DDPMScheduler] = None,
        learning_rate: float = 1e-5,
        lr_scheduler_type: str = "cosine",
        lr_warmup_steps: int = 500,
        train_batch_size: int = 4,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        max_grad_norm: float = 1.0,
        conditioning_scale: float = 1.0,
        prediction_type: str = "epsilon",  # or "v_prediction" or "sample"
        prior_preservation_weight: float = 0.0,
        snr_gamma: Optional[float] = None,
    ):
        """
        Initialize the PromptDiffusionDiTTrainer.
        
        Args:
            controlnet: The PromptDiffusionDiTControlNetModel to train
            transformer: The DiT model (SD3Transformer2DModel)
            vae: VAE model for encoding/decoding images
            text_encoder: Primary text encoder for caption embedding
            text_encoder_2: Secondary text encoder for caption embedding
            noise_scheduler: The noise scheduler
            learning_rate: Learning rate
            lr_scheduler_type: Type of learning rate scheduler
            lr_warmup_steps: Number of warmup steps for learning rate scheduler
            train_batch_size: Batch size for training
            adam_beta1: Beta1 parameter for Adam optimizer
            adam_beta2: Beta2 parameter for Adam optimizer
            adam_weight_decay: Weight decay for Adam optimizer
            adam_epsilon: Epsilon parameter for Adam optimizer
            max_grad_norm: Maximum gradient norm for gradient clipping
            conditioning_scale: Scale factor for ControlNet outputs
            prediction_type: Type of prediction to use for loss computation
            prior_preservation_weight: Weight for prior preservation loss
            snr_gamma: SNR gamma parameter for loss weighting
        """
        super().__init__()
        
        self.save_hyperparameters(ignore=["controlnet", "transformer", "vae", "text_encoder", "text_encoder_2", "noise_scheduler"])
        
        self.controlnet = controlnet
        self.sd3_pipe: StableDiffusion3Pipeline = (
            StableDiffusion3Pipeline.from_pretrained(sd3_pipe_id).to(dtype=dtype).to(device)
        )
        self.transformer = self.sd3_pipe.transformer
        self.sd3_pipe.text_encoder.requires_grad_(False).eval()
        self.sd3_pipe.text_encoder_2.requires_grad_(False).eval()
        self.sd3_pipe.vae.requires_grad_(False).eval()
        
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        
        self.noise_scheduler = noise_scheduler or DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_warmup_steps = lr_warmup_steps
        self.train_batch_size = train_batch_size
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_weight_decay = adam_weight_decay
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        self.conditioning_scale = conditioning_scale
        self.prediction_type = prediction_type
        self.prior_preservation_weight = prior_preservation_weight
        self.snr_gamma = snr_gamma
        
        # Freeze the transformer model
        self._freeze_model_components()
    
    def _freeze_model_components(self):
        """Freeze the non-trainable components."""
        # Freeze transformer model
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        # Freeze VAE if available
        if self.vae is not None:
            for param in self.vae.parameters():
                param.requires_grad = False
        
        # Freeze text encoders if available
        if self.text_encoder is not None:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        if self.text_encoder_2 is not None:
            for param in self.text_encoder_2.parameters():
                param.requires_grad = False
    
    def encode_prompt(self, prompt, device, num_images_per_prompt=1, negative_prompt=None):
        """
        Encode the prompt into text encoder hidden states with pooled outputs.
        
        Args:
            prompt: The text prompt to encode
            device: The device to use
            num_images_per_prompt: Number of images per prompt
            negative_prompt: Optional negative prompt
            
        Returns:
            tuple: (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        """
        if self.text_encoder is None or self.text_encoder_2 is None:
            raise ValueError("Text encoders must be provided for prompt encoding")
        
        # Call existing encode_prompt implementation based on pipeline design
        # Placeholder for the encode_prompt logic - this should be implemented based on your specific text encoders
        
        # Simple placeholder - in practice, use the proper text encoding implementation
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        
        # Placeholder tensor shapes - replace with actual implementation
        prompt_embeds = torch.randn(batch_size, 77, 1024, device=device)
        pooled_prompt_embeds = torch.randn(batch_size, 1024, device=device)
        
        if negative_prompt is not None:
            negative_batch_size = len(negative_prompt) if isinstance(negative_prompt, list) else 1
            negative_prompt_embeds = torch.randn(negative_batch_size, 77, 1024, device=device)
            negative_pooled_prompt_embeds = torch.randn(negative_batch_size, 1024, device=device)
        else:
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None
            
        # Repeat for multiple images per prompt
        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
    
    def prepare_latents(self, batch_size, num_channels, height, width, dtype, device, generator=None):
        """
        Prepare random latent tensors for training.
        
        Args:
            batch_size: The batch size
            num_channels: Number of latent channels
            height: Height of latent tensor
            width: Width of latent tensor
            dtype: Data type
            device: Device to use
            generator: Optional random generator
            
        Returns:
            torch.Tensor: Random latent tensors
        """
        latents = torch.randn(
            (batch_size, num_channels, height, width),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        return latents
    
    def compute_snr(self, timesteps):
        """
        Compute the signal-to-noise ratio for a given timestep.
        
        Args:
            timesteps: The timestep values
            
        Returns:
            torch.Tensor: SNR values
        """
        if self.snr_gamma is None:
            return None
        
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
        
        # Expand the tensors to match timesteps
        noise_scheduler_timesteps = torch.tensor(self.noise_scheduler.timesteps, device=timesteps.device)
        timesteps = timesteps.to(noise_scheduler_timesteps.dtype)
        
        step_indices = ((timesteps[:, None] - noise_scheduler_timesteps[None, :]) ** 2).argmin(1)
        
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(timesteps.device)[step_indices]
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(timesteps.device)[step_indices]
        
        # Compute SNR
        signal_to_noise_ratio = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
        
        return signal_to_noise_ratio
    
    def get_diffusion_loss(self, model_pred, target, timesteps=None):
        """
        Compute the diffusion loss based on prediction type.
        
        Args:
            model_pred: Model predictions
            target: Target values
            timesteps: Optional timesteps for SNR weighting
            
        Returns:
            torch.Tensor: Computed loss
        """
        if self.prediction_type == "epsilon":
            loss = F.mse_loss(model_pred, target, reduction="none")
        elif self.prediction_type == "v_prediction":
            # v-prediction loss formula
            alpha_t = self.noise_scheduler.alphas_cumprod[timesteps]
            alpha_t = alpha_t.flatten()
            alpha_t = alpha_t.to(model_pred.device)[..., None, None, None]
            
            # Predicted x0
            pred_x0 = alpha_t.sqrt() * model_pred - (1 - alpha_t).sqrt() * target
            # Predicted epsilon
            pred_epsilon = (1 - alpha_t).sqrt() * model_pred + alpha_t.sqrt() * target
            
            # v-prediction target
            v_target = alpha_t.sqrt() * target - (1 - alpha_t).sqrt() * model_pred
            
            loss = F.mse_loss(model_pred, v_target, reduction="none")
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # Apply SNR weighting if applicable
        if self.snr_gamma is not None and timesteps is not None:
            snr = self.compute_snr(timesteps)
            snr_weight = (snr ** self.snr_gamma) / (1 + snr)
            loss = loss * snr_weight.reshape(-1, 1, 1, 1)
        
        return loss.mean()
    
    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        
        Args:
            batch: The input batch containing images and conditioning information
            batch_idx: The index of the batch
            
        Returns:
            dict: Training loss
        """
        # Unpack the batch
        pixel_values = batch["pixel_values"]  # Original images
        controlnet_cond = batch["controlnet_cond"]  # Primary conditioning image
        controlnet_query_cond = batch["controlnet_query_cond"]  # Query conditioning image
        input_ids = batch.get("input_ids")  # Text prompts
        
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # Encode the prompt if text is available
        if input_ids is not None and self.text_encoder is not None:
            # Convert input_ids to prompts if needed
            prompts = input_ids  # This might need conversion depending on your dataset format
            
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
                prompts, device
            )
        else:
            # No text conditioning - use random embeddings
            prompt_embeds = torch.randn(batch_size, 77, 1024, device=device)
            pooled_prompt_embeds = torch.randn(batch_size, 1024, device=device)
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None
        
        # VAE encode if available, otherwise use the pixel values directly
        if self.vae is not None:
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        else:
            # If no VAE, assume pixel_values are already in latent space
            latents = pixel_values
            
        # Sample noise to add to the latents
        noise = torch.randn_like(latents)
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device
        ).long()
        
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # ControlNet forward pass
        controlnet_outputs = self.controlnet(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            controlnet_cond=controlnet_cond,
            controlnet_query_cond=controlnet_query_cond,
            conditioning_scale=self.conditioning_scale,
            return_dict=True,
        )
        
        # Apply the conditioned transformer - precomputed features
        transformer_outputs = self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={"pooled_projections": pooled_prompt_embeds},
            down_block_additional_residuals=controlnet_outputs.controlnet_block_samples,
            return_dict=True,
        )
        
        # Get the model prediction
        model_pred = transformer_outputs.sample
        
        # Calculate the target based on prediction type
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        elif self.prediction_type == "sample":
            target = latents
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
            
        # Compute loss
        loss = self.get_diffusion_loss(model_pred, target, timesteps)
        
        # Log the loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {"loss": loss}
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Actions to perform at the end of each training batch.
        """
        if batch_idx % 100 == 0:
            # Log additional metrics or save intermediate results here
            # This is just a placeholder
            pass
    
    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.
        
        Args:
            batch: The input batch containing images and conditioning information
            batch_idx: The index of the batch
            
        Returns:
            dict: Validation loss
        """
        # Similar to training_step but without gradient updates
        with torch.no_grad():
            results = self.training_step(batch, batch_idx)
            # Log as validation loss
            self.log("val_loss", results["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
        return results
    
    def configure_optimizers(self):
        """
        Configure optimizers for training.
        
        Returns:
            dict: Optimizer configuration
        """
        # Only optimize parameters that require grad
        params_to_optimize = [p for p in self.controlnet.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,
            eps=self.adam_epsilon,
        )
        
        # Configure the learning rate scheduler
        lr_scheduler = get_scheduler(
            self.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }


class PromptDiffusionDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for PromptDiffusion dataset."""
    
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        Initialize the data module.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            train_batch_size: Batch size for training
            val_batch_size: Batch size for validation
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for data loading
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def train_dataloader(self):
        """
        Create the training dataloader.
        
        Returns:
            DataLoader: Training dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def val_dataloader(self):
        """
        Create the validation dataloader.
        
        Returns:
            DataLoader: Validation dataloader or None if no validation dataset
        """
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    # Example of usage (placeholder code)
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PromptDiffusionDiTControlNetModel")
    parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to pretrained SD3 model")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_train_steps", type=int, default=10000, help="Maximum number of training steps")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    
    args = parser.parse_args()
    
    # This would be implemented with actual model loading and training setup
    print("PromptDiffusionDiT training script - implement actual training code here")