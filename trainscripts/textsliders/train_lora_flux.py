# FLUX Concept Sliders Training Script
# Based on train_lora_xl.py and train-flux-concept-sliders.ipynb

from typing import List, Optional
import argparse
import ast
from pathlib import Path
import gc
import copy
import logging
import os
import sys
import random
from contextlib import ExitStack

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

# Add the parent directory to the path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'flux-sliders'))

from utils.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
import train_util
import model_util
import prompt_util
from prompt_util import (
    PromptEmbedsCache,
    PromptEmbedsPair,
    PromptSettings,
)
import debug_util
import config_util
from config_util import RootConfig

# FLUX specific imports
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling

# Custom FLUX pipeline import
try:
    from custom_flux_pipeline import FluxPipeline
except ImportError:
    print("Warning: custom_flux_pipeline not found. Please ensure it's in the utils directory.")
    FluxPipeline = None

NUM_IMAGES_PER_PROMPT = 1


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, subfolder: str = "text_encoder", device="cuda:0"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, device_map=device
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_text_encoders(pretrained_model_name_or_path, class_one, class_two, weight_dtype, device):
    text_encoder_one = class_one.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="text_encoder", 
        torch_dtype=weight_dtype,
        device_map=device
    )
    text_encoder_two = class_two.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="text_encoder_2", 
        torch_dtype=weight_dtype,
        device_map=device
    )
    return text_encoder_one, text_encoder_two


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    # Handle text_ids based on batch size to avoid deprecation warning
    effective_batch_size = batch_size * num_images_per_prompt
    if effective_batch_size == 1:
        # New FLUX API for batch_size=1 expects 2D tensor (sequence_length, 3)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    else:
        # For batch_size > 1, keep original 3D format (batch_size, sequence_length, 3)
        text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length):
    device = text_encoders[0].device
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length=max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        text_ids = text_ids.to(device)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def get_sigmas(timesteps, noise_scheduler_copy, n_dim=4, device='cuda:0', dtype=torch.bfloat16):
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


class PromptEmbedsFlux:
    """FLUX version of prompt embeddings container"""
    def __init__(self, text_embeds, pooled_embeds, text_ids):
        self.text_embeds = text_embeds
        self.pooled_embeds = pooled_embeds
        self.text_ids = text_ids


def train(
    config: RootConfig,
    prompts: list[PromptSettings],
    device,
):
    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts]),
        "config": config.json(),
    }
    save_path = Path(config.save.path)

    if config.logging.verbose:
        print(metadata)

    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.train.precision)

    # Load FLUX models
    pretrained_model_name_or_path = config.pretrained_model.name_or_path
    
    # Load tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        torch_dtype=weight_dtype,
        device_map=device
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        torch_dtype=weight_dtype,
        device_map=device
    )
    
    # Load scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="scheduler",
        torch_dtype=weight_dtype,
        device_map=device
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    # Load text encoders
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path, device=device
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
       pretrained_model_name_or_path, subfolder="text_encoder_2", device=device
    )
    text_encoder_one, text_encoder_two = load_text_encoders(
        pretrained_model_name_or_path, text_encoder_cls_one, text_encoder_cls_two, weight_dtype, device
    )
    
    # Load VAE and Transformer
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
        device_map='auto'
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="transformer", 
        torch_dtype=weight_dtype
    )
    
    # Move models to device and set requires_grad
    text_encoder_one.to(device, dtype=weight_dtype)
    text_encoder_two.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    transformer.to(device, dtype=weight_dtype)
    
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    
    text_encoder_one.eval()
    text_encoder_two.eval()
    vae.eval()
    transformer.eval()

    # Create LoRA network similar to notebook
    networks = {}
    params = []
    modules = DEFAULT_TARGET_REPLACE
    modules += UNET_TARGET_REPLACE_MODULE_CONV
    
    # For now, use single slider (num_sliders = 1)
    num_sliders = 1
    for i in range(num_sliders):
        networks[i] = LoRANetwork(
            transformer,
            rank=config.network.rank,
            multiplier=1.0,
            alpha=config.network.alpha,
            train_method=config.network.training_method,
        ).to(device, dtype=weight_dtype)
        params.extend(networks[i].prepare_optimizer_params())

    # Setup optimizer
    optimizer_module = train_util.get_optimizer(config.train.optimizer)
    optimizer_kwargs = {}
    if config.train.optimizer_args is not None and len(config.train.optimizer_args) > 0:
        for arg in config.train.optimizer_args.split(" "):
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value
            
    optimizer = optimizer_module(params, lr=config.train.lr, **optimizer_kwargs)
    optimizer.zero_grad()
    
    # Setup scheduler
    lr_scheduler = get_scheduler(
        config.train.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=200,
        num_training_steps=config.train.iterations,
        num_cycles=1,
        power=1.0,
    )
    
    criteria = torch.nn.MSELoss()

    print("Prompts")
    for settings in prompts:
        print(settings)

    # Debug
    for i in range(num_sliders):
        debug_util.check_requires_grad(networks[i])
        debug_util.check_training_mode(networks[i])

    # Prepare text embeddings similar to notebook
    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]
    
    # FLUX specific parameters
    max_sequence_length = getattr(config.train, 'max_sequence_length', 512)
    if 'schnell' in pretrained_model_name_or_path:
        max_sequence_length = 256
    
    height = getattr(config.train, 'height', 512)
    width = getattr(config.train, 'width', 512)
    num_inference_steps = getattr(config.train, 'num_inference_steps', 30)
    guidance_scale = getattr(config.train, 'guidance_scale', 3.5)
    
    if 'schnell' in pretrained_model_name_or_path:
        num_inference_steps = 4
        guidance_scale = 0
    
    # Training parameters
    weighting_scheme = getattr(config.train, 'weighting_scheme', 'none')
    logit_mean = getattr(config.train, 'logit_mean', 0.0)
    logit_std = getattr(config.train, 'logit_std', 1.0)
    mode_scale = getattr(config.train, 'mode_scale', 1.29)
    bsz = getattr(config.train, 'batch_size', 1)
    eta = getattr(config.train, 'eta', 2)

    # Compute text embeddings for all prompts at once like in notebook
    all_prompts = []
    for settings in prompts:
        all_prompts.extend([settings.target, settings.positive, settings.neutral])
        
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
            all_prompts, text_encoders, tokenizers, max_sequence_length
        )
        
    # Split embeddings back to individual prompts
    embeds_per_prompt = len(all_prompts) // len(prompts)
    prompt_embeds_list = prompt_embeds.chunk(len(all_prompts))
    pooled_embeds_list = pooled_prompt_embeds.chunk(len(all_prompts))
    text_ids_list = text_ids.chunk(len(all_prompts))
    
    # Store embeddings for each prompt setting
    prompt_embeddings = {}
    for i, settings in enumerate(prompts):
        base_idx = i * 3
        prompt_embeddings[i] = {
            'target': (prompt_embeds_list[base_idx], pooled_embeds_list[base_idx], text_ids_list[base_idx]),
            'positive': (prompt_embeds_list[base_idx + 1], pooled_embeds_list[base_idx + 1], text_ids_list[base_idx + 1]),
            'negative': (prompt_embeds_list[base_idx + 2], pooled_embeds_list[base_idx + 2], text_ids_list[base_idx + 2])
        }

    # Create pipeline for latent generation before cleaning up text encoders
    if FluxPipeline is not None:
        pipe = FluxPipeline(
            noise_scheduler,
            vae,
            text_encoder_one,
            tokenizer_one,
            text_encoder_two,
            tokenizer_two,
            transformer,
        )
        pipe.set_progress_bar_config(disable=True)
    else:
        pipe = None
        print("Warning: FluxPipeline not available, training may not work properly")

    # Clean up tokenizers and text encoders
    del tokenizer_one, tokenizer_two
    del text_encoder_one, text_encoder_two
    flush()

    pbar = tqdm(range(config.train.iterations))
    
    # Use first prompt setting for simplicity (can be extended later)
    target_embeds = prompt_embeddings[0]['target']
    positive_embeds = prompt_embeddings[0]['positive'] 
    negative_embeds = prompt_embeddings[0]['negative']

    # Fix for new FLUX API: ensure txt_ids have correct shape for training
    def fix_txt_ids_shape(embeds):
        prompt_embeds, pooled_embeds, txt_ids = embeds
        if txt_ids.dim() == 3 and txt_ids.shape[0] == 1:
            # Convert 3D (1, seq_len, 3) to 2D (seq_len, 3) for batch_size=1
            txt_ids = txt_ids.squeeze(0)
        return prompt_embeds, pooled_embeds, txt_ids

    target_embeds = fix_txt_ids_shape(target_embeds)
    positive_embeds = fix_txt_ids_shape(positive_embeds)
    negative_embeds = fix_txt_ids_shape(negative_embeds)

    for i in pbar:
        optimizer.zero_grad()

        # Compute density for timestep sampling
        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=bsz,
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=mode_scale,
        )
        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = noise_scheduler_copy.timesteps[indices].to(device=device)
        
        # Get initial latents
        timestep_to_infer = (indices[0] * (num_inference_steps/noise_scheduler_copy.config.num_train_timesteps)).long().item()
        
        with torch.no_grad():
            if pipe is not None:
                # Use target prompt for initial generation
                target_prompt = prompts[0].target
                packed_noisy_model_input = pipe(
                    target_prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=max_sequence_length,
                    num_images_per_prompt=bsz,
                    generator=None,
                    from_timestep=0,
                    till_timestep=timestep_to_infer,
                    output_type='latent'
                )
                vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
                
                if i == 0:
                    model_input = FluxPipeline._unpack_latents(
                        packed_noisy_model_input,
                        height=height,
                        width=width,
                        vae_scale_factor=vae_scale_factor,
                    )
            else:
                # Fallback: create random latents
                latent_channels = transformer.config.in_channels
                packed_noisy_model_input = torch.randn(
                    (bsz, latent_channels, height // 8, width // 8),
                    device=device, dtype=weight_dtype
                )
                model_input = packed_noisy_model_input

        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            model_input.shape[0],
            model_input.shape[2],
            model_input.shape[3],
            device,
            weight_dtype,
        ) if FluxPipeline is not None else None

        # Fix for new FLUX API: ensure img_ids has correct shape
        if latent_image_ids is not None:
            # If it's 3D and batch_size=1, convert to 2D to avoid deprecation warning
            if latent_image_ids.dim() == 3 and latent_image_ids.shape[0] == 1:
                latent_image_ids = latent_image_ids.squeeze(0)  # Remove batch dimension

        # Handle guidance
        if transformer.config.guidance_embeds:
            guidance = torch.tensor([guidance_scale], device=device)
            guidance = guidance.expand(model_input.shape[0])
        else:
            guidance = None

        # Forward pass with LoRA
        with ExitStack() as stack:
            for net in networks:
                stack.enter_context(networks[net])
            model_pred = transformer(
                hidden_states=packed_noisy_model_input,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=target_embeds[1],
                encoder_hidden_states=target_embeds[0],
                txt_ids=target_embeds[2],
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

        if FluxPipeline is not None:
            model_pred = FluxPipeline._unpack_latents(
                model_pred,
                height=int(model_input.shape[2] * vae_scale_factor / 2),
                width=int(model_input.shape[3] * vae_scale_factor / 2),
                vae_scale_factor=vae_scale_factor,
            )

        # Compute target predictions without LoRA
        with torch.no_grad():
            target_pred = transformer(
                hidden_states=packed_noisy_model_input,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=target_embeds[1],
                encoder_hidden_states=target_embeds[0],
                txt_ids=target_embeds[2],
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            
            positive_pred = transformer(
                hidden_states=packed_noisy_model_input,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=positive_embeds[1],
                encoder_hidden_states=positive_embeds[0],
                txt_ids=positive_embeds[2],
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            negative_pred = transformer(
                hidden_states=packed_noisy_model_input,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=negative_embeds[1],
                encoder_hidden_states=negative_embeds[0],
                txt_ids=negative_embeds[2],
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            if FluxPipeline is not None:
                target_pred = FluxPipeline._unpack_latents(
                    target_pred,
                    height=int(model_input.shape[2] * vae_scale_factor / 2),
                    width=int(model_input.shape[3] * vae_scale_factor / 2),
                    vae_scale_factor=vae_scale_factor,
                )
                
                positive_pred = FluxPipeline._unpack_latents(
                    positive_pred,
                    height=int(model_input.shape[2] * vae_scale_factor / 2),
                    width=int(model_input.shape[3] * vae_scale_factor / 2),
                    vae_scale_factor=vae_scale_factor,
                )

                negative_pred = FluxPipeline._unpack_latents(
                    negative_pred,
                    height=int(model_input.shape[2] * vae_scale_factor / 2),
                    width=int(model_input.shape[3] * vae_scale_factor / 2),
                    vae_scale_factor=vae_scale_factor,
                )

            # Compute ground truth prediction
            gt_pred = target_pred + eta * (positive_pred - negative_pred)
            gt_pred = (gt_pred / gt_pred.norm()) * positive_pred.norm()

        # Compute loss
        concept_loss = torch.mean(
            ((model_pred.float() - gt_pred.float()) ** 2).reshape(gt_pred.shape[0], -1),
            1,
        )
        concept_loss = concept_loss.mean()

        concept_loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        
        loss = concept_loss.item()
        logs = {"concept_loss": loss, "lr": lr_scheduler.get_last_lr()[0]}
        pbar.set_postfix(**logs)

        if config.logging.verbose and i % 100 == 0:
            print(f"Step {i}, Loss: {loss:.6f}")

    print('Training Done')
    
    # Save the model
    save_path.mkdir(parents=True, exist_ok=True)
    for i in range(num_sliders):
        networks[i].save_weights(
            save_path / f"slider_{i}.pt",
            dtype=save_weight_dtype,
        )
    print(f"Model saved to {save_path / 'slider_0.pt'}")


def main(args):
    config_file = args.config_file
    config = config_util.load_config_from_yaml(config_file)
    
    attributes = args.attributes
    if attributes is not None:
        attributes = attributes.split(',')
        
    if args.name is not None:
        config.save.name = args.name

    if args.prompts_file is not None:
        config.prompts_file = args.prompts_file
    if args.alpha is not None:
        config.network.alpha = args.alpha
    if args.rank is not None:
        config.network.rank = args.rank
        
    config.save.name += f'_alpha{config.network.alpha}'
    config.save.name += f'_rank{config.network.rank}'
    config.save.name += f'_{config.network.training_method}'
    config.save.path += f'/{config.save.name}'
    
    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, attributes)
    print(prompts)
    device = torch.device(f"cuda:{args.device}")
    train(config, prompts, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=True,
        help="Config file for training.",
    )
    parser.add_argument(
        "--prompts_file",
        required=False,
        help="Prompts file for training.",
        default=None
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=False,
        default=None,
        help="LoRA weight.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        required=False,
        help="Rank of LoRA.",
        default=None,
    )
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="Device to train on.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default=None,
        help="Slider name.",
    )
    parser.add_argument(
        "--attributes",
        type=str,
        required=False,
        default=None,
        help="Attributes to disentangle (comma separated string)",
    )
    
    args = parser.parse_args()
    main(args)
