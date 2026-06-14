from __future__ import annotations

import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from ._protocol import GenerationCompareProtocol, PromptConditioning


def resolve_dtype(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" else torch.float32


class SDXLGenerationCompare(GenerationCompareProtocol):
    @property
    def default_args(self) -> dict:
        return {
            "guidance_scale": 5.0,
            "steps": 40,
            "width": 1024,
            "height": 1024,
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "rollback_steps": 2,
            "substeps": 1,
            "delta_percentile": 98.0,
            "trigger_run_length": 3,
            "min_mask_ratio": 0.003,
            "max_mask_ratio": 0.10,
            "smooth_kernel": 1,
            "dilate_kernel": 1,
            "cooldown_steps": 1,
            "max_rollbacks": 16,
            "outside_replay_force": 0.0,
            "rewrite_noise_strength": 1.0,
        }

    def build_pipeline(
        self, model_id: str, device: str, low_vram: bool
    ) -> StableDiffusionXLPipeline:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=resolve_dtype(device),
        )
        pipeline.scheduler = DDIMScheduler.from_config(
            pipeline.scheduler.config,
            clip_sample=False,
        )
        if low_vram and device == "cuda":
            pipeline.vae.enable_slicing()
            pipeline.enable_model_cpu_offload(device=device)
        else:
            pipeline = pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)
        if device == "cuda":
            pipeline.enable_attention_slicing()
        return pipeline

    def build_prompt_conditioning(
        self,
        pipeline: StableDiffusionXLPipeline,
        prompt: str,
        negative_prompt: str | None,
        guidance_scale: float,
        device: str,
        width: int,
        height: int,
    ) -> PromptConditioning:
        do_cfg = guidance_scale > 1.0
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=torch.device(device),
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
        )
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        add_time_ids = pipeline._get_add_time_ids(
            (height, width),
            (0, 0),
            (height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_text_embeds = pooled_prompt_embeds
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds],
                dim=0,
            )
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        return PromptConditioning(
            prompt_embeds=prompt_embeds.to(device),
            added_cond_kwargs={
                "text_embeds": add_text_embeds.to(device),
                "time_ids": add_time_ids.to(device),
            },
        )
