from __future__ import annotations

from typing import cast

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from ._protocol import GenerationCompareProtocol, PromptConditioning


def resolve_dtype(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" else torch.float32


class SD15GenerationCompare(GenerationCompareProtocol):
    @property
    def default_args(self) -> dict:
        return {
            "guidance_scale": 7.5,
            "steps": 30,
            "width": 512,
            "height": 512,
            "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "rollback_steps": 3,
            "substeps": 1,
            "delta_percentile": 97.0,
            "trigger_run_length": 3,
            "min_mask_ratio": 0.005,
            "max_mask_ratio": 0.15,
            "smooth_kernel": 1,
            "dilate_kernel": 3,
            "cooldown_steps": 0,
            "max_rollbacks": 16,
            "outside_replay_force": 0.0,
            "rewrite_noise_strength": 1.0,
        }

    def build_pipeline(
        self, model_id: str, device: str, low_vram: bool
    ) -> StableDiffusionPipeline:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=resolve_dtype(device),
            safety_checker=None,
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
        pipeline: StableDiffusionPipeline,
        prompt: str,
        negative_prompt: str | None,
        guidance_scale: float,
        device: str,
        width: int,
        height: int,
    ) -> PromptConditioning:
        del width, height
        do_cfg = guidance_scale > 1.0
        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            prompt,
            device,
            1,
            do_cfg,
            negative_prompt=negative_prompt,
        )
        if do_cfg:
            prompt_embeds = torch.cat(
                [cast(torch.Tensor, negative_prompt_embeds), prompt_embeds],
                dim=0,
            )
        return PromptConditioning(prompt_embeds=prompt_embeds)
