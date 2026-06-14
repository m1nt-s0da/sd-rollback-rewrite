from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from tqdm.auto import tqdm

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from ._protocol import GenerationCompareProtocol, PromptConditioning
from ._sd15 import SD15GenerationCompare
from ._sdxl import SDXLGenerationCompare

MODEL_FAMILIES: dict[str, GenerationCompareProtocol] = {
    "sd15": SD15GenerationCompare(),
    "sdxl": SDXLGenerationCompare(),
}


@dataclass
class RollbackEvent:
    step_index: int
    timestep: int
    mask_image: Image.Image


@dataclass
class RunResult:
    image: Image.Image
    executed_steps: int
    rollback_count: int
    rollback_events: list[RollbackEvent]


def infer_model_family(model_id: str | None) -> str:
    if model_id is None:
        return "sd15"
    model_id_lower = model_id.lower()
    if "sdxl" in model_id_lower or "xl-" in model_id_lower or "xl/" in model_id_lower:
        return "sdxl"
    return "sd15"


def resolve_generation_compare(args: argparse.Namespace) -> GenerationCompareProtocol:
    family = args.model_family or infer_model_family(args.model_id)
    return MODEL_FAMILIES[family]


def apply_model_defaults(args: argparse.Namespace) -> argparse.Namespace:
    generation_compare = resolve_generation_compare(args)
    default_args = generation_compare.default_args
    for key, value in default_args.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.model_family is None:
        args.model_family = infer_model_family(args.model_id)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare standard generation, local rollback generation, and compute-matched standard generation"
    )
    parser.add_argument(
        "--model-family",
        choices=tuple(MODEL_FAMILIES.keys()),
        default=None,
        help="Generation model family profile; if omitted, inferred from --model-id",
    )
    parser.add_argument(
        "--prompt", required=True, help="Prompt for text-to-image generation"
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Optional negative prompt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sd_rollback_rewrite.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--model-id",
        "--model",
        dest="model_id",
        default=None,
        help="Diffusers model id for Stable Diffusion",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Base number of denoising steps; defaults depend on the selected model family",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale; defaults depend on the selected model family",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width in pixels; must be divisible by 8",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output height in pixels; must be divisible by 8",
    )
    parser.add_argument(
        "--panel-size",
        type=int,
        default=320,
        help="Thumbnail size for each comparison panel",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed; each batch sample uses seed + index",
    )
    parser.add_argument(
        "--num-samples",
        "--batch-size",
        dest="num_samples",
        type=int,
        default=1,
        help="How many different seeds to generate and compare sequentially; --batch-size is kept as a compatibility alias",
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Enable lower-VRAM pipeline settings such as VAE slicing and CPU offload on CUDA",
    )
    parser.add_argument(
        "--rollback-steps",
        type=int,
        default=None,
        help="How many denoising steps to locally rewind when rewriting; defaults depend on the selected model family",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=None,
        help="How many substeps to use inside each rollback replay step; defaults depend on the selected model family",
    )
    parser.add_argument(
        "--delta-percentile",
        type=float,
        default=None,
        help="Percentile threshold for normalized delta anomaly detection; defaults depend on the selected model family",
    )
    parser.add_argument(
        "--trigger-run-length",
        type=int,
        default=None,
        help="How many consecutive steps a pixel must stay anomalous before rollback can fire; defaults depend on the selected model family",
    )
    parser.add_argument(
        "--min-mask-ratio",
        type=float,
        default=None,
        help="Minimum latent-space area ratio for a rewrite mask; defaults depend on the selected model family",
    )
    parser.add_argument(
        "--max-mask-ratio",
        type=float,
        default=None,
        help="Maximum latent-space area ratio for a rewrite mask; defaults depend on the selected model family",
    )
    parser.add_argument(
        "--smooth-kernel",
        type=int,
        default=None,
        help="Average-pooling kernel size for anomaly smoothing; defaults depend on the selected model family",
    )
    parser.add_argument(
        "--dilate-kernel",
        type=int,
        default=None,
        help="Max-pooling kernel size for rewrite mask dilation; defaults depend on the selected model family",
    )
    parser.add_argument(
        "--cooldown-steps",
        type=int,
        default=None,
        help="How many outer steps to wait before another rollback can fire; defaults depend on the selected model family",
    )
    parser.add_argument(
        "--max-rollbacks",
        type=int,
        default=None,
        help="Maximum number of local rollbacks allowed per generation; defaults depend on the selected model family",
    )
    parser.add_argument(
        "--outside-replay-force",
        type=float,
        default=None,
        help="How strongly to force non-rewrite regions back onto the original replay path: 0 disables it, 1 fully enforces it; defaults depend on the selected model family",
    )
    parser.add_argument(
        "--rewrite-noise-strength",
        type=float,
        default=None,
        help="How strongly to replace the original noise with new noise inside rewrite regions: 0 keeps original noise, 1 fully replaces it; defaults depend on the selected model family",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cpu",
        help="Inference device",
    )
    return apply_model_defaults(parser.parse_args())


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.width % 8 != 0 or args.height % 8 != 0:
        raise ValueError("--width and --height must be divisible by 8")
    if args.panel_size <= 0:
        raise ValueError("--panel-size must be positive")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.rollback_steps <= 0:
        raise ValueError("--rollback-steps must be positive")
    if args.substeps <= 0:
        raise ValueError("--substeps must be positive")
    if not 0.0 < args.delta_percentile < 100.0:
        raise ValueError("--delta-percentile must be between 0 and 100")
    if args.trigger_run_length <= 0:
        raise ValueError("--trigger-run-length must be positive")
    if not 0.0 <= args.min_mask_ratio <= 1.0:
        raise ValueError("--min-mask-ratio must be between 0 and 1")
    if not 0.0 < args.max_mask_ratio <= 1.0:
        raise ValueError("--max-mask-ratio must be between 0 and 1")
    if args.min_mask_ratio > args.max_mask_ratio:
        raise ValueError("--min-mask-ratio must be <= --max-mask-ratio")
    if args.smooth_kernel <= 0 or args.dilate_kernel <= 0:
        raise ValueError("--smooth-kernel and --dilate-kernel must be positive")
    if args.cooldown_steps < 0:
        raise ValueError("--cooldown-steps must be >= 0")
    if args.max_rollbacks < 0:
        raise ValueError("--max-rollbacks must be >= 0")
    if not 0.0 <= args.outside_replay_force <= 1.0:
        raise ValueError("--outside-replay-force must be between 0 and 1")
    if not 0.0 <= args.rewrite_noise_strength <= 1.0:
        raise ValueError("--rewrite-noise-strength must be between 0 and 1")


def make_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def build_prompt_embeddings(
    generation_compare: GenerationCompareProtocol,
    pipeline: Any,
    prompt: str,
    negative_prompt: str | None,
    guidance_scale: float,
    device: str,
    width: int,
    height: int,
) -> PromptConditioning:
    return generation_compare.build_prompt_conditioning(
        pipeline,
        prompt,
        negative_prompt,
        guidance_scale,
        device,
        width,
        height,
    )


def sample_initial_latents(
    pipeline: Any,
    dtype: torch.dtype,
    device: str,
    seed: int,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = pipeline.prepare_latents(
        1,
        pipeline.unet.config.in_channels,
        height,
        width,
        dtype,
        device,
        generator,
    )
    base_noise = latents / pipeline.scheduler.init_noise_sigma
    return latents, base_noise


def decode_latents_to_image(
    pipeline: Any,
    latents: torch.Tensor,
) -> Image.Image:
    latents = latents.detach()
    needs_upcasting = pipeline.vae.dtype == torch.float16 and getattr(
        pipeline.vae.config, "force_upcast", False
    )
    if needs_upcasting:
        pipeline.vae.to(dtype=torch.float32)
        latents = latents.to(
            next(iter(pipeline.vae.post_quant_conv.parameters())).dtype
        )
    elif latents.dtype != pipeline.vae.dtype:
        pipeline.vae.to(dtype=latents.dtype)

    has_latents_mean = (
        hasattr(pipeline.vae.config, "latents_mean")
        and pipeline.vae.config.latents_mean is not None
    )
    has_latents_std = (
        hasattr(pipeline.vae.config, "latents_std")
        and pipeline.vae.config.latents_std is not None
    )
    if has_latents_mean and has_latents_std:
        latents_mean = (
            torch.tensor(pipeline.vae.config.latents_mean)
            .view(1, 4, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(pipeline.vae.config.latents_std)
            .view(1, 4, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = (
            latents * latents_std / pipeline.vae.config.scaling_factor + latents_mean
        )
    else:
        latents = latents / pipeline.vae.config.scaling_factor

    decoded = pipeline.vae.decode(
        latents,
        return_dict=False,
    )[0].detach()
    if needs_upcasting:
        pipeline.vae.to(dtype=torch.float16)
    images = cast(
        list[Image.Image],
        pipeline.image_processor.postprocess(decoded, output_type="pil"),
    )
    return images[0]


def scheduler_sigma(
    pipeline: Any,
    timestep: torch.Tensor,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not hasattr(pipeline.scheduler, "alphas_cumprod"):
        return torch.tensor(1.0, device=device, dtype=dtype)
    alpha_cumprod = pipeline.scheduler.alphas_cumprod.to(device=device, dtype=dtype)
    alpha_t = alpha_cumprod[timestep]
    return torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-6))


def predict_guided_noise(
    pipeline: Any,
    scheduler,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    conditioning: PromptConditioning,
    guidance_scale: float,
) -> torch.Tensor:
    do_cfg = guidance_scale > 1.0
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
    latent_model_input = scheduler.scale_model_input(
        latent_model_input,
        timestep,
    )
    noise_prediction = pipeline.unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=conditioning.prompt_embeds,
        added_cond_kwargs=conditioning.added_cond_kwargs,
        return_dict=False,
    )[0]
    if not do_cfg:
        return noise_prediction
    noise_prediction_uncond, noise_prediction_text = noise_prediction.chunk(2)
    return noise_prediction_uncond + guidance_scale * (
        noise_prediction_text - noise_prediction_uncond
    )


def compute_delta_score_map(
    current_latents: torch.Tensor,
    previous_latents: torch.Tensor | None,
    current_sigma: torch.Tensor,
    previous_sigma: torch.Tensor | None,
) -> torch.Tensor:
    if previous_latents is None or previous_sigma is None:
        return torch.zeros_like(current_latents[:, :1], dtype=torch.float32)
    score_map = torch.linalg.vector_norm(
        (current_latents - previous_latents).float(),
        dim=1,
        keepdim=True,
    )
    sigma_delta = torch.abs(current_sigma.float() - previous_sigma.float())
    return score_map / sigma_delta.clamp_min(1e-6).view(1, 1, 1, 1)


def build_substep_timesteps(
    current_timestep: int,
    next_timestep: int,
    substeps: int,
) -> list[int]:
    if substeps <= 1 or current_timestep <= next_timestep:
        return [current_timestep, next_timestep]

    points = torch.linspace(
        float(current_timestep),
        float(next_timestep),
        steps=substeps + 1,
    )
    rounded = [int(round(point.item())) for point in points]

    timesteps = [rounded[0]]
    for value in rounded[1:]:
        if value != timesteps[-1]:
            timesteps.append(value)

    if timesteps[-1] != next_timestep:
        timesteps.append(next_timestep)
    return timesteps


def run_replay_transition(
    pipeline: Any,
    sample: torch.Tensor,
    current_timestep: torch.Tensor,
    next_timestep: int,
    conditioning: PromptConditioning,
    guidance_scale: float,
    substeps: int,
) -> tuple[torch.Tensor, int]:
    current_value = int(current_timestep.item())
    if substeps == 1 or current_value <= next_timestep:
        replay_noise = predict_guided_noise(
            pipeline,
            pipeline.scheduler,
            sample,
            current_timestep,
            conditioning,
            guidance_scale,
        )
        next_sample = pipeline.scheduler.step(
            replay_noise,
            current_timestep,
            sample,
            eta=0.0,
            return_dict=False,
        )[0]
        return next_sample, 1

    timesteps = build_substep_timesteps(current_value, next_timestep, substeps)
    current_sample = sample

    for start_value, end_value in zip(timesteps[:-1], timesteps[1:]):
        start_timestep = torch.tensor(
            start_value, device=sample.device, dtype=torch.long
        )
        replay_noise = predict_guided_noise(
            pipeline,
            pipeline.scheduler,
            current_sample,
            start_timestep,
            conditioning,
            guidance_scale,
        )
        current_sample = ddim_get_prev_sample(
            cast(DDIMScheduler, pipeline.scheduler),
            current_sample,
            start_value,
            end_value,
            replay_noise,
        )

    return current_sample, len(timesteps) - 1


def ddim_get_prev_sample(
    scheduler: DDIMScheduler,
    sample: torch.Tensor,
    timestep: int,
    prev_timestep: int,
    model_output: torch.Tensor,
) -> torch.Tensor:
    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(device=sample.device)
    if prev_timestep >= 0:
        alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep].to(
            device=sample.device
        )
    else:
        alpha_prod_t_prev = scheduler.final_alpha_cumprod.to(device=sample.device)

    alpha_prod_t = alpha_prod_t.to(dtype=sample.dtype)
    alpha_prod_t_prev = alpha_prod_t_prev.to(dtype=sample.dtype)
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (
        sample - beta_prod_t.sqrt() * model_output
    ) / alpha_prod_t.sqrt()
    pred_sample_direction = (1 - alpha_prod_t_prev).sqrt() * model_output
    return alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction


def ddim_rewind_latents_from_current(
    scheduler: DDIMScheduler,
    sample: torch.Tensor,
    noise: torch.Tensor,
    current_timestep: int,
    target_timestep: int,
) -> torch.Tensor:
    alpha_prod_t = scheduler.alphas_cumprod[current_timestep].to(
        device=sample.device,
        dtype=sample.dtype,
    )
    alpha_prod_s = scheduler.alphas_cumprod[target_timestep].to(
        device=sample.device,
        dtype=sample.dtype,
    )
    coeff_sample = alpha_prod_s.sqrt() / alpha_prod_t.sqrt()
    coeff_noise = (1 - alpha_prod_s).sqrt() - coeff_sample * (1 - alpha_prod_t).sqrt()
    return coeff_sample * sample + coeff_noise * noise.to(
        device=sample.device,
        dtype=sample.dtype,
    )


def smooth_score_map(score_map: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = make_odd(max(1, kernel_size))
    if kernel_size == 1:
        return score_map
    padding = kernel_size // 2
    return F.avg_pool2d(score_map, kernel_size=kernel_size, stride=1, padding=padding)


def dilate_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = make_odd(max(1, kernel_size))
    if kernel_size == 1:
        return mask
    padding = kernel_size // 2
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)


def blur_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = make_odd(max(1, kernel_size))
    if kernel_size == 1:
        return mask
    padding = kernel_size // 2
    return F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)


def build_rewrite_mask(
    persistent_hits: torch.Tensor,
    smoothed_delta: torch.Tensor,
    run_length: int,
    min_area_ratio: float,
    max_area_ratio: float,
    dilate_kernel: int,
) -> torch.Tensor:
    trigger_mask = (persistent_hits >= run_length).to(smoothed_delta.dtype)
    if trigger_mask.max().item() <= 0:
        return torch.zeros_like(trigger_mask)

    masked_scores = smoothed_delta * trigger_mask
    flat_scores = masked_scores.flatten()
    positive_scores = flat_scores[flat_scores > 0]
    if positive_scores.numel() == 0:
        return torch.zeros_like(trigger_mask)

    latent_area = trigger_mask.shape[-2] * trigger_mask.shape[-1]
    min_pixels = max(1, math.ceil(latent_area * min_area_ratio))
    max_pixels = max(min_pixels, math.floor(latent_area * max_area_ratio))

    active_pixels = int(trigger_mask.sum().item())
    if active_pixels < min_pixels:
        return torch.zeros_like(trigger_mask)

    if active_pixels > max_pixels:
        quantile = 1.0 - (max_pixels / positive_scores.numel())
        quantile = float(min(max(quantile, 0.0), 1.0))
        threshold = torch.quantile(positive_scores, quantile)
        trigger_mask = ((masked_scores >= threshold) & (masked_scores > 0)).to(
            smoothed_delta.dtype
        )

    hard_mask = trigger_mask
    dilated_mask = dilate_mask(hard_mask, dilate_kernel)
    soft_mask = blur_mask(dilated_mask, dilate_kernel)
    soft_mask = torch.maximum(soft_mask, hard_mask).clamp(0.0, 1.0)
    if int(hard_mask.sum().item()) < min_pixels:
        return torch.zeros_like(soft_mask)
    return soft_mask


def render_comparison(
    grouped_results: list[tuple[int, list[tuple[str, str, RunResult]]]],
    panel_size: int,
) -> Image.Image:
    header_height = 80
    margin = 20
    row_gap = 28
    columns = len(grouped_results[0][1])
    sidebar_width = compute_rollback_sidebar_width(panel_size)
    width = margin * 2 + columns * panel_size + sidebar_width + margin
    height = margin
    row_heights: list[int] = []
    for _seed, results in grouped_results:
        rewrite_result = results[1][2]
        row_height = max(
            panel_size, compute_rollback_sidebar_height(rewrite_result, panel_size)
        )
        row_heights.append(row_height)
        height += header_height + row_height + row_gap
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    current_y = margin
    for (seed, results), row_height in zip(grouped_results, row_heights):
        draw.text((margin, current_y), f"seed {seed}", fill=(0, 0, 0))
        for index, (title, subtitle, result) in enumerate(results):
            x = margin + index * panel_size
            draw.text((x, current_y + 22), title, fill=(0, 0, 0))
            draw.text((x, current_y + 44), subtitle, fill=(0, 0, 0))
            image = fit_image_to_panel(result.image, panel_size)
            image_x = x + (panel_size - image.width) // 2
            image_y = current_y + header_height + (row_height - image.height) // 2
            canvas.paste(image, (image_x, image_y))
        sidebar_x = margin + columns * panel_size + margin
        render_rollback_sidebar(
            canvas,
            sidebar_x,
            current_y + header_height,
            sidebar_width,
            row_height,
            results[1][2],
            panel_size,
        )
        current_y += header_height + row_height + row_gap

    return canvas


def fit_image_to_panel(image: Image.Image, panel_size: int) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail((panel_size, panel_size), Image.Resampling.LANCZOS)
    return fitted


def rollback_mask_to_image(mask: torch.Tensor) -> Image.Image:
    mask_cpu = mask.detach().float().cpu().squeeze(0).squeeze(0).clamp(0.0, 1.0)
    mask_uint8 = (mask_cpu * 255).round().to(torch.uint8).numpy()
    return Image.fromarray(mask_uint8, mode="L").convert("RGB")


def compute_rollback_sidebar_height(result: RunResult, panel_size: int) -> int:
    header_height = 36
    if not result.rollback_events:
        return header_height + 24

    thumb_size = compute_rollback_thumb_size(panel_size)
    columns = min(4, len(result.rollback_events))
    rows = math.ceil(len(result.rollback_events) / columns)
    return header_height + rows * (thumb_size + 28)


def compute_rollback_thumb_size(panel_size: int) -> int:
    return max(24, min(64, panel_size // 4))


def compute_rollback_sidebar_width(panel_size: int) -> int:
    thumb_size = compute_rollback_thumb_size(panel_size)
    columns = 4
    horizontal_padding = 16
    x_gap = 12
    return horizontal_padding + columns * thumb_size + (columns - 1) * x_gap


def render_rollback_sidebar(
    canvas: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    rewrite_result: RunResult,
    panel_size: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x, y, x + width, y + height), outline=(180, 180, 180), width=1)
    draw.text((x + 8, y + 8), "rollbacks", fill=(0, 0, 0))

    if not rewrite_result.rollback_events:
        draw.text((x + 8, y + 26), "none", fill=(80, 80, 80))
        return

    thumb_size = compute_rollback_thumb_size(panel_size)
    columns = min(4, len(rewrite_result.rollback_events))
    x_gap = 12
    y_gap = 12
    start_y = y + 30
    for event_index, event in enumerate(rewrite_result.rollback_events):
        column = event_index % columns
        row = event_index // columns
        item_x = x + 8 + column * (thumb_size + x_gap)
        item_y = start_y + row * (thumb_size + 28)
        draw.text(
            (item_x, item_y),
            f"s{event.step_index + 1}/t{event.timestep}",
            fill=(0, 0, 0),
        )
        thumbnail = event.mask_image.copy()
        thumbnail.thumbnail((thumb_size, thumb_size), Image.Resampling.NEAREST)
        canvas.paste(thumbnail, (item_x, item_y + 14))


def blend_outside_reference(
    rewrite_mask: torch.Tensor,
    rewritten_latents: torch.Tensor,
    original_reference: torch.Tensor,
    outside_replay_force: float,
) -> torch.Tensor:
    if outside_replay_force <= 0.0:
        return rewritten_latents

    outside_reference = torch.lerp(
        rewritten_latents,
        original_reference,
        outside_replay_force,
    )
    return rewrite_mask * rewritten_latents + (1.0 - rewrite_mask) * outside_reference


def run_standard_generation(
    pipeline: Any,
    conditioning: PromptConditioning,
    latents: torch.Tensor,
    guidance_scale: float,
    steps: int,
    device: str,
    progress_label: str,
) -> RunResult:
    pipeline.scheduler.set_timesteps(steps, device=device)
    current_latents = latents.clone()

    with torch.inference_mode():
        for timestep in tqdm(
            pipeline.scheduler.timesteps,
            desc=progress_label,
            leave=False,
        ):
            guided_noise = predict_guided_noise(
                pipeline,
                pipeline.scheduler,
                current_latents,
                timestep,
                conditioning,
                guidance_scale,
            )
            current_latents = pipeline.scheduler.step(
                guided_noise,
                timestep,
                current_latents,
                return_dict=False,
            )[0]

    return RunResult(
        image=decode_latents_to_image(pipeline, current_latents),
        executed_steps=steps,
        rollback_count=0,
        rollback_events=[],
    )


def run_local_rewrite_generation(
    pipeline: Any,
    conditioning: PromptConditioning,
    latents: torch.Tensor,
    base_noise: torch.Tensor,
    seed: int,
    args: argparse.Namespace,
    progress_label: str,
) -> RunResult:
    pipeline.scheduler.set_timesteps(args.steps, device=args.device)
    timesteps = pipeline.scheduler.timesteps
    current_latents = latents.clone()
    original_step_latents: list[torch.Tensor] = [current_latents.clone()]
    previous_step_latents: torch.Tensor | None = None
    previous_sigma: torch.Tensor | None = None
    persistent_hits: torch.Tensor | None = None
    cooldown_remaining = 0
    rollback_count = 0
    executed_steps = 0
    rollback_events: list[RollbackEvent] = []
    rewrite_noise_generator = torch.Generator(device=args.device).manual_seed(
        seed + 10_000
    )

    with torch.inference_mode():
        for step_index, timestep in enumerate(
            tqdm(
                timesteps,
                desc=progress_label,
                leave=False,
            )
        ):
            timestep_batch = timestep.reshape(1).to(device=args.device)
            step_input_latents = current_latents.clone()
            guided_noise = predict_guided_noise(
                pipeline,
                pipeline.scheduler,
                step_input_latents,
                timestep,
                conditioning,
                args.guidance_scale,
            )
            current_sigma = scheduler_sigma(
                pipeline,
                timestep_batch,
                args.device,
                step_input_latents.dtype,
            )
            delta_score_map = compute_delta_score_map(
                step_input_latents,
                previous_step_latents,
                current_sigma,
                previous_sigma,
            )
            smoothed_delta = smooth_score_map(delta_score_map, args.smooth_kernel)

            if persistent_hits is None:
                persistent_hits = torch.zeros_like(smoothed_delta, dtype=torch.int64)

            threshold = torch.quantile(
                smoothed_delta.flatten(),
                args.delta_percentile / 100.0,
            )
            anomaly_mask = (smoothed_delta >= threshold).to(torch.int64)
            persistent_hits = torch.where(
                anomaly_mask > 0,
                persistent_hits + 1,
                torch.zeros_like(persistent_hits),
            )

            next_latents = pipeline.scheduler.step(
                guided_noise,
                timestep,
                step_input_latents,
                return_dict=False,
            )[0]
            executed_steps += 1
            current_latents = next_latents
            original_step_latents.append(next_latents.detach().clone())

            fired = False
            if (
                cooldown_remaining == 0
                and rollback_count < args.max_rollbacks
                and persistent_hits is not None
            ):
                rewrite_mask = build_rewrite_mask(
                    persistent_hits,
                    smoothed_delta,
                    args.trigger_run_length,
                    args.min_mask_ratio,
                    args.max_mask_ratio,
                    args.dilate_kernel,
                ).to(dtype=current_latents.dtype)
                if rewrite_mask.max().item() > 0:
                    rollback_start = max(0, step_index - args.rollback_steps + 1)
                    scheduler = cast(DDIMScheduler, pipeline.scheduler)
                    current_timestep_value = int(timestep.item())
                    rollback_events.append(
                        RollbackEvent(
                            step_index=step_index,
                            timestep=current_timestep_value,
                            mask_image=rollback_mask_to_image(rewrite_mask),
                        )
                    )
                    rewrite_noise = torch.randn(
                        base_noise.shape,
                        generator=rewrite_noise_generator,
                        device=base_noise.device,
                        dtype=base_noise.dtype,
                    )
                    rewrite_noise = torch.lerp(
                        base_noise,
                        rewrite_noise,
                        args.rewrite_noise_strength,
                    )
                    rollback_timestep = int(timesteps[rollback_start].item())
                    original_rewind_latents = original_step_latents[rollback_start].to(
                        device=current_latents.device,
                        dtype=current_latents.dtype,
                    )
                    original_projected_rewind = ddim_rewind_latents_from_current(
                        scheduler,
                        step_input_latents,
                        base_noise,
                        current_timestep_value,
                        rollback_timestep,
                    )
                    rewrite_rewind_latents = ddim_rewind_latents_from_current(
                        scheduler,
                        step_input_latents,
                        rewrite_noise,
                        current_timestep_value,
                        rollback_timestep,
                    )
                    rewrite_delta = rewrite_rewind_latents - original_projected_rewind
                    rewritten_latents = (
                        original_rewind_latents + rewrite_mask * rewrite_delta
                    )
                    has_rewrite_delta = args.rewrite_noise_strength > 0.0
                    for replay_index in range(rollback_start, step_index + 1):
                        replay_timestep = timesteps[replay_index]
                        replay_next_timestep = (
                            int(timesteps[replay_index + 1].item())
                            if replay_index + 1 < len(timesteps)
                            else 0
                        )
                        original_reference = original_step_latents[replay_index + 1].to(
                            device=rewritten_latents.device,
                            dtype=rewritten_latents.dtype,
                        )
                        if has_rewrite_delta:
                            rewritten_latents, transition_steps = run_replay_transition(
                                pipeline,
                                rewritten_latents,
                                replay_timestep,
                                replay_next_timestep,
                                conditioning,
                                args.guidance_scale,
                                args.substeps,
                            )
                        else:
                            rewritten_latents = original_reference
                            transition_steps = 0
                        rewritten_latents = blend_outside_reference(
                            rewrite_mask,
                            rewritten_latents,
                            original_reference,
                            args.outside_replay_force,
                        )
                        executed_steps += transition_steps

                    current_latents = rewritten_latents
                    rollback_count += 1
                    cooldown_remaining = args.cooldown_steps
                    persistent_hits = torch.zeros_like(persistent_hits)
                    previous_step_latents = None
                    previous_sigma = None
                    fired = True

            if not fired:
                previous_step_latents = step_input_latents.detach().clone()
                previous_sigma = current_sigma.detach().clone()
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1

    return RunResult(
        image=decode_latents_to_image(pipeline, current_latents),
        executed_steps=executed_steps,
        rollback_count=rollback_count,
        rollback_events=rollback_events,
    )


def run_comparison(args: argparse.Namespace) -> Path:
    validate_args(args)

    generation_compare = resolve_generation_compare(args)
    pipeline = generation_compare.build_pipeline(
        args.model_id,
        args.device,
        args.low_vram,
    )
    conditioning = build_prompt_embeddings(
        generation_compare,
        pipeline,
        args.prompt,
        args.negative_prompt,
        args.guidance_scale,
        args.device,
        args.width,
        args.height,
    )
    grouped_results: list[tuple[int, list[tuple[str, str, RunResult]]]] = []
    for batch_index in tqdm(range(args.num_samples), desc="samples"):
        seed = args.seed + batch_index
        initial_latents, base_noise = sample_initial_latents(
            pipeline,
            conditioning.prompt_embeds.dtype,
            args.device,
            seed,
            args.width,
            args.height,
        )

        baseline_result = run_standard_generation(
            pipeline,
            conditioning,
            initial_latents,
            args.guidance_scale,
            args.steps,
            args.device,
            progress_label=f"seed {seed} standard",
        )
        rewrite_result = run_local_rewrite_generation(
            pipeline,
            conditioning,
            initial_latents,
            base_noise,
            seed,
            args,
            progress_label=f"seed {seed} rewrite",
        )
        compute_matched_result = run_standard_generation(
            pipeline,
            conditioning,
            initial_latents,
            args.guidance_scale,
            rewrite_result.executed_steps,
            args.device,
            progress_label=f"seed {seed} matched",
        )

        grouped_results.append(
            (
                seed,
                [
                    (
                        "standard",
                        f"steps={baseline_result.executed_steps}",
                        baseline_result,
                    ),
                    (
                        "rewrite enabled",
                        f"steps={rewrite_result.executed_steps}, rollbacks={rewrite_result.rollback_count}",
                        rewrite_result,
                    ),
                    (
                        "compute-matched standard",
                        f"steps={compute_matched_result.executed_steps}",
                        compute_matched_result,
                    ),
                ],
            )
        )

    comparison = render_comparison(grouped_results, args.panel_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(args.output)
    return args.output


def main() -> None:
    args = parse_args()
    output_path = run_comparison(args)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
