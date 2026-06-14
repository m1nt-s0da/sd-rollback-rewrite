from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass
class PromptConditioning:
    prompt_embeds: torch.Tensor
    added_cond_kwargs: dict[str, torch.Tensor] | None = None


class GenerationCompareProtocol(Protocol):
    @property
    def default_args(self) -> dict: ...

    def build_pipeline(self, model_id: str, device: str, low_vram: bool) -> Any: ...

    def build_prompt_conditioning(
        self,
        pipeline: Any,
        prompt: str,
        negative_prompt: str | None,
        guidance_scale: float,
        device: str,
        width: int,
        height: int,
    ) -> PromptConditioning: ...
