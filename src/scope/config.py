from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import torch


@dataclass(slots=True)
class SimulationConfig:
    """Shared simulation metadata and PyTorch runtime configuration."""

    roi_bounds: tuple[float, float, float, float]
    start_time: datetime
    end_time: datetime
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    chunk_size: int | None = 1024
    require_grad: bool = False
    extra_options: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        min_y, min_x, max_y, max_x = self.roi_bounds
        if min_y > max_y or min_x > max_x:
            raise ValueError(f"roi_bounds min must not exceed max: got ({min_y}, {min_x}, {max_y}, {max_x})")
        if self.start_time > self.end_time:
            raise ValueError(f"start_time ({self.start_time}) must not be after end_time ({self.end_time})")
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        # Validate that the device string is parseable by PyTorch
        torch.device(self.device)

    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def torch_dtype(self) -> torch.dtype:
        return self.dtype

    def chunks(self, total: int) -> Sequence[slice]:
        if not self.chunk_size or self.chunk_size <= 0:
            return [slice(0, total)]
        return [slice(i, min(i + self.chunk_size, total)) for i in range(0, total, self.chunk_size)]
