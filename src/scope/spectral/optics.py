from __future__ import annotations

import math

import torch


def fresnel_tav(alfa: float, nr: torch.Tensor) -> torch.Tensor:
    """Average Fresnel transmission for an incidence angle in degrees."""

    if float(alfa) == 0.0:
        return 4 * nr / ((nr + 1) * (nr + 1))

    angle = torch.as_tensor(alfa * (math.pi / 180.0), device=nr.device, dtype=nr.dtype)
    sin_angle = torch.sin(angle)

    n2 = nr**2
    np_ = n2 + 1
    nm = n2 - 1
    a = ((nr + 1) ** 2) / 2
    k = -((n2 - 1) ** 2) / 4

    b2 = sin_angle**2 - np_ / 2
    b1 = torch.sqrt(torch.clamp(b2**2 + k, min=0.0))
    b = b1 - b2

    ts = ((k**2) / (6 * b**3) + k / b - b / 2) - ((k**2) / (6 * a**3) + k / a - a / 2)
    tp1 = -2 * n2 * (b - a) / (np_**2)
    tp2 = -2 * n2 * np_ * torch.log(b / a) / (nm**2)
    tp3 = n2 * (1 / b - 1 / a) / 2
    tp4 = 16 * n2**2 * (n2**2 + 1) * torch.log((2 * np_ * b - nm**2) / (2 * np_ * a - nm**2)) / (np_**3 * nm**2)
    tp5 = 16 * n2**3 * (1 / (2 * np_ * b - nm**2) - 1 / (2 * np_ * a - nm**2)) / (np_**3)
    return (ts + tp1 + tp2 + tp3 + tp4 + tp5) / (2 * sin_angle**2)
