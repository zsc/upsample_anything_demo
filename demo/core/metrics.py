import math

import torch


def l1_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().mean()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().item()
    if mse <= 1e-12:
        return 99.99
    return -10.0 * math.log10(mse)
