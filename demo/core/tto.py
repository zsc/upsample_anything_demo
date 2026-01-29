import time
from typing import Dict, Tuple

import torch

from .gs_jbu import GSJBUParams, render_gsjbu
from .metrics import l1_loss, psnr


def optimize_params(
    I_hr: torch.Tensor,
    I_lr: torch.Tensor,
    iters: int,
    lr: float,
    radius: int,
    chunk: int = 16384,
) -> Tuple[GSJBUParams, torch.Tensor, Dict[str, float]]:
    device = I_hr.device
    H_lr, W_lr = I_lr.shape[-2:]
    params = GSJBUParams(H_lr, W_lr, device=device)

    if iters <= 0:
        with torch.no_grad():
            I_hat = render_gsjbu(I_lr, I_hr, I_lr, params, stride=I_hr.shape[-1] // W_lr, radius=radius, chunk=chunk)
        metrics = {
            "l1_recon": l1_loss(I_hat, I_hr).item(),
            "psnr_recon": psnr(I_hat, I_hr),
        }
        return params, I_hat, metrics

    optimizer = torch.optim.Adam(params.parameters(), lr=lr)
    start = time.time()

    for step in range(1, iters + 1):
        optimizer.zero_grad(set_to_none=True)
        I_hat = render_gsjbu(I_lr, I_hr, I_lr, params, stride=I_hr.shape[-1] // W_lr, radius=radius, chunk=chunk)
        loss = l1_loss(I_hat, I_hr)
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 1 or step == iters:
            print(f"[TTO] iter {step:03d}/{iters} | L1 {loss.item():.6f}")

    elapsed = time.time() - start
    print(f"[TTO] done in {elapsed:.2f}s")

    with torch.no_grad():
        I_hat = render_gsjbu(I_lr, I_hr, I_lr, params, stride=I_hr.shape[-1] // W_lr, radius=radius, chunk=chunk)

    metrics = {
        "l1_recon": l1_loss(I_hat, I_hr).item(),
        "psnr_recon": psnr(I_hat, I_hr),
    }
    return params, I_hat, metrics
