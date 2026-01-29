import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y: float) -> float:
    y = torch.tensor(float(y))
    return (y + torch.log(-torch.expm1(-y))).item()


class GSJBUParams(nn.Module):
    def __init__(
        self,
        H_lr: int,
        W_lr: int,
        init_sx: float = 16.0,
        init_sy: float = 16.0,
        init_theta: float = 0.0,
        init_sr: float = 0.12,
        eps: float = 1e-6,
        clamp_s: Tuple[float, float] = (1.0, 64.0),
        clamp_r: Tuple[float, float] = (0.01, 1.0),
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        raw_sx = _inv_softplus(init_sx - eps)
        raw_sy = _inv_softplus(init_sy - eps)
        raw_sr = _inv_softplus(init_sr - eps)
        raw_th = math.atanh(max(min(init_theta / math.pi, 0.999), -0.999))

        self.raw_sx = nn.Parameter(torch.full((1, H_lr, W_lr), raw_sx, device=device))
        self.raw_sy = nn.Parameter(torch.full((1, H_lr, W_lr), raw_sy, device=device))
        self.raw_sr = nn.Parameter(torch.full((1, H_lr, W_lr), raw_sr, device=device))
        self.raw_theta = nn.Parameter(torch.full((1, H_lr, W_lr), raw_th, device=device))

        self.eps = eps
        self.clamp_s = clamp_s
        self.clamp_r = clamp_r

    def values(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sx = F.softplus(self.raw_sx) + self.eps
        sy = F.softplus(self.raw_sy) + self.eps
        sr = F.softplus(self.raw_sr) + self.eps
        theta = math.pi * torch.tanh(self.raw_theta)

        sx = torch.clamp(sx, self.clamp_s[0], self.clamp_s[1])
        sy = torch.clamp(sy, self.clamp_s[0], self.clamp_s[1])
        sr = torch.clamp(sr, self.clamp_r[0], self.clamp_r[1])
        return sx, sy, theta, sr


def render_gsjbu(
    V_lr: torch.Tensor,
    I_hr: torch.Tensor,
    I_lr: torch.Tensor,
    params: GSJBUParams,
    stride: int,
    radius: int,
    chunk: int = 16384,
) -> torch.Tensor:
    device = I_hr.device
    H, W = I_hr.shape[-2:]
    H_lr, W_lr = I_lr.shape[-2:]
    C = V_lr.shape[1]
    N = H * W

    sx_map, sy_map, th_map, sr_map = params.values()
    V_hat = torch.zeros((1, C, H, W), device=device, dtype=V_lr.dtype)
    V_hat_flat = V_hat.view(1, C, -1)

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        idx = torch.arange(start, end, device=device)
        py = torch.div(idx, W, rounding_mode="floor")
        px = idx - py * W

        py_f = py.float()
        px_f = px.float()

        u = (py_f + 0.5) / stride - 0.5
        v = (px_f + 0.5) / stride - 0.5
        qy0 = torch.floor(u).long()
        qx0 = torch.floor(v).long()

        Ip = I_hr[:, :, py, px]

        logw_list = []
        val_list = []

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                qy = torch.clamp(qy0 + dy, 0, H_lr - 1)
                qx = torch.clamp(qx0 + dx, 0, W_lr - 1)

                sx = sx_map[:, qy, qx]
                sy = sy_map[:, qy, qx]
                theta = th_map[:, qy, qx]
                sr = sr_map[:, qy, qx]

                Dx = (px_f + 0.5) - (qx.float() + 0.5) * stride
                Dy = (py_f + 0.5) - (qy.float() + 0.5) * stride

                c = torch.cos(theta)
                t = torch.sin(theta)
                dxp = c * Dx + t * Dy
                dyp = -t * Dx + c * Dy

                log_ws = -0.5 * ((dxp / sx) ** 2 + (dyp / sy) ** 2)

                Iq = I_lr[:, :, qy, qx]
                diff2 = ((Ip - Iq) ** 2).sum(dim=1)
                log_wr = -diff2 / (2.0 * (sr**2))

                logw = log_ws + log_wr
                logw_list.append(logw)

                Vq = V_lr[:, :, qy, qx]
                val_list.append(Vq)

        logW = torch.stack(logw_list, dim=0)
        logW = logW - logW.max(dim=0, keepdim=True).values
        Wk = torch.softmax(logW, dim=0)

        Vk = torch.stack(val_list, dim=0)
        out = (Wk.unsqueeze(2) * Vk).sum(dim=0)

        V_hat_flat[:, :, idx] = out

    return V_hat
