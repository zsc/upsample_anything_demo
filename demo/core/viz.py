import base64
import io
from typing import Tuple

import numpy as np
from PIL import Image
import torch


def _to_uint8(img: torch.Tensor) -> np.ndarray:
    img = img.detach().float().cpu()
    if img.ndim == 4:
        img = img[0]
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    img = img.clamp(0.0, 1.0)
    img = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return img


def to_base64_png(img: torch.Tensor) -> str:
    arr = _to_uint8(img)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def pca_to_rgb(F: torch.Tensor, sample: int = 10000, seed: int = 0) -> torch.Tensor:
    F = F.detach().float().cpu()
    if F.ndim == 3:
        F = F.unsqueeze(0)
    _, C, H, W = F.shape
    X = F[0].permute(1, 2, 0).reshape(-1, C)

    if C < 3:
        pad = torch.zeros((X.shape[0], 3 - C))
        X = torch.cat([X, pad], dim=1)
        C = 3

    torch.manual_seed(seed)
    if X.shape[0] > sample:
        idx = torch.randperm(X.shape[0])[:sample]
        Xs = X[idx]
    else:
        Xs = X

    mean = Xs.mean(dim=0, keepdim=True)
    Xs = Xs - mean
    try:
        _, _, Vh = torch.linalg.svd(Xs, full_matrices=False)
        V = Vh[:3].T
    except Exception:
        V = torch.eye(C)[:, :3]

    Xc = X - X.mean(dim=0, keepdim=True)
    Y = Xc @ V
    Y = Y.reshape(H, W, 3)

    mins = Y.amin(dim=(0, 1), keepdim=True)
    maxs = Y.amax(dim=(0, 1), keepdim=True)
    Y = (Y - mins) / (maxs - mins + 1e-6)
    return Y.permute(2, 0, 1).unsqueeze(0)


def normalize_map_to_rgb(M: torch.Tensor, clamp: Tuple[float, float] | None = None) -> torch.Tensor:
    M = M.detach().float().cpu()
    if M.ndim == 4:
        M = M[0]
    if M.ndim == 3:
        M = M[0]
    if clamp is not None:
        M = M.clamp(clamp[0], clamp[1])
    minv = M.min()
    maxv = M.max()
    M = (M - minv) / (maxv - minv + 1e-6)
    return M.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
