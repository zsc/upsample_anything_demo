import os
import json
import time
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.features import extract_features
from core.gs_jbu import GSJBUParams, render_gsjbu
from core.metrics import l1_loss, psnr
from core.preprocess import downsample, load_image, resize_to_divisible
from core.segmentation import label_map_to_rgb, run_mask2former_seg
from core.tto import optimize_params
from core.viz import normalize_map_to_rgb, pca_to_rgb, to_base64_png

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_config(config_str: str | None) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "target": 256,
        "stride": 16,
        "iters": 50,
        "radius": 2,
        "lr": 1e-3,
        "use_gpu": True,
        "use_mask2former": False,
        "seg_resolution": 192,
        "chunk": 16384,
    }
    if not config_str:
        return defaults
    try:
        user_cfg = json.loads(config_str)
    except json.JSONDecodeError:
        return defaults
    defaults.update(user_cfg)
    return defaults


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/run")
async def run_demo(
    image: UploadFile = File(...),
    config: str = Form(None),
) -> JSONResponse:
    cfg = parse_config(config)
    target = int(cfg["target"])
    stride = int(cfg["stride"])
    iters = int(cfg["iters"])
    radius = int(cfg["radius"])
    lr = float(cfg["lr"])
    use_gpu = bool(cfg["use_gpu"])
    use_mask2former = bool(cfg.get("use_mask2former", False))
    seg_resolution = int(cfg.get("seg_resolution", 192))
    chunk = int(cfg.get("chunk", 16384))

    device = get_device(use_gpu)
    t0 = time.time()

    img_bytes = await image.read()
    I_hr = load_image(img_bytes)
    I_hr = resize_to_divisible(I_hr, target=target, stride=stride).to(device)
    I_lr = downsample(I_hr, stride=stride)

    H, W = I_hr.shape[-2:]
    H_lr, W_lr = I_lr.shape[-2:]

    print(f"[Device] {device} | HR {H}x{W} | LR {H_lr}x{W_lr}")

    I_lr_up_bilinear = F.interpolate(I_lr, size=(H, W), mode="bilinear", align_corners=False)
    I_lr_up_bicubic = F.interpolate(I_lr, size=(H, W), mode="bicubic", align_corners=False)

    fixed_params = GSJBUParams(H_lr, W_lr, device=device)
    with torch.no_grad():
        I_hat_fixed = render_gsjbu(I_lr, I_hr, I_lr, fixed_params, stride=stride, radius=radius, chunk=chunk)

    if iters > 0:
        params, I_hat_tto, metrics = optimize_params(
            I_hr=I_hr,
            I_lr=I_lr,
            iters=iters,
            lr=lr,
            radius=radius,
            chunk=chunk,
        )
    else:
        params = fixed_params
        I_hat_tto = I_hat_fixed
        metrics = {
            "l1_recon": l1_loss(I_hat_tto, I_hr).item(),
            "psnr_recon": psnr(I_hat_tto, I_hr),
        }

    err_map = (I_hat_tto - I_hr).abs().mean(dim=1, keepdim=True)

    t_stage_a = time.time() - t0

    with torch.no_grad():
        F_lr = extract_features(I_hr, out_hw=(H_lr, W_lr))
        F_hr_bilinear = F.interpolate(F_lr, size=(H, W), mode="bilinear", align_corners=False)
        F_hr_gsjbu = render_gsjbu(F_lr, I_hr, I_lr, params, stride=stride, radius=radius, chunk=chunk)

    t_stage_b = time.time() - t0 - t_stage_a

    seg_status = "disabled"
    seg_images: Dict[str, str] = {}
    if use_mask2former:
        try:
            label_map = run_mask2former_seg(I_hr, seg_resolution, device)
            seg_status = "ok"
        except Exception as exc:
            if device.type != "cpu":
                try:
                    label_map = run_mask2former_seg(I_hr, seg_resolution, torch.device("cpu"))
                    seg_status = "ok (cpu fallback)"
                except Exception as cpu_exc:
                    label_map = None
                    seg_status = f"error: {cpu_exc}"
            else:
                label_map = None
                seg_status = f"error: {exc}"

        if label_map is not None:
            label_lr = F.interpolate(
                label_map.unsqueeze(0).unsqueeze(0).float(),
                size=(H_lr, W_lr),
                mode="nearest",
            ).long()[0, 0]
            seg_lr_rgb = label_map_to_rgb(label_lr)
            seg_lr_up = F.interpolate(seg_lr_rgb, size=(H, W), mode="nearest")
            seg_hr_bilinear = F.interpolate(seg_lr_rgb, size=(H, W), mode="bilinear", align_corners=False)
            seg_hr_gsjbu = render_gsjbu(seg_lr_rgb.to(device), I_hr, I_lr, params, stride=stride, radius=radius, chunk=chunk).cpu()

            seg_images = {
                "seg_lr": to_base64_png(seg_lr_up),
                "seg_hr_bilinear": to_base64_png(seg_hr_bilinear),
                "seg_hr_gsjbu": to_base64_png(seg_hr_gsjbu),
            }

    with torch.no_grad():
        feat_lr_pca = pca_to_rgb(F_lr)
        feat_lr_pca = F.interpolate(feat_lr_pca.to(device), size=(H, W), mode="bilinear", align_corners=False).cpu()
        feat_hr_bilinear_pca = pca_to_rgb(F_hr_bilinear)
        feat_hr_gsjbu_pca = pca_to_rgb(F_hr_gsjbu)

        sx, sy, theta, sr = params.values()
        theta_map = (theta + torch.pi) / (2 * torch.pi)
        sigma_x = normalize_map_to_rgb(sx, clamp=(1.0, 64.0))
        sigma_y = normalize_map_to_rgb(sy, clamp=(1.0, 64.0))
        sigma_r = normalize_map_to_rgb(sr, clamp=(0.01, 1.0))
        theta_viz = normalize_map_to_rgb(theta_map, clamp=(0.0, 1.0))

    images = {
        "I_hr": to_base64_png(I_hr),
        "I_lr": to_base64_png(I_lr),
        "I_lr_up_bilinear": to_base64_png(I_lr_up_bilinear),
        "I_lr_up_bicubic": to_base64_png(I_lr_up_bicubic),
        "I_hat_fixed": to_base64_png(I_hat_fixed),
        "I_hat_tto": to_base64_png(I_hat_tto),
        "err_abs": to_base64_png(normalize_map_to_rgb(err_map)),
        "feat_lr_pca": to_base64_png(feat_lr_pca),
        "feat_hr_bilinear_pca": to_base64_png(feat_hr_bilinear_pca),
        "feat_hr_gsjbu_pca": to_base64_png(feat_hr_gsjbu_pca),
        "sigma_x": to_base64_png(sigma_x),
        "sigma_y": to_base64_png(sigma_y),
        "theta": to_base64_png(theta_viz),
        "sigma_r": to_base64_png(sigma_r),
    }
    images.update(seg_images)

    meta = {
        "H": H,
        "W": W,
        "stride": stride,
        "iters": iters,
        "radius": radius,
        "device": str(device),
        "time_stage_a": round(t_stage_a, 3),
        "time_stage_b": round(t_stage_b, 3),
        "seg_status": seg_status,
    }

    return JSONResponse({"meta": meta, "images": images, "metrics": metrics})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
