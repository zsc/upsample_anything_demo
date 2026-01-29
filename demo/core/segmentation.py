import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

import numpy as np
from PIL import Image
import torch

_MODEL_ID = "facebook/mask2former-swin-base-coco-panoptic"
_MODEL = None
_PROCESSOR = None


def _to_pil(I_hr: torch.Tensor, size: int | None = None) -> Image.Image:
    img = I_hr.detach().float().cpu()
    if img.ndim == 4:
        img = img[0]
    img = img.clamp(0.0, 1.0)
    arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    pil = Image.fromarray(arr)
    if size is not None:
        pil = pil.resize((size, size), Image.BILINEAR)
    return pil


def _load_model(device: torch.device):
    global _MODEL, _PROCESSOR
    if _MODEL is None or _PROCESSOR is None:
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
        os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

        _PROCESSOR = Mask2FormerImageProcessor.from_pretrained(_MODEL_ID)
        _MODEL = Mask2FormerForUniversalSegmentation.from_pretrained(_MODEL_ID)
        _MODEL.eval()
    _MODEL.to(device)
    return _MODEL, _PROCESSOR


@torch.no_grad()
def run_mask2former_seg(
    I_hr: torch.Tensor,
    seg_size: int,
    device: torch.device,
) -> torch.Tensor:
    model, processor = _load_model(device)
    pil = _to_pil(I_hr, size=seg_size)
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    segs = processor.post_process_semantic_segmentation(outputs, target_sizes=[(seg_size, seg_size)])
    seg = segs[0].to("cpu")
    return seg.long()


def label_map_to_rgb(label: torch.Tensor, num_colors: int = 256) -> torch.Tensor:
    if label.ndim != 2:
        raise ValueError("label map must be 2D")

    label = label.detach().cpu()
    palette = torch.empty((num_colors, 3), dtype=torch.uint8)
    for i in range(num_colors):
        palette[i, 0] = (i * 37) % 255
        palette[i, 1] = (i * 59) % 255
        palette[i, 2] = (i * 83) % 255

    idx = (label % num_colors).to(torch.long)
    rgb = palette[idx]
    rgb = rgb.numpy().astype(np.uint8)
    rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return rgb.unsqueeze(0)
