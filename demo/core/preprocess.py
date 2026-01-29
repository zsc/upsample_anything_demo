import io
from typing import Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def load_image(data: Union[bytes, bytearray]) -> torch.Tensor:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


def resize_to_divisible(I: torch.Tensor, target: int, stride: int) -> torch.Tensor:
    desired = int(target)
    if desired % stride != 0:
        desired = max(stride, (desired // stride) * stride)
    if I.shape[-2:] != (desired, desired):
        I = F.interpolate(I, size=(desired, desired), mode="bilinear", align_corners=False)
    return I


def downsample(I_hr: torch.Tensor, stride: int) -> torch.Tensor:
    H, W = I_hr.shape[-2:]
    return F.interpolate(I_hr, size=(H // stride, W // stride), mode="bilinear", align_corners=False)
