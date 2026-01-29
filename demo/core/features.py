from typing import Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_features(
    I_hr: torch.Tensor,
    out_hw: Tuple[int, int] | None = None,
) -> torch.Tensor:
    device = I_hr.device
    try:
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    except Exception:
        from torchvision.models import resnet18

        model = resnet18(weights=None)

    model.eval().to(device)
    trunk = torch.nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
    ).to(device)
    feats = trunk(I_hr)
    if out_hw is None:
        return feats
    return F.interpolate(feats, size=out_hw, mode="bilinear", align_corners=False)
