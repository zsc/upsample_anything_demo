import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
import torch.nn.functional as F

from core.gs_jbu import GSJBUParams, render_gsjbu
from core.preprocess import downsample
from core.segmentation import label_map_to_rgb, run_mask2former_seg


class SegUpsampleTest(unittest.TestCase):
    def test_segmentation_upsampling_shapes(self) -> None:
        torch.manual_seed(0)
        H = 64
        W = 64
        stride = 8
        radius = 2
        seg_res = 32

        I_hr = torch.rand(1, 3, H, W)
        I_lr = downsample(I_hr, stride=stride)

        label_map = torch.randint(0, 20, (seg_res, seg_res))
        label_lr = F.interpolate(
            label_map.unsqueeze(0).unsqueeze(0).float(),
            size=I_lr.shape[-2:],
            mode="nearest",
        ).long()[0, 0]

        seg_lr_rgb = label_map_to_rgb(label_lr)
        params = GSJBUParams(I_lr.shape[-2], I_lr.shape[-1])

        seg_hr_bilinear = F.interpolate(seg_lr_rgb, size=(H, W), mode="bilinear", align_corners=False)
        seg_hr_gsjbu = render_gsjbu(seg_lr_rgb, I_hr, I_lr, params, stride=stride, radius=radius, chunk=1024)

        self.assertEqual(tuple(seg_lr_rgb.shape), (1, 3, I_lr.shape[-2], I_lr.shape[-1]))
        self.assertEqual(tuple(seg_hr_bilinear.shape), (1, 3, H, W))
        self.assertEqual(tuple(seg_hr_gsjbu.shape), (1, 3, H, W))
        self.assertFalse(torch.isnan(seg_hr_gsjbu).any())

        self.assertGreaterEqual(seg_hr_gsjbu.min().item(), -1e-6)
        self.assertLessEqual(seg_hr_gsjbu.max().item(), 1.0 + 1e-6)

    @unittest.skipUnless(
        os.environ.get("RUN_MASK2FORMER_TEST") == "1",
        "Set RUN_MASK2FORMER_TEST=1 to run the heavy Mask2Former integration test.",
    )
    def test_mask2former_inference(self) -> None:
        torch.manual_seed(0)
        H = 64
        W = 64
        seg_res = 64
        device = torch.device("cpu")

        I_hr = torch.rand(1, 3, H, W)
        seg = run_mask2former_seg(I_hr, seg_res, device)

        self.assertEqual(tuple(seg.shape), (seg_res, seg_res))
        self.assertEqual(seg.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
