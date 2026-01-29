# Upsample Anything Demo（GS-JBU + TTO）

本项目是 *Upsample Anything* 论文核心思想的可交互演示：  
先在 test-time 仅用 RGB 重建优化各向异性核参数（TTO），再将同一组核权重迁移到特征图上做“纯混合”上采样（no value synthesis）。

项目代码与可运行 Demo 位于 `demo/` 目录。

## 安装与启动

```bash
cd demo
pip install -r requirements.txt
python app.py
```

浏览器打开：`http://127.0.0.1:8000`

## 参数说明

- `stride`：下采样步长 s（决定 LR 分辨率）。
- `iters`：TTO 迭代次数，设为 0 表示不做 TTO，仅使用固定核。
- `radius`：邻域窗口半径 R，窗口大小为 `(2R+1)^2`。

## 性能建议

- CPU 模式下建议降低 `iters` 或分辨率（例如 224 或 256）。
- `radius` 越大越慢，默认 2 较平衡。
- 若设备支持 MPS/GPU，可在界面中启用加速选项。

## 常见问题

- **权重下载失败**：resnet18 无法下载时会自动使用随机权重（仍可运行，仅视觉差异更大）。
- **太慢**：降低分辨率 / `iters` / `radius`。

## 算法简介

GS-JBU + TTO 思路：先在 RGB 上优化每个 LR 像素的各向异性核参数，再把同一组权重迁移到特征图上做纯混合上采样（不生成新值）。

若需详细设计与实现说明，请查看 `demo/README.md`。
