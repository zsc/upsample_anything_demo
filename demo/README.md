# Upsample Anything Demo（GS-JBU + TTO）

本项目是 *Upsample Anything* 论文核心思想的可交互演示：
先在 test-time 仅用 RGB 重建优化各向异性核参数（TTO），再将同一组核权重迁移到特征图做“纯混合”上采样（no value synthesis）。

## 启动方式

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
- `Mask2Former Resolution`：低分辨率分割的输入尺寸（越小越快）。

## 性能建议

- CPU 模式下建议降低 `iters` 或分辨率（例如 224 或 256）。
- `radius` 越大越慢，默认 2 较平衡。
- 若设备支持 MPS/GPU，可勾选 “Use GPU / MPS”。
- Mask2Former 分割可在低分辨率运行（例如 128/192），但首次会下载权重。

## 常见问题

- **权重下载失败**：resnet18 无法下载时会自动使用随机权重（仍可运行，仅视觉差异更大）。
- **Mask2Former 权重下载失败**：会在 UI 中显示 seg status，或自动回退到 CPU 尝试。
- **太慢**：降低分辨率 / `iters` / `radius`。

## 算法背景（简要）

本 Demo 实现的是 GS-JBU + TTO 思路：先在 RGB 上优化每个 LR 像素的各向异性核参数，再把同一组权重迁移到特征图上做纯混合上采样（不生成新值）。

### 参数与约束（逐 LR 像素）

每个 LR 像素 q 有 4 个可学习参数：

- 空间尺度：`σx[q], σy[q]`
- 旋转角：`θ[q]`
- range 尺度：`σr[q]`

使用重参数化并 clamp 保证数值稳定：

```
σx = softplus(raw_sx) + eps,  σy = softplus(raw_sy) + eps
σr = softplus(raw_sr) + eps
θ  = π * tanh(raw_theta)

σx, σy ∈ [1, 64],  σr ∈ [0.01, 1.0]
```

### 权重计算（对每个 HR 像素 p）

对 HR 像素 p，其 LR 邻域为 Ω(p)，对每个候选 q 计算：

```
Δx = (px + 0.5) - (qx + 0.5) * s
Δy = (py + 0.5) - (qy + 0.5) * s

dx' =  cos(θ[q]) * Δx + sin(θ[q]) * Δy
dy' = -sin(θ[q]) * Δx + cos(θ[q]) * Δy

log_ws = -0.5 * ((dx'/σx[q])^2 + (dy'/σy[q])^2)
log_wr = -||Ip - Iq||^2 / (2 * σr[q]^2)
log_w  = log_ws + log_wr

w = softmax(log_w over q ∈ Ω(p))   (log-sum-exp 稳定化)
```

其中 `Ip` 为 HR 像素 RGB，`Iq` 为 LR 像素 RGB。

### 渲染（纯混合）

对任意 LR 信号 `V_lr`（RGB 或特征），输出为：

```
V_hat(p) = Σ_{q∈Ω(p)} w(p←q) * V_lr(q)
```

### TTO 目标

仅在 RGB 重建上优化参数：

```
L = mean( |I_hat_hr - I_hr| )
```

优化完成后，将同一组权重用于特征上采样对比（bilinear vs GS-JBU）。

## Mask2Former 分割上采样（低分辨率示例）

当勾选 “Run Mask2Former (low-res)” 时，系统会在低分辨率上运行 Mask2Former，得到语义分割结果，
再把分割图作为 `V_lr` 进行上采样，对比 bilinear vs GS-JBU 的边缘对齐效果。

流程要点：

```
I_hr --(resize to seg_resolution)--> Mask2Former --> seg_map
seg_map --(nearest)-> seg_lr --(GS-JBU / bilinear)-> seg_hr
```

## 开发状态

当前为 MVP 可跑通版本，包含：上传图片、TTO 优化、固定 JBU、特征上采样对比、参数/误差可视化与基础指标输出。

## TODO

- 增加可选的进度回传（SSE 或轮询 /api/progress）。
- 增加相似度热力图交互（点选像素，展示 bilinear vs GS-JBU）。
- 增加 `return_files=true` 写入 `static/out/` 的大图模式。
- 增加更多可视化（例如核参数直方图、迭代曲线）。
- 针对 CPU 做更细的默认参数建议与自动降参提示。
