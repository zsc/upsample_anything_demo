# Upsample Anything (GS-JBU + TTO) Python + HTML Demo — SPEC

> 目标：做一个可本地运行的 **Python(FastAPI) + HTML/JS** 交互式 Demo，演示论文 *Upsample Anything* 的核心：  
> **先在 test-time 仅用 RGB 重建去优化每个 LR 像素的各向异性核参数（TTO），再把同一组核权重迁移到特征图上做“纯混合”上采样（no value synthesis）**。  
> 该 SPEC 设计为可直接交给 **gemini-cli / codex** 生成代码。

---

## 1. 范围与目标

### 1.1 必做目标（MVP）
1. Web 页面上传一张图片（RGB）。
2. 用户可设置：`stride(s)`、`iters`、`radius(R)`、学习率等。
3. 后端执行两阶段：
   - **Stage A (TTO)**：仅用图像重建损失优化参数 `σx, σy, θ, σr`（逐 LR 像素）。
   - **Stage B (Rendering)**：用学到的权重核把 **LR feature** 上采样到 HR。
4. 前端展示（至少）：
   - 原图 `I_hr`
   - 下采样 `I_lr`（以及其 bilinear 上采样到 HR 的对比）
   - TTO 重建结果 `Î_hr`
   - `|Î_hr - I_hr|` 误差图（可视化）
   - Feature upsampling 对比：`bilinear(F_lr)` vs `GS-JBU(F_lr)`
5. 提供至少 2 个 baseline：
   - Bilinear / Bicubic 上采样
   - 固定参数的 Joint Bilateral（等价：固定各向同性高斯 + 固定 σr）**不做 TTO**

### 1.2 非目标（本 Demo 不做）
- 不追求完全复现论文全部 benchmark 或速度（但要合理优化，默认图像不应卡死）。
- 不实现复杂的异步队列/分布式（可选做轻量进度回传）。
- 不要求下载超大模型；特征提取默认用轻量 CNN 或“伪特征”。

---

## 2. 用户体验与交互（前端）

### 2.1 页面布局（建议单页）
- 左侧：控制面板
  - 上传图片按钮
  - 选择预设分辨率（例如：`224`, `256`, `384`；默认 `256`）
  - 参数：
    - `stride s`：`8/14/16/32`（默认 `16`）
    - `iters`：`0~100`（默认 `50`，`0` 表示不做 TTO）
    - `radius R`：`1~4`（默认 `2`，窗口大小 `K=(2R+1)^2`）
    - `lr`：默认 `1e-3`
    - `use_gpu`：自动检测后可切换
  - 特征提取模式：
    - `fake`：把 `I_lr` 经过 `1x1 conv`/线性投影当作 feature（最快，无需权重下载）
    - `resnet18_layer2`：torchvision resnet18 截取中间层特征（若下载权重失败则 fallback 到 fake）
  - 按钮：`Run`
- 右侧：结果展示（建议 tab）
  - Tab A: Image Reconstruction
    - 原图、下采样、Bilinear、(可选 JBU fixed)、TTO 重建、误差图
  - Tab B: Feature Upsampling
    - LR feature 可视化（PCA->RGB）
    - bilinear(F_lr) PCA
    - GS-JBU(F_lr) PCA
    - (可选) Feature similarity map：点击图像某点，显示与该点的 cosine-sim 热力图（bilinear vs GS-JBU）

### 2.2 前端技术要求
- 纯 HTML + 原生 JS（或极少量依赖），不引入重框架。
- 图片显示用 `<img>`（后端返回 base64 PNG 或静态文件路径均可）。
- 参数控件用 `<input type="range">` / `<select>`。
- 可选：显示进度条（若实现 `/api/progress` 或 SSE）。

---

## 3. 后端架构（Python / FastAPI）

### 3.1 依赖（requirements.txt）
- `fastapi`
- `uvicorn[standard]`
- `pydantic`
- `numpy`
- `pillow`
- `torch`（CPU/GPU/MPS）
    - 运行在 Apple Silicon MPS 上时尽量利用加速
- `torchvision`（可选，用于 resnet18 特征）
- （可选）`scikit-learn`（PCA；也可自己用 torch SVD 实现以减少依赖）

> 注意：Demo 要保证无 GPU/MPS 也能跑（速度会慢，但默认 256^2 + R=2 + iters=50 应可接受）。

### 3.2 文件结构（建议）
```

demo/
app.py
requirements.txt
README.md
core/
preprocess.py
gs_jbu.py
tto.py
features.py
viz.py
metrics.py
static/
index.html
app.js
styles.css

````

### 3.3 API 设计

#### GET `/`
- 返回 `static/index.html`

#### POST `/api/run`
- `multipart/form-data`
  - `image`: 上传的 RGB 图片
  - `config`: JSON 字符串（或拆成普通表单字段也可）
- 返回 JSON：
```json
{
  "meta": {
    "H": 256, "W": 256, "stride": 16, "iters": 50, "radius": 2,
    "device": "mps" 
  },
  "images": {
    "I_hr": "data:image/png;base64,...",
    "I_lr_up_bilinear": "data:image/png;base64,...",
    "I_hat_tto": "data:image/png;base64,...",
    "err_abs": "data:image/png;base64,...",
    "feat_lr_pca": "data:image/png;base64,...",
    "feat_hr_bilinear_pca": "data:image/png;base64,...",
    "feat_hr_gsjbu_pca": "data:image/png;base64,...",
    "sigma_x": "data:image/png;base64,...",
    "sigma_y": "data:image/png;base64,...",
    "theta": "data:image/png;base64,...",
    "sigma_r": "data:image/png;base64,..."
  },
  "metrics": {
    "l1_recon": 0.0123,
    "psnr_recon": 28.7
  }
}
````

> 可选扩展：支持 `return_files=true` 时写入 `static/out/` 并返回 URL，避免超大 base64。

#### （可选）POST `/api/similarity`

* 输入：用户点击点 `(x,y)` + 指定模式（bilinear vs gsjbu）
* 输出：相似度热力图（base64 png）

---

## 4. 核心算法 SPEC（GS-JBU + TTO）

### 4.1 张量与坐标约定

* `I_hr`: `float32`，shape `[1,3,H,W]`，范围 `[0,1]`
* `stride = s`
* `H_lr = H // s`, `W_lr = W // s`
* `I_lr`: 由 `I_hr` **bilinear 下采样**到 `[1,3,H_lr,W_lr]`
* 像素中心坐标：

  * HR 像素 `(py, px)` 的中心为 `(py + 0.5, px + 0.5)`（单位：HR 像素）
  * LR 像素 `(qy, qx)` 对应 HR 中心 `μ_q = ((qy + 0.5)*s, (qx + 0.5)*s)`

### 4.2 参数（逐 LR 像素）

对每个 LR 像素 q 学习 4 个参数：

* `σx[q], σy[q]`：空间各向异性高斯的长短轴尺度（>0）
* `θ[q]`：旋转角（弧度，建议限制在 `[-π, π]`）
* `σr[q]`：range kernel（颜色相似性）尺度（>0）

#### 4.2.1 可训练参数的数值约束（强制）

使用 re-parameterization：

* `σx = softplus(raw_sx) + eps`
* `σy = softplus(raw_sy) + eps`
* `σr = softplus(raw_sr) + eps`
* `θ = π * tanh(raw_theta)`  （将角度压到 `[-π, π]`）

并且在 forward 中 **clamp** 避免极端值：

* `σx, σy`: clamp 到 `[1.0, 64.0]`（可调）
* `σr`: clamp 到 `[0.01, 1.0]`（可调）

#### 4.2.2 初始化（默认）

* `σx = σy = 16.0`
* `σr = 0.12`
* `θ = 0.0`

> 初始化通过设置 `raw_*` 使其映射到上述值。

### 4.3 权重计算（对每个 HR 像素 p）

对 HR 像素 p，构造其 LR 邻域集合 `Ω(p)`：

* 先把 HR 中心坐标映射到 LR 连续坐标：

  * `u = (py + 0.5)/s - 0.5`
  * `v = (px + 0.5)/s - 0.5`
* `q0 = (floor(u), floor(v))`
* 在窗口 `dy,dx ∈ [-R, R]` 构造候选 q：`q = q0 + (dy,dx)`，并 clamp 到图像范围。

对每个候选 q，计算：

1. 空间项（各向异性高斯，使用旋转简化形式）

* `Δx = (px + 0.5) - (qx + 0.5)*s`
* `Δy = (py + 0.5) - (qy + 0.5)*s`
* 旋转：

  * `c = cos(θ[q])`, `t = sin(θ[q])`
  * `dxp =  c*Δx + t*Δy`
  * `dyp = -t*Δx + c*Δy`
* 空间 log-weight：

  * `log_ws = -0.5 * ((dxp/σx[q])^2 + (dyp/σy[q])^2)`

2. range 项（颜色相似性，引导来自 RGB）

* 取颜色：

  * `Ip = I_hr[:, :, py, px]`（shape `[1,3]`）
  * `Iq = I_lr[:, :, qy, qx]`（shape `[1,3]`；固定不参与梯度）
* `diff2 = sum((Ip - Iq)^2 over channel)`
* `log_wr = - diff2 / (2 * σr[q]^2)`

3. 合并并归一化（**必须用 log-space 稳定化**）

* `log_w = log_ws + log_wr`
* `w = softmax(log_w over q ∈ Ω(p))`

### 4.4 渲染（纯混合，不生成新值）

对任意 LR 信号 `V_lr`（可为 RGB 或 feature）：

* `V_hat(p) = Σ_{q∈Ω(p)} w(p←q) * V_lr(q)`

> 注：这就是 demo 要强调的 **no value synthesis**：输出仅是 LR 值的加权混合。

---

## 5. Stage A: Test-Time Optimization (TTO)

### 5.1 目标函数

* 输入：`I_hr`（固定）
* 构造：`I_lr = downsample_bilinear(I_hr, scale=1/s)`（固定）
* 用 GS-JBU 以 `I_lr` 作为 `V_lr` 渲染 `Î_hr`
* 损失（默认）：

  * `L = mean(abs(Î_hr - I_hr))`  （L1）
* （可选）加入轻量正则避免核崩坏（默认关闭，必要时启用）：

  * `λ1 * mean(|σx-σx0| + |σy-σy0|)`
  * `λ2 * mean(|θ|)`（或 TV 正则）
  * `λ3 * mean(|σr-σr0|)`

### 5.2 优化器与默认超参

* Optimizer: `Adam`
* `lr=1e-3`
* `iters=50`（可调）
* 设备：优先 `mps`，否则 `cpu`

### 5.3 性能实现要求（必须）

* 不允许构造 `[H,W,H_lr,W_lr]` 的全量权重（会爆内存）。
* 必须基于 **局部窗口 K=(2R+1)^2** 做向量化计算。
* 建议实现为：

  * 将 HR 像素展平为 `N = H*W`
  * 对每个 offset (dy,dx) 计算一个候选 q 的贡献，最终在 K 维度 softmax
  * 支持按 chunk 处理（例如每次处理 `N_chunk=16384`）以控内存。

### 5.4 参考伪代码（必须按此思路实现）

```python
def render_gsjbu(V_lr, I_hr, I_lr, params, stride, R, chunk=16384):
    # V_lr: [1,C,H_lr,W_lr] (C=3 for image, C=F for features)
    # I_hr: [1,3,H,W]
    # I_lr: [1,3,H_lr,W_lr]
    # returns: V_hat: [1,C,H,W]

    H, W = I_hr.shape[-2:]
    N = H * W
    C = V_lr.shape[1]
    V_hat = zeros([1,C,H,W])

    # build flattened HR pixel center coords (py,px)
    # process in chunks over flattened pixels
    for start in range(0, N, chunk):
        end = min(start+chunk, N)
        py, px = flat_index_to_coords(start:end, W)  # each shape [M]
        # continuous LR coords u,v
        u = (py + 0.5)/stride - 0.5
        v = (px + 0.5)/stride - 0.5
        qy0 = floor(u).long()
        qx0 = floor(v).long()

        # gather K neighbors
        logw_list = []
        val_list = []
        for dy in range(-R, R+1):
          for dx in range(-R, R+1):
            qy = clamp(qy0+dy, 0, H_lr-1)
            qx = clamp(qx0+dx, 0, W_lr-1)

            # fetch params at q
            sx, sy, th, sr = params_at(qy,qx)

            # delta in HR units (center-to-center)
            Dx = (px + 0.5) - (qx + 0.5)*stride
            Dy = (py + 0.5) - (qy + 0.5)*stride

            # rotate
            c = cos(th); t = sin(th)
            dxp =  c*Dx + t*Dy
            dyp = -t*Dx + c*Dy

            log_ws = -0.5*((dxp/sx)**2 + (dyp/sy)**2)

            Ip = I_hr[..., py, px]           # [1,3,M]
            Iq = I_lr[..., qy, qx]           # [1,3,M]
            diff2 = ((Ip-Iq)**2).sum(dim=1)  # [1,M]
            log_wr = - diff2 / (2*(sr**2))

            logw = log_ws + log_wr           # [1,M]
            logw_list.append(logw)

            Vq = V_lr[..., qy, qx]           # [1,C,M]
            val_list.append(Vq)

        # stack to [K, 1, M]
        logW = stack(logw_list, dim=0)
        # stable softmax across K
        logW = logW - logW.max(dim=0, keepdim=True).values
        Wk = softmax(logW, dim=0)            # [K,1,M]

        Vk = stack(val_list, dim=0)          # [K,1,C,M]
        out = (Wk.unsqueeze(2) * Vk).sum(dim=0)  # [1,C,M]

        V_hat[..., py, px] = out

    return V_hat
```

---

## 6. Stage B: Feature Upsampling

### 6.1 特征定义（两种模式）

#### A) fake（默认，必做）

* `F_lr = Conv1x1(I_lr)` 或 `F_lr = linear(project I_lr)`
* 输出通道数：`C=32`（可配）
* 这保证 demo 无需下载任何模型也能演示 feature 的“边缘对齐效果”。

#### B) resnet18_layer2（可选，但推荐）

* 用 torchvision resnet18（能下载就用预训练，不能就随机权重或 fallback）。
* 截取某层输出作为 feature（例如 layer2 输出）。
* 将其通过 `adaptive_avg_pool` 或插值调整到 `H_lr, W_lr` 作为 `F_lr`（确保 stride 对齐，或在 UI 提示 stride 与 extractor 的实际下采样不一致时做 resize 适配）。

### 6.2 特征可视化（必做）

* 对 `F`（shape `[1,C,H,W]`）做 PCA 到 3 通道并归一化到 `[0,1]`，输出 RGB。
* 至少输出：

  * `feat_lr_pca`（将 LR PCA 图 bilinear 放大到 HR 仅用于展示）
  * `feat_hr_bilinear_pca`
  * `feat_hr_gsjbu_pca`

> PCA 实现要求：固定随机种子；当 C 很大时可随机采样像素做 PCA 以提速。

---

## 7. Baselines（必须实现）

1. `bilinear`：`torch.nn.functional.interpolate(V_lr, size=(H,W), mode="bilinear")`
2. `bicubic`（可选）：`mode="bicubic"`
3. `fixed JBU`：等价于 GS-JBU 但参数固定：

   * `σx=σy=16, θ=0, σr=0.12`（或 UI 可调）
   * 不做 TTO

对比输出至少在 image reconstruction 与 feature upsampling 两个 tab 都展示（若实现了 JBU）。

---

## 8. 指标与日志（最低要求）

* 输出 `l1_recon = mean(|Î_hr - I_hr|)`
* 输出 `psnr_recon`（可简单实现：`-10*log10(mse)`）
* 后端日志打印：

  * device / dtype
  * 每 10 iter 打印 loss
  * 总耗时（stage A / stage B）

---

## 9. 数值稳定性与工程约束（必须遵守）

1. softmax 必须用 log-sum-exp 稳定化（减 max）。
2. `σx, σy, σr` 必须正且 clamp。
3. 输入图像必须 resize 成 `H,W` 可被 stride 整除：

   * 优先策略：`resize` 到用户选定的方形（默认 256）并保证可整除
   * 或者 center crop/pad（任选一种，但要一致）
4. CPU 模式下默认参数要保守（例如 `iters=30` 或提供提示）。
5. chunking 必须存在，避免内存爆炸。

---

## 10. README（必须包含）
* 中文写
* 安装与启动：

  * `pip install -r requirements.txt`
  * `python app.py`
  * 浏览器打开 `http://127.0.0.1:8000`
* 参数解释（stride/iters/radius）
* 性能建议
* 常见问题：

  * 权重下载失败 -> 自动 fallback
  * 太慢 -> 降分辨率/iters/radius

---

## 11. 验收标准（Acceptance Criteria）

1. 启动服务后访问首页能加载 UI。
2. 上传任意 JPG/PNG：

   * 返回并显示：`I_hr`、`I_lr_up_bilinear`、`I_hat_tto`（iters>0）与误差图
3. `iters=0` 时：

   * 不做 TTO，仍输出 fixed JBU 或 bilinear 结果（且不报错）
4. feature tab 至少能展示：

   * bilinear(F_lr) PCA 与 GS-JBU(F_lr) PCA 的差异可见（边缘更清晰/跨边缘污染更少）
5. 后端在 CPU 上默认 256x256、R=2、iters=50 可以在可接受时间内跑完（若过慢，README 要提示如何降参；代码要可运行不崩溃）。

---

## 12. 实现任务拆解（给 codex/gemini 的 TODO）

1. `preprocess.py`

   * `load_image(file)->torch`
   * `resize_to_divisible(I, target=256, stride=s)->I_hr`
   * `downsample(I_hr, s)->I_lr`
2. `gs_jbu.py`

   * 参数类：`GSJBUParams`（raw 参数 + transform）
   * `render_gsjbu(V_lr, I_hr, I_lr, params, s, R, chunk)->V_hat`
3. `tto.py`

   * `optimize_params(I_hr, I_lr, init, iters, lr, R)->params, I_hat, metrics`
4. `features.py`

   * `extract_features(I_hr, I_lr, mode, out_hw=(H_lr,W_lr))->F_lr`
5. `viz.py`

   * `to_base64_png(tensor)`
   * `pca_to_rgb(F)->rgb_tensor`
   * `normalize_map_to_rgb(map)->rgb`
6. `app.py`

   * FastAPI routes
   * 静态文件服务
   * `/api/run` 实现：串起 preprocess -> TTO -> feature render -> 返回 JSON
7. `static/app.js`

   * form 提交 + fetch `/api/run`
   * 渲染 `<img>`、显示 metrics
   * 错误处理（toast 或 alert）
