const imageInput = document.getElementById("imageInput");
const resolution = document.getElementById("resolution");
const stride = document.getElementById("stride");
const iters = document.getElementById("iters");
const radius = document.getElementById("radius");
const lr = document.getElementById("lr");
const segResolution = document.getElementById("segResolution");
const useGpu = document.getElementById("useGpu");
const useMask2Former = document.getElementById("useMask2Former");
const runBtn = document.getElementById("runBtn");
const status = document.getElementById("status");
const metrics = document.getElementById("metrics");

const itersVal = document.getElementById("itersVal");
const radiusVal = document.getElementById("radiusVal");

iters.addEventListener("input", () => {
  itersVal.textContent = iters.value;
});

radius.addEventListener("input", () => {
  radiusVal.textContent = radius.value;
});

function setImage(id, data) {
  const el = document.getElementById(id);
  if (!el) return;
  el.src = data || "";
}

function setMetrics(meta, metricsData) {
  metrics.innerHTML = "";
  if (!metricsData) return;
  const items = [
    `L1: ${metricsData.l1_recon.toFixed(4)}`,
    `PSNR: ${metricsData.psnr_recon.toFixed(2)}dB`,
    `Device: ${meta.device}`,
    `Stage A: ${meta.time_stage_a}s`,
    `Stage B: ${meta.time_stage_b}s`,
  ];
  items.forEach((text) => {
    const div = document.createElement("div");
    div.className = "metric";
    div.textContent = text;
    metrics.appendChild(div);
  });
}

function setStatus(msg) {
  status.textContent = msg;
}

runBtn.addEventListener("click", async () => {
  if (!imageInput.files.length) {
    alert("Please upload an image first.");
    return;
  }

  const config = {
    target: parseInt(resolution.value, 10),
    stride: parseInt(stride.value, 10),
    iters: parseInt(iters.value, 10),
    radius: parseInt(radius.value, 10),
    lr: parseFloat(lr.value),
    use_gpu: useGpu.checked,
    use_mask2former: useMask2Former.checked,
    seg_resolution: parseInt(segResolution.value, 10),
  };

  const form = new FormData();
  form.append("image", imageInput.files[0]);
  form.append("config", JSON.stringify(config));

  runBtn.disabled = true;
  setStatus("Running... This may take a moment.");

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      body: form,
    });
    if (!response.ok) {
      throw new Error(`Server error ${response.status}`);
    }
    const data = await response.json();

    setImage("img-hr", data.images.I_hr);
    setImage("img-lr", data.images.I_lr);
    setImage("img-lr-bilinear", data.images.I_lr_up_bilinear);
    setImage("img-lr-bicubic", data.images.I_lr_up_bicubic);
    setImage("img-fixed", data.images.I_hat_fixed);
    setImage("img-tto", data.images.I_hat_tto);
    setImage("img-err", data.images.err_abs);

    setImage("feat-lr", data.images.feat_lr_pca);
    setImage("feat-bilinear", data.images.feat_hr_bilinear_pca);
    setImage("feat-gsjbu", data.images.feat_hr_gsjbu_pca);

    setImage("param-sx", data.images.sigma_x);
    setImage("param-sy", data.images.sigma_y);
    setImage("param-theta", data.images.theta);
    setImage("param-sr", data.images.sigma_r);

    setMetrics(data.meta, data.metrics);

    if (data.images.seg_lr) {
      setImage("seg-lr", data.images.seg_lr);
      setImage("seg-bilinear", data.images.seg_hr_bilinear);
      setImage("seg-gsjbu", data.images.seg_hr_gsjbu);
    } else {
      setImage("seg-lr", "");
      setImage("seg-bilinear", "");
      setImage("seg-gsjbu", "");
    }

    const segStatus = data.meta?.seg_status ? ` | seg: ${data.meta.seg_status}` : "";
    setStatus(`Done.${segStatus}`);
  } catch (err) {
    console.error(err);
    setStatus("Error: " + err.message);
    alert("Failed to run demo. Check server logs for details.");
  } finally {
    runBtn.disabled = false;
  }
});

const tabs = document.querySelectorAll(".tab-btn");
const panels = document.querySelectorAll(".tab-panel");

tabs.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabs.forEach((t) => t.classList.remove("active"));
    panels.forEach((p) => p.classList.remove("active"));

    btn.classList.add("active");
    const target = document.getElementById(btn.dataset.tab);
    if (target) target.classList.add("active");
  });
});
