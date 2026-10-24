const MAX_SIDE = 2000;
const MAX_BYTES = 15 * 2 ** 20;

const $ = (id) => document.getElementById(id);
const fileInput = $("file"), dropzone = $("dropzone"), go = $("go"), status = $("status");
const before = $("before"), after = $("after"), slider = $("slider"), divider = $("divider");

let file = null;
let urls = [];

function setStatus(msg, isError = false) {
  status.textContent = msg;
  status.classList.toggle("error", isError);
}

function loadDims(f) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(f);
    const img = new Image();
    img.onload = () => resolve({ url, w: img.naturalWidth, h: img.naturalHeight });
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("Could not read image")); };
    img.src = url;
  });
}

async function choose(f) {
  if (!f) return;
  file = null;
  go.disabled = true;
  if (!/^image\/(png|jpeg|webp)$/.test(f.type)) return setStatus("Use a PNG, JPEG or WebP image.", true);
  if (f.size > MAX_BYTES) return setStatus("File is larger than 15 MB.", true);
  try {
    const { url, w, h } = await loadDims(f);
    if (Math.max(w, h) > MAX_SIDE) {
      URL.revokeObjectURL(url);
      return setStatus(`Image is ${w}×${h}. The long side must be ${MAX_SIDE} px or less.`, true);
    }
    urls.forEach(URL.revokeObjectURL);
    urls = [url];
    before.src = url;
    file = f;
    dropzone.classList.add("has-file");
    dropzone.querySelector(".dz-title").textContent = f.name;
    dropzone.querySelector(".dz-hint").textContent = `${w}×${h} · click to change`;
    go.disabled = false;
    setStatus("");
  } catch (e) {
    setStatus(e.message, true);
  }
}

async function upscale() {
  const model = document.querySelector("input[name=model]:checked").value;
  const format = document.querySelector("input[name=format]:checked").value;
  const body = new FormData();
  body.append("file", file);
  body.append("model", model);
  body.append("format", format);

  go.disabled = true;
  setStatus("Upscaling… the first request after a quiet period takes longer while the GPU starts.");
  const t0 = performance.now();
  try {
    const res = await fetch(`${window.UPSCALE_API}/upscale`, { method: "POST", body });
    if (!res.ok) {
      let detail = `Request failed (${res.status})`;
      try { detail = (await res.json()).detail || detail; } catch {}
      throw new Error(detail);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    urls.push(url);
    after.src = url;

    const size = res.headers.get("X-Output-Size");
    const gpu = res.headers.get("X-Upscale-Seconds");
    const total = ((performance.now() - t0) / 1000).toFixed(1);
    $("meta").textContent = [size, gpu && `${gpu} s on GPU`, `${total} s total`, `${(blob.size / 2 ** 20).toFixed(1)} MB`]
      .filter(Boolean).join(" · ");

    const base = file.name.replace(/\.[^.]+$/, "");
    const dl = $("download");
    dl.href = url;
    dl.download = `${base}_${model}.${format === "jpeg" ? "jpg" : "png"}`;

    $("result-panel").hidden = false;
    setPosition(50);
    setStatus("");
    $("result-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    setStatus(e.message === "Failed to fetch" ? "Could not reach the upscaling service." : e.message, true);
  } finally {
    go.disabled = !file;
  }
}

function setPosition(p) {
  slider.value = p;
  before.style.clipPath = `inset(0 ${100 - p}% 0 0)`;
  divider.style.left = `${p}%`;
}

fileInput.addEventListener("change", () => choose(fileInput.files[0]));
["dragenter", "dragover"].forEach((t) => dropzone.addEventListener(t, (e) => { e.preventDefault(); dropzone.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => dropzone.addEventListener(t, (e) => { e.preventDefault(); dropzone.classList.remove("over"); }));
dropzone.addEventListener("drop", (e) => choose(e.dataTransfer.files[0]));
go.addEventListener("click", upscale);
slider.addEventListener("input", () => setPosition(+slider.value));
