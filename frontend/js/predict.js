document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("file");
  const pickBtn = document.getElementById("pick");
  const submitBtn = document.getElementById("submit");
  const preview = document.getElementById("preview");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const statusEl = document.getElementById("status");
  const resultBox = document.getElementById("result");
  const topLabelEl = document.getElementById("top-label");
  const probTableBody = document.querySelector("#prob-table tbody");
  const saveInfo = document.getElementById("save-info");
  const resetCropBtn = document.getElementById("reset-crop");
  const modelInfo = document.getElementById("model-info");
  const loader = document.getElementById("loader");

  const state = {
    img: null,
    objectUrl: null,
    drawInfo: null,
    isDragging: false,
    selection: null, // {x0,y0,x1,y1} in canvas coords
  };

  const BACKEND_BASE = "http://localhost:8001";

  pickBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => updatePreview());
  resetCropBtn.addEventListener("click", () => {
    state.selection = null;
    draw();
  });
  submitBtn.addEventListener("click", () => submitPrediction());

  canvas.addEventListener("mousedown", handlePointerDown);
  canvas.addEventListener("mousemove", handlePointerMove);
  canvas.addEventListener("mouseup", handlePointerUp);
  canvas.addEventListener("mouseleave", handlePointerUp);

  canvas.addEventListener(
    "touchstart",
    (evt) => {
      evt.preventDefault();
      handlePointerDown(evt.touches[0]);
    },
    { passive: false }
  );
  canvas.addEventListener(
    "touchmove",
    (evt) => {
      evt.preventDefault();
      handlePointerMove(evt.touches[0]);
    },
    { passive: false }
  );
  canvas.addEventListener("touchend", (evt) => {
    evt.preventDefault();
    handlePointerUp();
  });
  canvas.addEventListener("touchcancel", (evt) => {
    evt.preventDefault();
    handlePointerUp();
  });

  window.addEventListener("resize", () => {
    fitCanvasToContainer();
    draw();
  });

  fitCanvasToContainer();
  draw();
  fetchInfo();

  function setLoading(isLoading) {
    if (loader) loader.hidden = !isLoading;
    submitBtn.disabled = isLoading;
    pickBtn.disabled = isLoading;
    resetCropBtn.disabled = isLoading;
  }

  function handlePointerDown(evt) {
    if (!state.img) return;
    state.isDragging = true;
    const { x, y } = getCanvasPos(evt);
    state.selection = { x0: x, y0: y, x1: x, y1: y };
    draw();
  }

  function handlePointerMove(evt) {
    if (!state.isDragging || !state.img) return;
    const { x, y } = getCanvasPos(evt);
    state.selection.x1 = x;
    state.selection.y1 = y;
    draw();
  }

  function handlePointerUp() {
    state.isDragging = false;
  }

  function getCanvasPos(evt) {
    const rect = canvas.getBoundingClientRect();
    const clientX = evt.clientX !== undefined ? evt.clientX : evt.pageX;
    const clientY = evt.clientY !== undefined ? evt.clientY : evt.pageY;
    const x = ((clientX - rect.left) / rect.width) * canvas.width;
    const y = ((clientY - rect.top) / rect.height) * canvas.height;
    return { x, y };
  }

  async function fetchInfo() {
    try {
      const res = await fetch(`${BACKEND_BASE}/info`);
      const data = await res.json();
      const name = data.model_name || data.model_path || "Model tidak diketahui";
      const size = data.input_size ? `${data.input_size.width}x${data.input_size.height}` : "N/A";
      modelInfo.textContent = `Model: ${name} • Input: ${size}`;
    } catch (err) {
      modelInfo.textContent = "Model: tidak dapat memuat info (cek backend).";
    }
  }

  function updatePreview() {
    const file = fileInput.files[0];
    if (!file) return;
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = URL.createObjectURL(file);
    const newImg = new Image();
    newImg.onload = () => {
      state.img = newImg;
      state.selection = null;
      fitCanvasToContainer();
      draw();
    };
    newImg.src = state.objectUrl;
  }

  function fitCanvasToContainer() {
    const bounds = preview.getBoundingClientRect();
    canvas.width = Math.max(320, Math.round(bounds.width));
    canvas.height = Math.max(320, Math.round(bounds.width * 0.75));
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!state.img) {
      ctx.fillStyle = "#9bbfaa";
      ctx.font = "16px sans-serif";
      ctx.fillText("Preview gambar daun jagung akan muncul di sini", 16, 32);
      return;
    }
    const scale = Math.min(canvas.width / state.img.width, canvas.height / state.img.height);
    const displayW = state.img.width * scale;
    const displayH = state.img.height * scale;
    const offsetX = (canvas.width - displayW) / 2;
    const offsetY = (canvas.height - displayH) / 2;
    state.drawInfo = { scale, offsetX, offsetY, displayW, displayH };
    ctx.drawImage(state.img, offsetX, offsetY, displayW, displayH);

    if (state.selection) {
      const { x0, y0, x1, y1 } = state.selection;
      const x = Math.min(x0, x1);
      const y = Math.min(y0, y1);
      const w = Math.abs(x1 - x0);
      const h = Math.abs(y1 - y0);
      ctx.fillStyle = "rgba(0,0,0,0.3)";
      ctx.fillRect(0, 0, canvas.width, y);
      ctx.fillRect(0, y, x, h);
      ctx.fillRect(x + w, y, canvas.width - (x + w), h);
      ctx.fillRect(0, y + h, canvas.width, canvas.height - (y + h));
      ctx.strokeStyle = "#facc15";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
    }
  }

  function getSelectionOnOriginal() {
    if (!state.selection || !state.drawInfo || !state.img) return null;
    const { x0, y0, x1, y1 } = state.selection;
    const x = Math.min(x0, x1);
    const y = Math.min(y0, y1);
    const w = Math.abs(x1 - x0);
    const h = Math.abs(y1 - y0);
    const sx = Math.max(x, state.drawInfo.offsetX);
    const sy = Math.max(y, state.drawInfo.offsetY);
    const ex = Math.min(x + w, state.drawInfo.offsetX + state.drawInfo.displayW);
    const ey = Math.min(y + h, state.drawInfo.offsetY + state.drawInfo.displayH);
    if (ex <= sx || ey <= sy) return null;
    const selW = ex - sx;
    const selH = ey - sy;
    const origX = (sx - state.drawInfo.offsetX) / state.drawInfo.scale;
    const origY = (sy - state.drawInfo.offsetY) / state.drawInfo.scale;
    const origW = selW / state.drawInfo.scale;
    const origH = selH / state.drawInfo.scale;
    return { origX, origY, origW, origH };
  }

  async function submitPrediction() {
    const file = fileInput.files[0];
    if (!file) {
      setLoading(false);
      statusEl.textContent = "Pilih gambar terlebih dahulu.";
      statusEl.classList.add("error");
      return;
    }
    setLoading(true);
    statusEl.textContent = "Memproses...";
    statusEl.classList.remove("error");

    if (!state.img) {
      setLoading(false);
      statusEl.textContent = "Preview tidak tersedia.";
      statusEl.classList.add("error");
      return;
    }

    let blob;
    const cropInfo = getSelectionOnOriginal();
    if (cropInfo) {
      const temp = document.createElement("canvas");
      temp.width = cropInfo.origW;
      temp.height = cropInfo.origH;
      const tctx = temp.getContext("2d");
      tctx.drawImage(
        state.img,
        cropInfo.origX,
        cropInfo.origY,
        cropInfo.origW,
        cropInfo.origH,
        0,
        0,
        cropInfo.origW,
        cropInfo.origH
      );
      const finalCanvas = document.createElement("canvas");
      finalCanvas.width = 256;
      finalCanvas.height = 256;
      const fctx = finalCanvas.getContext("2d");
      fctx.drawImage(temp, 0, 0, 256, 256);
      blob = await new Promise((resolve) => finalCanvas.toBlob(resolve, "image/png"));
    } else {
      const finalCanvas = document.createElement("canvas");
      finalCanvas.width = 256;
      finalCanvas.height = 256;
      const fctx = finalCanvas.getContext("2d");
      fctx.drawImage(state.img, 0, 0, 256, 256);
      blob = await new Promise((resolve) => finalCanvas.toBlob(resolve, "image/png"));
    }

    const form = new FormData();
    form.append("file", blob, file.name || "upload.png");

    try {
      const backendUrl = `${BACKEND_BASE}/predict`;
      const res = await fetch(backendUrl, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediksi gagal");
      }
      const data = await res.json();
      renderResult(data);
      statusEl.textContent = "Sukses.";
    } catch (err) {
      statusEl.textContent = err.message;
      statusEl.classList.add("error");
      resultBox.style.display = "none";
    } finally {
      setLoading(false);
    }
  }

  function renderResult(data) {
    topLabelEl.textContent = `${data.top_label} • ${(data.confidence * 100).toFixed(2)}%`;
    probTableBody.innerHTML = "";
    data.probabilities
      .sort((a, b) => b.prob - a.prob)
      .forEach(({ label, prob }) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${label}</td><td>${(prob * 100).toFixed(2)}%</td>`;
        probTableBody.appendChild(tr);
      });
    saveInfo.textContent = "";
    saveInfo.style.display = "none";
    resultBox.style.display = "grid";
  }
});
