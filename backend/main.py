import io
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import h5py
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import load_model, Sequential
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_FILES = [
    MODELS_DIR / "Model_CNN_256px.keras",  # bawaan lama
    MODELS_DIR / "final_model.h5",  # bawaan baru dari notebook
]

# Ganti label sesuai urutan kelas saat training
CLASS_LABELS = ["Bercak", "Hawar", "Karat", "Sehat"]


def resolve_model_path() -> Path:
    """Cari path model dari env MODEL_PATH atau fallback ke .keras/.h5 di folder models."""
    env_model = os.getenv("MODEL_PATH")
    candidates = []

    if env_model:
        candidate = Path(env_model)
        if not candidate.is_absolute():
            candidate = (BASE_DIR / candidate).resolve()
        candidates.append(candidate)
    else:
        candidates.extend(DEFAULT_MODEL_FILES)
        # scan fallback .h5/.keras supaya tetap jalan walau nama file berubah
        candidates.extend(sorted(MODELS_DIR.glob("*.h5")))
        candidates.extend(sorted(MODELS_DIR.glob("*.keras")))

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path

    raise RuntimeError(
        f"Tidak ada model yang ditemukan. Set env MODEL_PATH atau letakkan file .keras/.h5 di {MODELS_DIR}"
    )


MODEL_PATH = resolve_model_path()


def _extract_input_shape_from_h5(path: Path) -> tuple[int, int, int] | None:
    """Ambil input shape dari model_config di file .h5 (jika ada)."""
    try:
        with h5py.File(path, "r") as f:
            raw = f.attrs.get("model_config")
            if raw is None:
                return None
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            data = json.loads(text)
    except Exception:
        return None

    def _search(obj):
        if isinstance(obj, dict):
            if "batch_shape" in obj:
                return obj["batch_shape"]
            for v in obj.values():
                found = _search(v)
                if found:
                    return found
        elif isinstance(obj, list):
            for v in obj:
                found = _search(v)
                if found:
                    return found
        return None

    batch = _search(data)
    if batch and len(batch) >= 4:
        try:
            return (int(batch[1]), int(batch[2]), int(batch[3]))
        except Exception:
            return None
    return None


def _build_cnn_model(input_shape: tuple[int, int, int], num_classes: int):
    """Bangun arsitektur yang sama dengan training (DenseNet121 + head kustom)."""
    base_model = DenseNet121(weights=None, include_top=False, input_shape=input_shape)
    base_model.trainable = False
    return Sequential(
        [
            base_model,
            BatchNormalization(),
            GlobalAveragePooling2D(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )


def load_cnn_model(path: Path):
    """Coba muat model; fallback khusus .h5 jika kena error dtype."""
    try:
        return load_model(path)
    except ValueError as exc:
        if path.suffix.lower() == ".h5":
            print(f"[model] load_model gagal ({exc}); mencoba fallback load_weights untuk .h5 ...")
            input_shape = _extract_input_shape_from_h5(path) or (256, 256, 3)
            try:
                model = _build_cnn_model(input_shape, len(CLASS_LABELS))
                model.load_weights(path)
                return model
            except Exception as inner:
                raise RuntimeError(
                    f"Gagal memuat model .h5 via fallback (input_shape={input_shape}): {inner}"
                ) from inner
        raise
    except Exception as exc:
        raise RuntimeError(f"Gagal memuat model di {path}: {exc}") from exc

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Muat model sekali saat startup
MODEL = load_cnn_model(MODEL_PATH)

# Ambil input size dari model (fallback ke 256x256)
try:
    _shape = MODEL.input_shape  # e.g., (None, 224, 224, 3)
    if isinstance(_shape, list):  # untuk model multi-input
        _shape = _shape[0]
    TARGET_SIZE = (_shape[1], _shape[2])
except Exception:
    TARGET_SIZE = (256, 256)

MODEL_META: Dict[str, Any] = {
    "model_path": str(MODEL_PATH),
    "model_name": MODEL_PATH.name,
    "input_size": {"width": TARGET_SIZE[0], "height": TARGET_SIZE[1]},
    "class_labels": CLASS_LABELS,
}


class Probability(BaseModel):
    label: str
    prob: float


class PredictionResult(BaseModel):
    top_label: str
    confidence: float
    probabilities: List[Probability]
    saved_filename: str


app = FastAPI(title="Corn Leaf Disease Detector", version="1.0.0")

# Izinkan akses dari mana saja (frontend lokal memakai fetch)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Baca bytes gambar, ubah ke RGB, resize ke ukuran input model, dan normalisasi."""
    with Image.open(io.BytesIO(file_bytes)) as img:
        img = img.convert("RGB")
        img = img.resize(TARGET_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def save_upload(file_bytes: bytes, original_name: str) -> str:
    """Simpan file upload ke folder uploads dengan nama unik."""
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_name = original_name.replace("/", "_").replace("\\", "_")
    unique_name = f"{timestamp}_{uuid.uuid4().hex}_{safe_name}"
    target_path = UPLOAD_DIR / unique_name
    with target_path.open("wb") as f:
        f.write(file_bytes)
    return unique_name


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_META}


@app.get("/")
def root():
    return {
        "message": "Backend siap. Gunakan POST /predict untuk prediksi gambar, /health untuk cek status."
    }


@app.get("/info")
def info():
    return MODEL_META


@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Harap unggah file gambar (jpg/png).")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File kosong.")

    # Simpan salinan untuk arsip penggunaan
    saved_name = save_upload(file_bytes, file.filename)

    # Preprocess & prediksi
    input_batch = preprocess_image(file_bytes)
    preds = MODEL.predict(input_batch)
    probs = preds[0].tolist()

    # Ambil label tertinggi
    top_idx = int(np.argmax(probs))
    top_label = CLASS_LABELS[top_idx] if top_idx < len(CLASS_LABELS) else str(top_idx)

    probability_list = [
        Probability(label=CLASS_LABELS[i] if i < len(CLASS_LABELS) else str(i), prob=float(p))
        for i, p in enumerate(probs)
    ]
    max_conf = float(probs[top_idx])

    return PredictionResult(
        top_label=top_label,
        confidence=max_conf,
        probabilities=probability_list,
        saved_filename=saved_name,
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
