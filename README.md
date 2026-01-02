# CornGuard — Deteksi Penyakit Daun Jagung

Aplikasi web sederhana (FastAPI + TensorFlow + HTML/JS statis) untuk memprediksi penyakit pada daun jagung menggunakan model CNN DenseNet121 terlatih. Frontend menyediakan kanvas untuk crop manual sebelum gambar dikirim ke backend.

## Struktur Proyek
- `backend/main.py` — API FastAPI, pemuatan model Keras, preprocessing, endpoint prediksi.
- `backend/models/final_model.h5` — model bawaan (DenseNet121, input 256×256, 4 kelas).
- `frontend/` — halaman statis: `index.html` (landing) dan `prediction.html` (form prediksi) plus CSS/JS.

## Fitur Utama
- Prediksi 4 kelas: Bercak, Hawar, Karat, Sehat.
- Pemuatan model fleksibel: `MODEL_PATH` env atau file `.keras/.h5` pertama di `backend/models/`.
- Preprocess otomatis: RGB, resize sesuai input model (fallback 256×256), normalisasi 0–1.
- Frontend mendukung crop manual di kanvas atau pakai gambar penuh; resize ke 256×256 di sisi klien.
- CORS diizinkan (`*`) sehingga bisa diakses dari alat dev lokal.

## Prasyarat
- Python 3.10+ disarankan (backend).
- Pip & virtualenv.
- Tidak ada build tool frontend; cukup sajikan folder `frontend/` secara statis.

## Menjalankan Lokal
1) Siapkan backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Opsional: tentukan model lain
# export MODEL_PATH=/path/ke/model.h5
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```
Catatan: frontend secara otomatis menggunakan `http://localhost:8001` saat berjalan di localhost.

2) Sajikan frontend
- Buka langsung `frontend/index.html` di browser, atau
- Jalankan server statis: `python -m http.server 5500 --directory frontend`
  lalu buka `http://localhost:5500/prediction.html`.

## Endpoints API
- `GET /health` — status dan meta model.
- `GET /info` — meta model (path, nama file, input size, labels).
- `POST /predict` — form-data `file` (image/*). Respons contoh:
```json
{
  "top_label": "Bercak",
  "confidence": 0.92,
  "probabilities": [
    {"label": "Bercak", "prob": 0.92},
    {"label": "Hawar", "prob": 0.04},
    {"label": "Karat", "prob": 0.03},
    {"label": "Sehat", "prob": 0.01}
  ],
  "saved_filename": "20241201T120000Z_abcd1234_upload.png"
}
```
Upload disalin ke `backend/uploads/` dengan nama unik.

## Catatan Deployment
- Lihat `cornguard.service` untuk contoh service systemd (worker gunicorn+uvicorn di port 8001, batas thread TensorFlow).
- Lihat `site1.conf` untuk contoh nginx: serve frontend dari `/var/www/site1/frontend`, proxy `/api/` ke `127.0.0.1:8001`, HTTPS Let’s Encrypt.
- Sesuaikan path `WorkingDirectory`, `PATH`, dan domain sebelum dipakai.

## Model & Pelatihan
- Model bawaan: `backend/models/final_model.h5` (DenseNet121, 4 kelas, input 256×256).
- Script pelatihan (ekspor Colab) ada di `untitled1.py` jika perlu menelusuri proses training atau mengganti model.

## Troubleshooting Singkat
- Model tidak ditemukan: set env `MODEL_PATH` atau pastikan ada file `.keras/.h5` di `backend/models/`.
- CORS/URL: di localhost gunakan port 8001; di produksi pastikan frontend disajikan di domain yang sama sehingga `/api` ter-proxy oleh nginx.
- Prediksi lambat: TensorFlow CPU; kurangi ukuran gambar di klien (sudah di-resize 256×256) atau batasi thread via env yang ada di unit service.
