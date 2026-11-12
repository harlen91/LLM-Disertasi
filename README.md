# 🛡️ IICD - LLM-Augmented Explainable Annotation & Distillation Studio

Aplikasi berbasis **Streamlit (Python)** untuk anotasi multi-dimensi dataset cyberbullying Bahasa Indonesia menggunakan **SumoPod AI Gateway API** dan distilasi pengetahuan (*Knowledge Distillation*) ke model Transformer Encoder **IndoBERTweet** serta arsitektur **Dual-Stream Hybrid Fusion** untuk penelitian Disertasi.

Aplikasi ini ditransformasikan langsung dari eksperimen notebook [`IICD_LLaMA3_8B_HuggingFace_Cache_Annotation_v2.ipynb`](../IICD_LLaMA3_8B_HuggingFace_Cache_Annotation_v2.ipynb).

---

## 🌟 Fitur Utama

1. **Cloud-First LLM Engine (SumoPod AI Gateway)**:
   - **SumoPod AI Gateway API (Backend Utama)**: Menggunakan API LLM cloud OpenAI-compatible via SumoPod (default `deepseek-ai/DeepSeek-V3`, mendukung `gpt-4o-mini`, `Qwen/Qwen2.5-7B-Instruct`, dll.). RAM super ringan (< 100 MB), cepat, pembayaran lokal (QRIS/IDR), dan siap dideploy ke Streamlit Community Cloud.
   - **Simulation / Mock Fallback Engine**: Fallback aman secara otomatis saat `SUMOPOD_API_KEY` belum diisi atau untuk uji antarmuka, tata letak, dan alur batch secara instan tanpa menggunakan kuota API.
2. **Playground & Explainability Inspector**:
   - Pengujian teks komentar individual secara real-time.
   - Badge visual untuk 6 taksonomi: *Cyberbullying (YES/NO)*, *Type*, *Emotion*, *Cultural Context*, *Implicit Level*, dan *Severity*.
   - Kotak penjelasan penalaran (*chain-of-thought rationale*) dan confidence score (0.00 – 1.00 numerik).
3. **Batch Pipeline dengan Kontrol Kecepatan & Auto-Checkpoint**:
   - **Tuning Throughput (Concurrent Workers)**: Pilihan memproses beberapa komentar secara paralel ke SumoPod API dalam satu batch request.
   - **Auto-Save & Resume**: Menyimpan berkas setiap *N* baris ke folder `output/checkpoints/` dengan deteksi checkpoint instan untuk melanjutkan pemrosesan.
   - Live stream preview data teranotasi, kecepatan pemrosesan, dan ETA.
4. **Dashboard Analitik & Visualisasi Taksonomi**:
   - Visualisasi interaktif Plotly: Proporsi Cyberbullying, Distribusi Tipe, Konteks Kultural, dan Tingkat Keterbukaan (Explicit vs Implicit).
   - Heatmap matriks silang (*Cultural Context vs Severity*, *Type vs Implicit*).
   - Ringkasan statistik frekuensi kategori dan rerata confidence.
5. **Training & Evaluasi Transformer IndoBERTweet**:
   - Kurasi otomatis: Filter error JSON, normalisasi string, dan pembagian dataset *Stratified* (80% Train, 10% Val, 10% Test).
   - Fine-tuning model `indolem/indobertweet-base-uncased` dengan akselerasi hardware (Apple Silicon MPS / CUDA).
   - Evaluasi komprehensif: Akurasi, Macro-F1, Confusion Matrix, dan ketahanan pada ujaran tersirat (E0/E1/E2).
   - Generator tabel otomatis berformat **LaTeX** siap salin untuk Bab 4 naskah Disertasi.

---

## 📂 Struktur Direktori

```text
iicd_annotation_app/
├── app.py                      # Entry point Streamlit Studio (5 Tab Utama)
├── config.py                   # Konfigurasi path, model, dan taksonomi
├── requirements.txt            # Daftar paket dependensi
├── run.sh                      # Script praktis untuk meluncurkan aplikasi
├── core/
│   ├── model_engine.py         # Engine SumoPod AI Gateway & Mock fallback
│   ├── prompt_templates.py     # Template pesan chat OpenAI-compatible & skema JSON ketat
│   ├── parser.py               # Robust JSON extractor, fallback, & auto-repair
│   └── schema.py               # Skema validasi data & normalisasi confidence (0-1)
├── data_manager/
│   ├── dataset_handler.py      # Pembaca CSV/Excel & auto-detect kolom teks
│   └── checkpoint_manager.py   # Pengelola auto-save, resume, & ekspor
├── modules/
│   ├── single_inference.py     # Tab 1: Single Text Playground
│   ├── batch_annotation.py     # Tab 2: Batch Annotation Engine
│   ├── analytics_dashboard.py  # Tab 3: Dashboard Plotly & Statistik
│   ├── model_training_page.py  # Tab 4: Training & Evaluasi IndoBERTweet
│   └── settings_page.py        # Tab 5: Pengaturan Backend & Hardware
├── scripts/
│   ├── curate_dataset.py       # Kurasi data, filter error JSON, & split 80/10/10
│   ├── train_indobertweet.py   # Script pelatihan Transformer IndoBERTweet
│   ├── evaluate_indobertweet.py # Script inferensi & pengujian cepat model
│   └── generate_dissertation_report.py # Generator laporan metrik, grafik, & LaTeX
└── output/
    ├── checkpoints/            # Penyimpanan otomatis progres batch
    ├── final/                  # Dataset bersih kurasi (train, val, test)
    ├── models/                 # Bobot model IndoBERTweet terbaik tersimpan
    └── evaluation_report/      # Laporan metrik, grafik 300 DPI, & tabel LaTeX
```

---

## 🚀 Cara Menjalankan Aplikasi

Pastikan berada di terminal dan berada pada lingkungan Python yang memiliki dependensi:

```bash
# Opsi 1: Menjalankan menggunakan shell script
cd "/Users/harlengilbertsimanullang/Documents/Disertasi/LLM-Augmented Explainable/iicd_annotation_app"
./run.sh

# Opsi 2: Menjalankan langsung via python modul streamlit
cd "/Users/harlengilbertsimanullang/Documents/Disertasi/LLM-Augmented Explainable/iicd_annotation_app"
python3 -m streamlit run app.py
```

Buka peramban (*browser*) pada alamat: **`http://localhost:8501`**
