# 📑 Laporan Evaluasi Komprehensif Disertasi

> [!WARNING]
> **Status Label Referensi: `pseudo_labeled_llm`**. Dataset uji belum memiliki kolom ground truth manusia yang tervalidasi secara independen. Seluruh metrik di bawah ini mengukur kesepakatan/distilasi terhadap pseudo-label LLM secara bebas kebocoran (*leakage-free*), bukan akurasi terhadap gold standard manusia.


## 1. Ringkasan Metrik Kinerja Utama (Test Set, $\tau^* = 0.76$)

| Metrik Evaluasi | Nilai | 95% Bootstrap CI | Interpretasi |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **20.00%** (0.2000) | [0.0, 0.6] | Proporsi prediksi tepat keseluruhan |
| **Macro F1-Score** | **0.1667** | [0.0, 0.375] | Rata-rata harmonik F1 adil kelas (Utama) |
| **Weighted F1-Score** | 0.0667 | - | Rata-rata F1 terbobot proporsi kelas |
| **Macro Precision / Recall** | 0.1000 / 0.5000 | - | Ketepatan vs sensitivitas makro |
| **ROC-AUC** | 0.0 | [0.0, 0.0] | Diskriminasi probabilitas ROC |
| **PR-AUC** | 0.6792 | [0.325, 0.7625] | Area kurva Precision-Recall |
| **Calibration (ECE / MCE)** | 0.0559 / 0.0559 | - | Kesesuaian kepercayaan probabilistik |
| **Brier Score / Log-Loss** | 0.1689 / 0.5243 | - | Akurasi probabilistik & ketidakpastian |

## 2. Kinerja per Kelas & Analisis Kesalahan (FP / FN)
- **Confusion Matrix**: `TN=1`, `FP=0`, `FN=4`, `TP=0`
- **False Positive Rate (FPR)**: 0.00% (0 sampel)
- **False Negative Rate (FNR)**: 100.00% (4 sampel)

## 3. Rincian Tingkat Keterbukaan (E0/E1/E2) & Tipe Cyberbullying

| Tingkat Keterbukaan | Jumlah | Benar | Akurasi (%) | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **E0** | 4 | 1 | 25.00% | 0.2000 |
| **E1** | 1 | 0 | 0.00% | 0.0000 |

| Tipe Taksonomi | Jumlah | Benar | Akurasi (%) |
| :--- | :---: | :---: | :---: |
| **BSH** | 2 | 0 | 0.00% |
| **INS** | 2 | 0 | 0.00% |
| **NONE** | 1 | 1 | 100.00% |

## 4. Studi Ablasi (Test Set Identik)

| Model | Accuracy | Macro-F1 | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| A. IndoBERTweet-Only (Stream A) | 0.2000 | 0.1667 | 0.0 | 0.6792 |
| B. LLM Representation-Only (Stream B) | N/A | N/A | N/A | blocked_circular_pseudo_label_leakage |
| C. Hybrid Dual-Stream Fusion (IndoBERTweet + LLM Knowledge) | 0.8000 | 0.4444 | 0.75 | 0.95 |

## 5. Artefak Grafik (300 DPI)
- `figure_confusion_matrix.png`
- `figure_implicitness_accuracy.png`
