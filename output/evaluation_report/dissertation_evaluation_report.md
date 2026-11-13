# 📑 Laporan Evaluasi Komprehensif Disertasi

> [!NOTE]
> **Kebijakan Metodologi (`pseudo_labeled_llm`)**: Penelitian ini menggunakan **LLM-generated pseudo-labels** sebagai target supervisi utama (tanpa kewajiban human ground truth), dengan **Stream B berupa 33D representasi bukti kontekstual non-target** (Linguistic, Cultural, Emotional, Pragmatic, Contextual) yang aktif secara konsisten pada training, validation, testing, dan inference. Metrik mengevaluasi kesepakatan/konsistensi model terhadap pseudo-label LLM.


## 1. Ringkasan Metrik Kinerja Utama (Test Set, $\tau^* = 0.54$)

| Metrik Evaluasi | Nilai | 95% Bootstrap CI | Interpretasi |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **56.52%** (0.5652) | [0.3913, 0.7391] | Kesepakatan prediksi tepat terhadap pseudo-label |
| **Macro F1-Score** | **0.4866** | [0.303, 0.6734] | Rata-rata harmonik F1 adil kelas (Utama) |
| **Weighted F1-Score** | 0.5652 | - | Rata-rata F1 terbobot proporsi kelas |
| **Macro Precision / Recall** | 0.4866 / 0.4866 | - | Ketepatan vs sensitivitas makro |
| **ROC-AUC** | 0.3929 | [0.13, 0.6776] | Diskriminasi probabilitas ROC |
| **PR-AUC** | 0.6389 | [0.4649, 0.9131] | Area kurva Precision-Recall |
| **Calibration (ECE / MCE)** | 0.1534 / 0.3423 | - | Kesesuaian kepercayaan probabilistik |
| **Brier Score / Log-Loss** | 0.2428 / 0.6799 | - | Akurasi probabilistik & ketidakpastian |

## 2. Kinerja per Kelas & Analisis Kesalahan (FP / FN)
- **Confusion Matrix**: `TN=2`, `FP=5`, `FN=5`, `TP=11`
- **False Positive Rate (FPR)**: 71.43% (5 sampel)
- **False Negative Rate (FNR)**: 31.25% (5 sampel)

## 3. Rincian Tingkat Keterbukaan (E0/E1/E2) & Tipe Cyberbullying

| Tingkat Keterbukaan | Jumlah | Benar | Akurasi (%) | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **E0** | 11 | 5 | 45.45% | 0.4500 |
| **E1** | 12 | 8 | 66.67% | 0.4000 |

| Tipe Taksonomi | Jumlah | Benar | Akurasi (%) |
| :--- | :---: | :---: | :---: |
| **BSH** | 3 | 1 | 33.33% |
| **INS** | 6 | 4 | 66.67% |
| **NONE** | 7 | 2 | 28.57% |
| **SAR** | 4 | 4 | 100.00% |
| **SEX** | 1 | 1 | 100.00% |
| **SHM** | 2 | 1 | 50.00% |

## 4. Studi Ablasi 6-Konfigurasi (E1–E6 pada Split Identik)

| Eksperimen | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | Gate z |
| :--- | :---: | :---: | :---: | :---: | :---: |
| E1. Transformer-Only Baseline (Stream A) | 0.6522 | 0.3947 | 0.5089 | 0.7191 | 0.4902 |
| E2. LLM Knowledge-Only Model (33D Contextual Cues) | 0.9130 | 0.8873 | 0.9777 | 0.9891 | - |
| E3. Full Hybrid Transformer-LLM Fusion | 0.5652 | 0.4866 | 0.3929 | 0.6389 | 0.4964 |
| E4. Hybrid without Cultural Knowledge | 0.5217 | 0.4868 | 0.4643 | 0.6782 | 0.4966 |
| E5. Hybrid without Emotional Knowledge | 0.6087 | 0.4623 | 0.4018 | 0.6412 | 0.4988 |
| E6. Hybrid without Gated Fusion (Concat) | 0.5217 | 0.5217 | 0.6071 | 0.8313 | 0.5000 |

## 5. Kinerja Multi-Task 6 Heads (Hybrid Model E3)

| Classification Head | Support | Accuracy (%) | Macro-F1 |
| :--- | :---: | :---: | :---: |
| **cyberbullying (binary)** | 23 | 56.52% | 0.4866 |
| **type** | 23 | 34.78% | 0.1436 |
| **emotion** | 23 | 65.22% | 0.3299 |
| **cultural** | 23 | 39.13% | 0.1406 |
| **implicit** | 23 | 69.57% | 0.6957 |
| **severity** | 23 | 52.17% | 0.2724 |

## 6. Konsistensi Guru-Murid & Reliabilitas Non-GT (LLM vs Hybrid Model)

| Dimensi | Support | Agreement Rate | Cohen's $\kappa$ |
| :--- | :---: | :---: | :---: |
| **cyberbullying** | 23 | 52.17% | -0.0858 |
| **type** | 23 | 34.78% | 0.1085 |
| **emotion** | 23 | 65.22% | 0.3260 |
| **cultural** | 23 | 39.13% | 0.0000 |
| **implicit** | 23 | 69.57% | 0.3925 |
| **severity** | 23 | 52.17% | 0.1705 |

## 7. Artefak Grafik (300 DPI)
- `figure_confusion_matrix.png`
- `figure_implicitness_accuracy.png`
