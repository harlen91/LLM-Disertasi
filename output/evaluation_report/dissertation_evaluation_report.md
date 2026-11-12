# 📑 Laporan Evaluasi Komprehensif Disertasi

> [!NOTE]
> **Kebijakan Metodologi (`pseudo_labeled_llm`)**: Penelitian ini menggunakan **LLM-generated pseudo-labels** sebagai target supervisi utama (tanpa kewajiban human ground truth), dengan **Stream B berupa 33D representasi bukti kontekstual non-target** (Linguistic, Cultural, Emotional, Pragmatic, Contextual) yang aktif secara konsisten pada training, validation, testing, dan inference. Metrik mengevaluasi kesepakatan/konsistensi model terhadap pseudo-label LLM.


## 1. Ringkasan Metrik Kinerja Utama (Test Set, $\tau^* = 0.20$)

| Metrik Evaluasi | Nilai | 95% Bootstrap CI | Interpretasi |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **80.00%** (0.8000) | [0.4, 1.0] | Kesepakatan prediksi tepat terhadap pseudo-label |
| **Macro F1-Score** | **0.4444** | [0.2857, 1.0] | Rata-rata harmonik F1 adil kelas (Utama) |
| **Weighted F1-Score** | 0.7111 | - | Rata-rata F1 terbobot proporsi kelas |
| **Macro Precision / Recall** | 0.4000 / 0.5000 | - | Ketepatan vs sensitivitas makro |
| **ROC-AUC** | 0.0 | [0.0, 0.0] | Diskriminasi probabilitas ROC |
| **PR-AUC** | 0.6792 | [0.325, 0.7625] | Area kurva Precision-Recall |
| **Calibration (ECE / MCE)** | 0.5124 / 0.7711 | - | Kesesuaian kepercayaan probabilistik |
| **Brier Score / Log-Loss** | 0.2797 / 0.7706 | - | Akurasi probabilistik & ketidakpastian |

## 2. Kinerja per Kelas & Analisis Kesalahan (FP / FN)
- **Confusion Matrix**: `TN=0`, `FP=1`, `FN=0`, `TP=4`
- **False Positive Rate (FPR)**: 100.00% (1 sampel)
- **False Negative Rate (FNR)**: 0.00% (0 sampel)

## 3. Rincian Tingkat Keterbukaan (E0/E1/E2) & Tipe Cyberbullying

| Tingkat Keterbukaan | Jumlah | Benar | Akurasi (%) | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **E0** | 3 | 2 | 66.67% | 0.4000 |
| **E1** | 2 | 2 | 100.00% | 1.0000 |

| Tipe Taksonomi | Jumlah | Benar | Akurasi (%) |
| :--- | :---: | :---: | :---: |
| **INS** | 4 | 4 | 100.00% |
| **NONE** | 1 | 0 | 0.00% |

## 4. Studi Ablasi 6-Konfigurasi (E1–E6 pada Split Identik)

| Eksperimen | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | Gate z |
| :--- | :---: | :---: | :---: | :---: | :---: |
| E1. Transformer-Only Baseline (Stream A) | 0.8000 | 0.4444 | 0.0 | 0.6792 | 0.4970 |
| E2. LLM Knowledge-Only Model (33D Contextual Cues) | 0.8000 | 0.7619 | 1.0 | 1.0 | - |
| E3. Full Hybrid Transformer-LLM Fusion | 0.8000 | 0.4444 | 0.0 | 0.6792 | 0.4941 |
| E4. Hybrid without Cultural Knowledge | 0.8000 | 0.4444 | 0.0 | 0.6792 | 0.4943 |
| E5. Hybrid without Emotional Knowledge | 0.8000 | 0.4444 | 0.0 | 0.6792 | 0.4937 |
| E6. Hybrid without Gated Fusion (Concat) | 0.8000 | 0.4444 | 1.0 | 1.0 | 0.5000 |

## 5. Kinerja Multi-Task 6 Heads (Hybrid Model E3)

| Classification Head | Support | Accuracy (%) | Macro-F1 |
| :--- | :---: | :---: | :---: |
| **cyberbullying (binary)** | 5 | 80.00% | 0.4444 |
| **type** | 5 | 20.00% | 0.1667 |
| **emotion** | 5 | 40.00% | 0.1905 |
| **cultural** | 5 | 20.00% | 0.1111 |
| **implicit** | 5 | 40.00% | 0.2667 |
| **severity** | 5 | 20.00% | 0.1667 |

## 6. Konsistensi Guru-Murid & Reliabilitas Non-GT (LLM vs Hybrid Model)

| Dimensi | Support | Agreement Rate | Cohen's $\kappa$ |
| :--- | :---: | :---: | :---: |
| **cyberbullying** | 5 | 80.00% | 0.0000 |
| **type** | 5 | 20.00% | 0.0909 |
| **emotion** | 5 | 40.00% | -0.1538 |
| **cultural** | 5 | 20.00% | 0.1304 |
| **implicit** | 5 | 40.00% | 0.2105 |
| **severity** | 5 | 20.00% | 0.0909 |

## 7. Artefak Grafik (300 DPI)
- `figure_confusion_matrix.png`
- `figure_implicitness_accuracy.png`
