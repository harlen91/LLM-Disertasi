# 📑 Laporan Evaluasi Komprehensif Disertasi: IndoBERTweet Cyberbullying Detection

Dokumen ini memuat analisis metrik kinerja model Transformer Encoder **IndoBERTweet** (`indolem/indobertweet-base-uncased`) yang dilatih menggunakan dataset IICD hasil kurasi anotasi LLaMA-3 (SumoPod).

---

## 1. Ringkasan Metrik Kinerja Utama (Test Set)

| Metrik Evaluasi | Nilai | Interpretasi Akademik |
| :--- | :--- | :--- |
| **Akurasi (Accuracy)** | **78.00%** (0.7800) | Proporsi total prediksi yang tepat secara keseluruhan. |
| **Macro F1-Score** | **0.7764** | Rata-rata harmonik F1 tanpa bias ukuran sampel kelas (Metrik Utama). |
| **Weighted F1-Score** | **0.7787** | Rata-rata F1 terbobot terhadap frekuensi tiap kelas. |
| **Macro Precision** | **0.7815** | Rata-rata ketepatan prediksi positif di kedua kelas. |
| **Macro Recall** | **0.7748** | Rata-rata kemampuan mendeteksi sampel relevan di kedua kelas. |
| **Area Under ROC (ROC-AUC)** | **0.8576** | Kemampuan diskriminasi probabilitas model membedakan kelas. |
| **Log-Loss** | **0.4745** | Ketidakpastian probabilitas prediksi (makin kecil makin baik). |

---

## 2. Kinerja per Kelas (Class-wise Classification Report)

| Kelas Target | Precision | Recall | F1-Score | Support (N) |
| :--- | :---: | :---: | :---: | :---: |
| **Non-Cyberbullying (0)** | 0.7727 | 0.8395 | 0.8047 | 81 |
| **Cyberbullying (1)** | 0.7903 | 0.7101 | 0.7481 | 69 |
| **Macro Average** | **0.7815** | **0.7748** | **0.7764** | **150** |

### Matriks Konfusi (Confusion Matrix):
- **True Negative (TN)**: 68 (Ujaran non-cyberbullying diprediksi benar)
- **False Positive (FP)**: 13 (Ujaran non-cyberbullying salah diprediksi sebagai cyberbullying / Type I error)
- **False Negative (FN)**: 20 (Ujaran cyberbullying luput terdeteksi / Type II error)
- **True Positive (TP)**: 49 (Ujaran cyberbullying berhasil dideteksi dengan tepat)

---

## 3. Analisis Ketahanan Multi-Dimensi Taksonomi

### A. Berdasarkan Tingkat Keterbukaan Ujaran (Implicit Level):

| Tingkat Keterbukaan | Deskripsi | Total Uji | Prediksi Benar | Akurasi (%) |
| :--- | :--- | :---: | :---: | :---: |
| **E0** | Explicit | 103 | 87 | **84.47%** |
| **E1** | Semi-Implicit | 46 | 30 | **65.22%** |
| **E2** | Implicit | 1 | 0 | **0.00%** |

### B. Berdasarkan Tipe Cyberbullying (Khusus Label Positif):

| Kode Tipe | Total Uji | Prediksi Benar | Akurasi (%) |
| :--- | :---: | :---: | :---: |
| **INS** | 47 | 32 | **68.09%** |
| **BSH** | 14 | 10 | **71.43%** |
| **THR** | 4 | 4 | **100.00%** |
| **SAR** | 2 | 1 | **50.00%** |
| **SHM** | 2 | 2 | **100.00%** |


---

## 4. Visualisasi Publikasi Tersimpan
Grafik siap cetak (300 DPI) telah diekspor ke direktori laporan:
- `figure_confusion_matrix.png`
- `figure_implicitness_accuracy.png`

---
*Laporan digenerate secara otomatis oleh Modul Evaluasi Disertasi Antigravity.*
