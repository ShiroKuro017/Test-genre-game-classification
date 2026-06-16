# Klasifikasi Genre Game dari Sinopsis

Mengklasifikasikan genre game berdasarkan teks sinopsis (deskripsi singkat) berbahasa Inggris menggunakan tiga pendekatan berbeda dan membandingkannya:

1. Klasifikasi single-label dengan machine learning klasik
2. Klasifikasi multi-label dengan machine learning klasik
3. Klasifikasi multi-label dengan Transformer (DistilBERT) yang di-fine-tune

Tujuan penelitian ini adalah mengevaluasi seberapa baik genre sebuah game dapat diprediksi dari teks sinopsisnya, sekaligus membandingkan pipeline klasik berbasis TF-IDF dengan model Transformer kontekstual.

Versi bahasa: Indonesia (berkas ini) dan Inggris ([README.md](README.md), versi utama untuk GitHub).

---

## Daftar Isi

1. [Latar Belakang dan Tujuan](#1-latar-belakang-dan-tujuan)
2. [Dataset](#2-dataset)
3. [Struktur Repositori](#3-struktur-repositori)
4. [Metodologi](#4-metodologi)
5. [Perbandingan Tiga Pendekatan](#5-perbandingan-tiga-pendekatan)
6. [Penjelasan Detail Tiap Berkas](#6-penjelasan-detail-tiap-berkas)
7. [Cara Menjalankan](#7-cara-menjalankan)
8. [Metrik Evaluasi](#8-metrik-evaluasi)
9. [Catatan dan Keterbatasan](#9-catatan-dan-keterbatasan)

---

## 1. Latar Belakang dan Tujuan

Toko game seperti Steam mendeskripsikan setiap judul dengan sinopsis singkat. Penelitian ini menyelidiki apakah genre sebuah game dapat diprediksi secara otomatis dari teks sinopsis tersebut menggunakan Natural Language Processing dan machine learning.

Tujuan utama adalah klasifikasi genre single-label (memprediksi genre utama sebuah game). Dua pendekatan tambahan, yaitu klasifikasi multi-label dan model berbasis Transformer, disertakan sebagai pembanding untuk mempelajari bagaimana persoalan ini berperilaku ketika sebuah game dapat memiliki beberapa genre sekaligus dan ketika digunakan model bahasa kontekstual modern.

---

## 2. Dataset

- Sumber: Steam Games Dataset oleh FronkonGames (Kaggle): https://www.kaggle.com/datasets/fronkongames/steam-games-dataset
- Dataset berisi data game Steam, termasuk deskripsi teks dan tag genre.
- Hanya sinopsis berbahasa Inggris yang digunakan.

### Penting: data mentah harus diperbaiki dulu

Dataset mentah mengandung error struktur: sebagian baris memiliki jumlah kolom yang tidak konsisten (kolom bergeser tidak sesuai), sehingga pembacaan CSV secara langsung gagal. Masalah ketidaksesuaian kolom ini didokumentasikan pada `Colomn Dataset tidak sesuai.pdf`.

Sebelum menjalankan notebook model mana pun, jalankan:

```
fix_steam_games_csv_from_json.ipynb
```

Notebook ini menyusun ulang data dari sumbernya dan menghasilkan berkas bersih yang kolomnya sudah benar (`games_fixed.csv`), yang kemudian dimuat oleh notebook model. Jangan memasukkan berkas mentah dari Kaggle langsung ke notebook model.

---

## 3. Struktur Repositori

| Berkas | Deskripsi |
|--------|-----------|
| `fix_steam_games_csv_from_json.ipynb` | Perbaikan dan pembersihan data. Memperbaiki kolom yang tidak sesuai pada dataset mentah dan menghasilkan `games_fixed.csv`. Jalankan terlebih dahulu. |
| `TEST03_2.ipynb` | Klasifikasi single-label (genre utama). Pipeline ML klasik (TF-IDF + Naive Bayes / KNN / Linear SVM). Ini adalah pendekatan utama penelitian. |
| `TEST03_MULTILABEL.ipynb` | Klasifikasi multi-label dengan ML klasik (One-vs-Rest). Pipeline fitur sama dengan notebook single-label. |
| `TEST04_TRANSFORMER_multilabel_2.ipynb` | Klasifikasi multi-label menggunakan Transformer (DistilBERT) yang di-fine-tune. |
| `Colomn Dataset tidak sesuai.pdf` | Dokumentasi masalah ketidaksesuaian kolom pada dataset mentah. |

---

## 4. Metodologi

### 4.1 Praproses Teks (notebook ML klasik)

Notebook klasik menerapkan praproses NLP standar pada teks sinopsis:

- Case folding (menjadi huruf kecil)
- Tokenisasi
- Penghapusan stopword (Inggris)
- Stemming (Porter Stemmer)
- Penghapusan token yang sangat pendek

Tiga skenario praproses dibandingkan:

- S1: hanya penghapusan stopword
- S2: hanya stemming
- S3: penghapusan stopword dan stemming

### 4.2 Ekstraksi Fitur (notebook ML klasik)

Teks diubah menjadi fitur numerik dengan TF-IDF menggunakan konfigurasi berikut (identik pada kedua notebook klasik):

- `max_features = 20000`
- `ngram_range = (1, 2)`
- `sublinear_tf = True`
- `min_df = 3`
- `max_df = 0.95`

### 4.3 Model (notebook ML klasik)

Tiga algoritma dilatih dan dibandingkan:

- Multinomial Naive Bayes (`alpha = 0.1`)
- K-Nearest Neighbors (`n_neighbors = 11`, jarak cosine)
- Linear SVM (`class_weight = 'balanced'`, `max_iter = 5000`)

Model terbaik dipilih otomatis berdasarkan F1-score tertinggi.

### 4.4 Model Transformer

Notebook Transformer melakukan fine-tuning DistilBERT (`distilbert-base-uncased`) untuk prediksi genre multi-label:

- Panjang urutan maksimum: 256 token
- Ukuran batch: 16
- Learning rate: 2e-5
- Epoch: 4
- Hanya pembersihan teks ringan (tanpa penghapusan stopword atau stemming, karena model memakai embedding kontekstual)
- Ketidakseimbangan kelas ditangani dengan undersampling
- Ambang keputusan per genre disetel pada data validasi

---

## 5. Perbandingan Tiga Pendekatan

| Aspek | Single-label (`TEST03_2`) | Multi-label Konvensional (`TEST03_MULTILABEL`) | Transformer (`TEST04_TRANSFORMER_multilabel_2`) |
|-------|---------------------------|------------------------------------------------|-------------------------------------------------|
| Jenis tugas | Satu genre per game (genre utama) | Beberapa genre per game | Beberapa genre per game |
| Representasi label | LabelEncoder (multi-kelas) | MultiLabelBinarizer + One-vs-Rest | Multi-hot, sigmoid per genre |
| Ekstraksi fitur | TF-IDF | TF-IDF | Embedding kontekstual DistilBERT |
| Konfigurasi fitur | max_features=20000, ngram (1,2), sublinear_tf, min_df=3, max_df=0.95 | Identik dengan single-label | Tokenizer dengan panjang maksimum 256 |
| Algoritma | Naive Bayes, KNN, Linear SVM | Naive Bayes, KNN, Linear SVM (masing-masing dibungkus One-vs-Rest) | DistilBERT yang di-fine-tune |
| Hyperparameter | NB alpha=0.1; KNN k=11 (cosine); SVM balanced, max_iter=5000 | Identik dengan single-label (versi One-vs-Rest) | batch=16, lr=2e-5, epoch=4 |
| Validasi silang | Stratified K-Fold (k=5) | K-Fold (k=5); stratified standar tidak berlaku untuk multi-label | Tidak ada; memakai data hold-out validation |
| Pembagian data | 80/20, stratified | 80/20 | Train/validation split dengan undersampling |
| Penanganan imbalance | class_weight='balanced' (SVM) | class_weight='balanced' (SVM) | Undersampling + penyetelan ambang per genre |
| Definisi akurasi | Akurasi standar | 1 - Hamming Loss | Berbasis ambang per genre |
| Metrik | Accuracy, Precision, Recall, F1 (macro) + MSE, RMSE | Accuracy, Precision, Recall, F1 (macro) + MSE, RMSE | Precision, Recall, F1 (macro) + ambang optimal |
| Praproses | Case folding, tokenisasi, stopword, stemming, hapus token pendek (S1/S2/S3) | Sama dengan single-label | Hanya pembersihan ringan |
| Output inferensi | Satu genre prediksi + peringkat keyakinan Top-5 | Beberapa genre prediksi + keyakinan per genre | Beberapa genre prediksi + keyakinan sigmoid |
| Bahasa data | Inggris | Inggris | Inggris |

### Poin Utama

- Notebook single-label dan multi-label konvensional kini memakai pipeline fitur dan hyperparameter yang identik. Perbedaannya murni metodologis: single-label memprediksi satu genre (LabelEncoder, Stratified K-Fold, akurasi standar), sedangkan multi-label memprediksi beberapa genre (One-vs-Rest, K-Fold, akurasi diukur sebagai 1 - Hamming Loss).
- Pendekatan Transformer berbeda secara fundamental: memakai embedding kontekstual alih-alih TF-IDF, tidak menerapkan penghapusan stopword atau stemming, tidak memakai K-Fold, dan mengandalkan undersampling serta penyetelan ambang per genre.
- Output inferensi ketiga notebook kini mengikuti gaya yang seragam: daftar skor keyakinan per genre dengan penanda untuk genre yang diprediksi.

---

## 6. Penjelasan Detail Tiap Berkas

### 6.1 `fix_steam_games_csv_from_json.ipynb` (penyiapan data)

Dataset mentah dari Kaggle tidak dapat dibaca langsung karena sebagian baris memiliki jumlah kolom yang berbeda dari header (kolom menjadi tidak sejajar). Notebook ini menyusun ulang record secara benar dan menulis CSV bersih (`games_fixed.csv`). Selalu jalankan notebook ini sebelum notebook model.

### 6.2 `TEST03_2.ipynb` (single-label, pendekatan utama)

Memprediksi genre utama sebuah game. Pipeline melakukan praproses (skenario S1/S2/S3), membangun fitur TF-IDF, melatih Naive Bayes, KNN, dan Linear SVM, membandingkannya dengan Stratified K-Fold cross-validation, lalu melaporkan Accuracy, Precision, Recall, F1, MSE, dan RMSE. Model terbaik (F1 tertinggi) dipilih otomatis, dan contoh inferensi mencetak genre prediksi beserta peringkat keyakinan Top-5.

### 6.3 `TEST03_MULTILABEL.ipynb` (multi-label, konvensional)

Memakai pipeline fitur dan hyperparameter yang sama dengan notebook single-label, tetapi memperlakukan persoalan sebagai multi-label menggunakan MultiLabelBinarizer dan klasifikator One-vs-Rest. Karena pembagian stratified standar tidak berlaku untuk target multi-label, digunakan K-Fold biasa, dan akurasi dilaporkan sebagai 1 - Hamming Loss. Contoh inferensi mencetak semua genre prediksi beserta keyakinan per genre.

### 6.4 `TEST04_TRANSFORMER_multilabel_2.ipynb` (Transformer)

Melakukan fine-tuning DistilBERT untuk prediksi genre multi-label. Alih-alih TF-IDF, model memakai embedding kontekstual, dengan pembersihan teks ringan saja. Diterapkan undersampling untuk menangani ketidakseimbangan kelas, dan ambang keputusan disetel per genre pada data validasi. Contoh inferensi mencetak genre prediksi beserta skor keyakinan sigmoid.

---

## 7. Cara Menjalankan

### Kebutuhan

Notebook klasik (`TEST03_2`, `TEST03_MULTILABEL`):

```
python >= 3.9
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
```

Notebook Transformer (`TEST04_TRANSFORMER_multilabel_2`) tambahan:

```
torch
transformers
```

Resource NLTK yang dipakai notebook klasik:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Langkah

1. Unduh dataset dari Kaggle: https://www.kaggle.com/datasets/fronkongames/steam-games-dataset
2. Jalankan `fix_steam_games_csv_from_json.ipynb` untuk memperbaiki data dan menghasilkan `games_fixed.csv`.
3. Buka dan jalankan salah satu notebook model:
   - `TEST03_2.ipynb` untuk klasifikasi single-label
   - `TEST03_MULTILABEL.ipynb` untuk klasifikasi multi-label
   - `TEST04_TRANSFORMER_multilabel_2.ipynb` untuk model Transformer

GPU disarankan untuk notebook Transformer.

---

## 8. Metrik Evaluasi

- Accuracy: proporsi prediksi benar. Pada notebook multi-label dilaporkan sebagai 1 - Hamming Loss.
- Precision, Recall, F1-score: dilaporkan dengan rata-rata macro agar setiap genre berkontribusi setara terlepas dari frekuensinya.
- MSE dan RMSE: dilaporkan pada notebook klasik sebagai ukuran error tambahan; label genre dikodekan ke angka terlebih dahulu sebelum dihitung.

---

## 9. Catatan dan Keterbatasan

- Klasifikasi single-label menetapkan tepat satu genre per game; game yang sebenarnya memiliki banyak genre disederhanakan ke genre utamanya. Notebook multi-label mengatasi keterbatasan ini.
- Dataset tidak seimbang antar genre, yang dimitigasi melalui bobot kelas seimbang (klasik) serta undersampling dan penyetelan ambang (Transformer).
- Hanya sinopsis berbahasa Inggris yang digunakan; hasil mungkin tidak berlaku untuk bahasa lain.
- Seluruh eksperimen bergantung pada dataset bersih yang dihasilkan oleh `fix_steam_games_csv_from_json.ipynb`; memakai berkas mentah yang belum diperbaiki akan menyebabkan error pemuatan data.
