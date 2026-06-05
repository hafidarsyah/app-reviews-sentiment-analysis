# App Reviews Sentiment Analysis

Proyek ini merupakan implementasi Pemrosesan Bahasa Alami (*Natural Language Processing* / NLP) untuk melakukan analisis sentimen terhadap ulasan pengguna aplikasi. Proyek ini dibangun untuk memenuhi kriteria *submission* akhir, mencakup seluruh alur kerja *Data Science* terintegrasi (*end-to-end*) mulai dari pengumpulan data (*scraping*) secara mandiri, prapemrosesan teks tingkat lanjut, ekstraksi fitur, hingga pelatihan dan evaluasi komparatif berbagai arsitektur model *Machine Learning* dan *Deep Learning*.

---

## 🏆 Kriteria Penilaian yang Terpenuhi

1. **Data Hasil Scraping Mandiri**
   Proses pengumpulan data ulasan dilakukan secara mandiri menggunakan *script* otomatis dengan jumlah sampel awal mencapai **20.000 data ulasan**, melampaui batas minimum pengerjaan proyek (3.000 sampel).

2. **Ekstraksi Fitur & Pelabelan Data**
   * Pelabelan otomatis dilakukan berdasarkan skala *rating* pengguna (Rating 1-2 sebagai **Negatif** dan Rating 4-5 sebagai **Positif**).
   * Teks mentah diproses melalui pembersihan ekspresi reguler (*regex*) dan normalisasi kata tidak baku menggunakan kamus *slang* bahasa Indonesia.
   * Ekstraksi fitur dieksplorasi secara komprehensif menggunakan tiga metode: **TF-IDF**, **Word2Vec**, dan **Keras Embedding Layer**.

3. **Algoritma Pelatihan Komparatif**
   Melatih dan membandingkan performa dari tiga jenis arsitektur algoritma yang berbeda:
   * **LinearSVC** (Support Vector Classifier)
   * **Logistic Regression**
   * **BiLSTM** (Bidirectional Long Short-Term Memory) berbasis *Deep Learning*

4. **Akurasi Pengujian Melampaui Target (Min. 85%)**
   Model terbaik yang dikembangkan berhasil mencapai tingkat akurasi *testing* sebesar **87.15%**, memenuhi standar performa tinggi untuk klasifikasi sentimen teks.

---

## 🔄 Alur Pemrosesan Data

### 1. Prapemrosesan Teks (*Text Preprocessing*)
Akurasi model dioptimalkan melalui fungsi pembersihan teks kustom yang meliputi:
* **Lowercasing:** Mengubah seluruh karakter teks menjadi huruf kecil.
* **Regex Cleaning:** Menghapus URL, *mention* username (`@`), dan *hashtag* (`#`).
* **Konteks Emosi:** Menghapus karakter khusus namun sengaja mempertahankan tanda seru (`!`) dan tanda tanya (`?`) karena membawa bobot konteks sentimen yang kuat bagi model.
* **Normalisasi Slang:** Mengonversi kata gaul, singkatan, dan kata tidak baku khas ulasan Indonesia (seperti *yg, gk, bgt, lemot, apk, ongkir, nyesel*) ke bentuk bakunya menggunakan kamus *slang dictionary*.

### 2. Penyeimbangan Kelas (*Class Balancing*)
Untuk menghindari isu data tidak seimbang (*imbalanced data*) yang memicu bias model, dilakukan eliminasi data netral (rating 3) dan penerapan *stratified random undersampling*. Proses ini menghasilkan dataset seimbang dengan total **10.000 sampel** (5.000 sampel Positif dan 5.000 sampel Negatif).

### 3. Pemisahan Data (*Data Splitting*)
Data dipisahkan secara terstratifikasi untuk menjaga distribusi kelas tetap proporsional:
* **Model Klasik (Scikit-Learn):** Menggunakan rasio pembagian **80% data latih** dan **20% data uji**.
* **Model Deep Learning (TensorFlow):** Menggunakan rasio pembagian **70% data latih** dan **30% data uji**.

---

## 📊 Perbandingan Hasil Evaluasi Model

Eksperimen dari ketiga skema pelatihan menghasilkan performa pada data uji sebagai berikut:

| Skema Pelatihan | Algoritma Model | Metode Ekstraksi Fitur | Pembagian Data | Akurasi Testing |
| :---: | :--- | :--- | :---: | :---: |
| **Skema 1 (Terbaik)** | **LinearSVC** | **TF-IDF (N-gram 1,2)** | **80 / 20** | **87.15%** |
| Skema 2 | Logistic Regression | Word2Vec | 80 / 20 | 85.25% |
| Skema 3 | BiLSTM (Deep Learning) | Embedding Layer | 70 / 30 | 85.53% |

*Catatan: Model LinearSVC + TF-IDF dipilih sebagai model utama karena menghasilkan akurasi tertinggi dengan waktu komputasi pelatihan yang sangat efisien.*

---

## 🛠️ Panduan Instalasi dan Penggunaan

### 1. Persiapan Lingkungan (*Environment Setup*)
Jalankan perintah berikut pada terminal Linux/WSL Anda untuk menginstal dependensi dasar, membuat, dan mengaktifkan *virtual environment*:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
