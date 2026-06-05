# Proyek Analisis Sentimen Ulasan Aplikasi

Proyek ini merupakan implementasi pemrosesan bahasa alami (*Natural Language Processing* / NLP) untuk melakukan analisis sentimen terhadap ulasan pengguna. Proyek ini dibangun untuk memenuhi kriteria submission akhir, mencakup seluruh alur kerja *Machine Learning* mulai dari pengumpulan data (*scraping*) secara mandiri, prapemrosesan teks, ekstraksi fitur, hingga pelatihan dan evaluasi berbagai arsitektur model.

## Kriteria Penilaian yang Terpenuhi

*   **Data Hasil Scraping Mandiri:** Menggunakan script khusus untuk melakukan ekstraksi data ulasan aplikasi dengan jumlah sampel awal mencapai 20.000 data.
*   **Ekstraksi Fitur & Pelabelan Data:** Pelabelan otomatis berdasarkan rating pengguna dimana teks diproses melalui pembersihan regex dan normalisasi kata gaul. Ekstraksi fitur yang digunakan meliputi TF-IDF, Word2Vec, serta Embedding Layer.
*   **Algoritma Machine Learning:** Melatih dan membandingkan 3 skema model yang berbeda yaitu LinearSVC, Logistic Regression, dan BiLSTM.
*   **Akurasi Minimal 85%:** Model terbaik menggunakan LinearSVC dengan ekstraksi fitur TF-IDF berhasil mencapai tingkat akurasi testing sebesar 87.15%.

## Alur Pemrosesan Data

*   **Prapemrosesan Teks:** Melibatkan proses *lowercasing*, penghapusan URL, *mention*, *hashtag*, normalisasi menggunakan kamus slang bahasa Indonesia, dan penghapusan karakter yang tidak relevan namun mempertahankan tanda seru dan tanya.
*   **Class Balancing:** Menyaring data netral dan melakukan sampling acak untuk menyeimbangkan distribusi kelas sentimen menjadi 5.000 sampel Positif dan 5.000 sampel Negatif.
*   **Pelatihan Model:** Data dipisah dengan pembagian berstrata, menggunakan rasio 80/20 untuk model Scikit-Learn dan 70/30 untuk model Deep Learning.

## Perbandingan Hasil Evaluasi Model

| Skema Pelatihan | Algoritma | Ekstraksi Fitur | Pembagian Data | Akurasi Testing |
| :--- | :--- | :--- | :--- | :--- |
| **Skema 1 (Terbaik)** | **LinearSVC** | **TF-IDF** | **80/20** | **87.15%** |
| Skema 2 | Logistic Regression | Word2Vec | 80/20 | 85.25% |
| Skema 3 | BiLSTM (Deep Learning) | Embedding Layer | 70/30 | 85.53% |

*Data tabel di atas merepresentasikan performa model pada data uji*.

---

## Panduan Instalasi dan Penggunaan

### 1. Persiapan Lingkungan

Jalankan perintah berikut pada terminal Linux/WSL Anda untuk menginstal dependensi dasar dan membuat *virtual environment*:

```
sudo apt update
sudo apt install python3 python3-pip python3-venv -y

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Pengambilan Data (Scraping)

Untuk melakukan *scraping* data terbaru secara mandiri, gunakan script scraping yang telah disediakan untuk mencapai minimal target 3.000 sampel.

```
python3 scraping.py --mode paginated --target 3000
```

### 3. Menjalankan Pelatihan Model

Anda dapat mereproduksi hasil pemrosesan dan metrik akurasi dengan menjalankan file `training.ipynb`. Buka Jupyter Notebook dan jalankan semua kode di dalam sel tersebut.

Proses pelatihan ini secara otomatis akan memproses dataset, melatih algoritma, serta menyimpan seluruh *artifacts* model seperti `tfidf_vectorizer.pkl`, `model_svc.pkl`, `model_lr.pkl`, `w2v_model.pkl`, `label_encoder.pkl`, `tokenizer.json`, dan `model_bilstm.keras` ke dalam direktori lokal.
