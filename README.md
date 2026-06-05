# Rice Image Classification (CNN)

Proyek ini merupakan implementasi model *Deep Learning* menggunakan arsitektur **Convolutional Neural Network (CNN)** berbasis `Sequential` untuk mengklasifikasikan 5 varietas beras yang berbeda. Model dikembangkan menggunakan TensorFlow/Keras dan dioptimalkan agar dapat dideploy ke berbagai platform dengan mengonversinya ke format **SavedModel**, **TF-Lite**, dan **TensorFlow.js (TFJS)**.

Proyek ini disusun dan diselesaikan menggunakan berkas utama `klasifikasi_gambar.ipynb` serta pengelolaan dependensi melalui berkas `requirements.txt`.

## 📌 Deskripsi Proyek & Spesifikasi Dataset

Proyek ini memenuhi seluruh kriteria submission evaluasi klasifikasi gambar sebagai berikut:
* **Sumber Dataset:** [Rice Image Dataset (Kaggle)](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset) diunduh via library `kagglehub`.
* **Total Data & Kelas:** 75.000 gambar resolusi asli $250 \times 250$ piksel (memenuhi kriteria $>1000$ gambar).
* **Varietas Beras (5 Kelas):** `Arborio`, `Basmati`, `Ipsala`, `Jasmine`, dan `Karacadag`.
* **Dataset yang Tidak Digunakan:** Menghindari penggunaan dataset yang sudah umum atau pernah digunakan di latihan kelas seperti *Rock, Paper, Scissors* atau *X-Ray* demi menjaga orisinalitas proyek.
* **Pembagian Dataset (Split):**
    * **Training Set (80%):** 60.000 gambar (diaplikasikan *Data Augmentation*: `RandomFlip`, `RandomRotation`, `RandomZoom`).
    * **Validation Set (10%):** 7.500 gambar.
    * **Test Set (10%):** 7.500 gambar.

---

## 🏗️ Arsitektur Model CNN

Sesuai ketentuan, model dibangun menggunakan arsitektur `tf.keras.models.Sequential` dengan memanfaatkan `Conv2D` dan `MaxPooling2D` sebagai basis ekstraksi fitur:

1.  **Rescaling Layer:** Mengubah rentang nilai piksel dari $[0, 255]$ menjadi $[0, 1]$ dengan resolusi input disesuaikan ke $150 \times 150$ piksel.
2.  **4 Blok Konvolusi & Pooling:**
    * `Conv2D` (32 filter, kernel $3 \times 3$, fungsi aktivasi ReLU) + `MaxPooling2D` ($2 \times 2$)
    * `Conv2D` (64 filter, kernel $3 \times 3$, fungsi aktivasi ReLU) + `MaxPooling2D` ($2 \times 2$)
    * `Conv2D` (128 filter, kernel $3 \times 3$, fungsi aktivasi ReLU) + `MaxPooling2D` ($2 \times 2$)
    * `Conv2D` (256 filter, kernel $3 \times 3$, fungsi aktivasi ReLU) + `MaxPooling2D` ($2 \times 2$)
3.  **Flatten Layer:** Mengubah representasi matriks fitur 2D menjadi vektor 1D.
4.  **Dense Layers (Fully Connected) dengan Regularisasi:**
    * `Dense` (512 unit, aktivasi ReLU) + `Dropout` (0.5) untuk mencegah overfitting.
    * `Dense` (256 unit, aktivasi ReLU) + `Dropout` (0.3) untuk mengurangi kompleksitas model.
5.  **Output Layer:** `Dense` (5 unit sesuai jumlah varietas beras dengan fungsi aktivasi Softmax).

---

## 🚀 Performa & Hasil Pelatihan

Proses pelatihan menggunakan optimizer **Adam** dengan *learning rate* $0.0001$ dan fungsi *loss* `sparse_categorical_crossentropy`. Pelatihan dikontrol menggunakan *custom callback* (`TargetAccuracyCallback`) yang menghentikan iterasi secara otomatis apabila akurasi *training* dan *validation* telah mencapai atau melampaui target minimal 96%.

* **Hasil Plot Akurasi & Loss:** Proses pelatihan divisualisasikan secara lengkap dalam berkas `klasifikasi_gambar.ipynb`. Grafik menunjukkan tren peningkatan akurasi secara konsisten dan penurunan nilai loss yang stabil seiring bertambahnya epoch, membuktikan model tidak mengalami *overfitting* atau *underfitting*.
* **Lama Pelatihan:** Mencapai target pemenuhan kriteria tinggi hanya dalam **2 Epoch** (berkat arsitektur sekuensial yang efisien dan optimasi pipeline `prefetch`).
* **Akurasi Training:** 97.33%
* **Akurasi Validation:** 96.74%
* **Akurasi Test Set:** **96.06%** *(Sangat jauh melampaui kriteria batas minimal evaluasi sebesar 85%)*

---

## 📁 Struktur Direktori Hasil Ekspor Model

Model wajib disimpan dalam berbagai format untuk diintegrasikan ke berbagai platform deployment (Server, Mobile, Web Browser). Model berhasil diekspor ke dalam direktori struktur `submission/` berikut:

```text
submission/
├── saved_model/
│   ├── fingerprint.pb
│   ├── saved_model.pb
│   └── variables/
├── tflite/
│   ├── label.txt
│   └── model.tflite
└── tfjs_model/
