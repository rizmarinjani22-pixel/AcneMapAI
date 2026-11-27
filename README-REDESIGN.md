# AcneMap AI — UI Redesign & Dynamic Skincare Insight

## Perubahan
- Antarmuka diperbarui dengan gaya modern healthcare, sidebar lavender, kartu statistik, dan layout responsif.
- Navigasi sidebar digunakan di dashboard, hasil analisis, riwayat, galeri, rutinitas, dan skincare insight.
- Hasil prediksi disimpan pada kolom `prediction_json` agar konsisten dan tidak perlu menjalankan ulang model setiap kali halaman hasil dibuka.
- Halaman rekomendasi membaca hasil analisis terbaru atau analisis tertentu melalui `?analysis_id=...`.
- Rekomendasi edukatif disesuaikan dengan kelas severity CNN, jumlah kandidat lesi, dan zona dengan jumlah kandidat tertinggi.
- Bila severity tidak tersedia, aplikasi menampilkan panduan dasar dan tidak menebak kondisi.

## Menjalankan
Gunakan environment Python project yang sudah memiliki dependensi Flask, PyTorch, torchvision, MediaPipe, Ultralytics, Pillow, dan NumPy. Jalankan:
`python app.py`

Saat aplikasi pertama kali dijalankan, `app.py` menambahkan kolom `prediction_json` ke database lama secara otomatis (tanpa menghapus riwayat).

## Catatan ilmiah & batasan
YOLO pada project ini mendeteksi kandidat lesi; model tidak mengklasifikasikan jenis lesi atau tipe kulit. Rekomendasi merupakan edukasi umum, bukan diagnosis atau resep medis. Kategori severity adalah keluaran model CNN yang perlu divalidasi secara klinis. Jerawat berat/nyeri/berbekas atau tidak membaik perlu dievaluasi dokter kulit.

## Pemeriksaan
- Seluruh file Python lolos pemeriksaan sintaks.
- Seluruh template Jinja lolos parsing.
- Uji runtime Flask dan inference model belum dapat dijalankan pada environment pemeriksaan ini karena Flask tidak tersedia di runtime tersebut.
