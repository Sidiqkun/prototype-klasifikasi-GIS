# Perbandingan Klasifikasi Citra Drone untuk Pemetaan Lahan Pertanian

Prototype untuk membandingkan tiga metode klasifikasi pada satu orthomosaic drone:

| Metode | Pendekatan |
|---|---|
| Random Forest | per-piksel: RGB + ExG + VARI (+NDVI jika ada NIR) + tekstur lokal |
| SVM (RBF) | fitur sama dengan RF, distandardisasi |
| CNN patch | CNN kecil (3 blok konvolusi) pada patch 15x15 piksel di sekitar tiap piksel |

Yang diukur: Overall Accuracy, Kappa, F1-macro, per-kelas, confusion matrix, waktu latih, dan estimasi waktu pemetaan penuh.
"Paling efisien" = metode dengan F1-macro dalam selisih 1 poin dari yang terbaik **dan** total waktu (latih + peta) terkecil.

## Pakai

```bash
pip install -r requirements.txt

# uji cepat dengan data sintetis (tanpa data sendiri)
python bandingkan.py --demo

# data sendiri
python bandingkan.py --image ortho.tif --labels label.tif \
    --names "Padi,Jagung,Tanah,Bangunan" --scale 0.25 --map
```

Jika tidak ada PyTorch/GPU: `--methods rf,svm`.

## Data yang dibutuhkan

1. **Orthomosaic** (GeoTIFF/PNG/JPG), minimal 3 band RGB. Band ke-4 dianggap alpha, kecuali pakai `--nir`.
2. **Raster label** dengan ukuran/area yang sama: nilai `0` = tidak berlabel, `1..K` = kelas.
   Buat di QGIS: digitasi poligon sampel -> *Rasterize (vector to raster)* dengan atribut kelas.
   Nilai label diurutkan naik, lalu dipasangkan dengan `--names`.

Tips: `--scale 0.25`-`0.5` mempercepat banyak untuk ortho besar. Digitasi poligon yang tersebar di seluruh lahan.

## Opsi penting

`--block` (ukuran blok split spasial, default 50 px), `--test-frac` (0.3), `--n-train`/`--n-test` (sampel per kelas, 3000),
`--patch` (15), `--epochs` (15), `--map` (peta penuh), `--seed`, `--out`.

## Output (`hasil/`)

`results.csv`, `report.txt`, `confusion_matrix.png`, `peta_hasil.png` (dengan `--map`).

## Catatan metodologi

- Data latih/uji dipisah **per blok spasial** dengan zona penyangga, supaya piksel bertetangga tidak bocor ke data uji (kalau acak per piksel, akurasi akan terlalu tinggi).
- Sampel uji seimbang per kelas, jadi OA = rata-rata recall kelas. Pakai F1-macro untuk pembanding utama.
- Satu lokasi dan satu waktu pengambilan: hasil tidak otomatis berlaku untuk lahan/musim lain.
- Waktu bergantung perangkat keras; catat spesifikasi (CPU/GPU) di laporan.
- `est_peta_s` = estimasi dari kecepatan prediksi data uji; `peta_aktual_s` terisi bila `--map`.
