# Federated Crops Production Prediction

Kerangka penelitian skripsi untuk memprediksi produksi padi menggunakan data cuaca NASA POWER dan data pertanian (padi) tingkat provinsi. Repository ini masih berada pada tahap pembangunan pipeline dan perbandingan model.

## Status saat ini

- Data cuaca sudah dinormalisasi dan diagregasi per provinsi.
- Data pertanian sudah dinormalisasi dari `data_paddy.xlsx`.
- Dataset gabungan provinsi sudah tersedia.
- Model yang akan dibandingkan: RNN, LSTM, GRU.
- Loss training ditetapkan menggunakan MSE.
- Evaluasi ditetapkan menggunakan MAE dan RMSE.
- Split temporal, training penuh, dan simulasi FL masih disiapkan sebagai tahap berikutnya.

## Struktur repository

```text
configs/       konfigurasi eksperimen terpusat
data/raw/      data mentah; tidak dimasukkan ke Git
data/processed/hasil preprocessing dan dataset gabungan
experiments/   kerangka eksperimen centralized dan federated
results/       laporan, metric, checkpoint, dan visualisasi
scripts/       entry point preprocessing
src/data/      loader, cleaning, agregasi, merging, sequence
src/models/    model recurrent dan factory
src/training/  trainer, loss, callback
src/federated/ client, server, partition, aggregation
src/evaluation/metric, evaluator, dan plot
```

## Konfigurasi

Parameter utama berada di [`configs/baseline.yaml`](configs/baseline.yaml). Mode eksperimen dapat diubah melalui:

```yaml
project:
  mode: centralized
```

Nilai yang tersedia adalah `centralized` dan `federated`. Konfigurasi ini masih berupa kerangka dan akan diperluas ketika desain eksperimen sudah final.

## Environment

Environment sementara disimpan di [`environment.yml`](environment.yml) berdasarkan environment Conda `torch-flwr` yang digunakan untuk pengembangan.

```powershell
conda env create -f environment.yml
conda activate torch-flwr
```

## Menjalankan pipeline data

```powershell
python main.py prepare-weather
python main.py prepare-agriculture
python main.py build-dataset
```

Atau sekaligus:

```powershell
python main.py prepare-data
```

Output utama:

```text
data/processed/weather_monthly_by_region.csv
data/processed/weather_monthly_by_province.csv
data/processed/agriculture_monthly_by_province.csv
data/processed/final_province_dataset.csv
```

## Preview eksperimen centralized

Kerangka centralized digunakan untuk membandingkan model tanpa federated learning.

```powershell
python -m experiments.run_centralized
```

Tahap ini baru memuat dataset, membuat sequence, membuat model, dan menyiapkan MSE loss. Split dan training belum diaktifkan.

## Preview eksperimen federated

```powershell
python -m experiments.run_federated
```

Komponen client, server, partition, dan FedAvg sudah disiapkan sebagai kerangka. Detail dataloader, simulasi Flower, dan protokol eksperimen masih akan ditetapkan kemudian.

## Dokumentasi skripsi

### Latar Belakang

_Placeholder: ntar ya nunggu kelar di draft skripsi._

### Rumusan Masalah

_Placeholder._

### Tujuan Penelitian

_Placeholder._

### Dataset dan Sumber Data

Data cuaca berasal dari NASA POWER. Data pertanian berasal dari workbook provinsi. File mentah sengaja tidak dimasukkan ke repository; sumber, struktur, periode, dan checksum perlu didokumentasikan pada tahap berikutnya.

### Preprocessing

_Placeholder: jelaskan normalisasi data, agregasi provinsi, penanganan nilai hilang (keknya gaada), dan pembuatan sequence._

### Model

_Placeholder: lihat hasil vanilla RNN sama GRU dulu._

### Desain Centralized Learning

_Placeholder: rencana mau tes dulu vanilla RNN sama GRU karena keterbatasan panjang data._

### Desain Federated Learning

_Placeholder: pembagian client berdasarkan provinsi dan mekanisme FedAvg..._

### Skenario Eksperimen

_Placeholder._

### Metrik Evaluasi

Metrik yang digunakan adalah Mean Absolute Error (MAE) dan Root Mean Squared Error (RMSE). Loss training menggunakan Mean Squared Error (MSE).

### Hasil dan Pembahasan

_Placeholder._

### Kesimpulan dan Saran

_Placeholder._
