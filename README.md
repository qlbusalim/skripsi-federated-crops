# Federated Crops Production Prediction

Kerangka penelitian skripsi untuk memprediksi produksi padi menggunakan data cuaca NASA POWER dan data pertanian (padi) tingkat provinsi. Repository ini masih berada pada tahap pembangunan pipeline dan perbandingan model.

## Status saat ini

- Data cuaca sudah dinormalisasi dan diagregasi per provinsi.
- Data pertanian sudah dinormalisasi dari `data_paddy.xlsx`.
- Dataset gabungan provinsi sudah tersedia.
<<<<<<< HEAD
- Model yang akan dibandingkan: RNN, LSTM, GRU.
=======
- Model RNN, LSTM, GRU, BiLSTM, dan BiGRU tersedia sebagai kerangka regresi.
>>>>>>> 18900a1 (chore restore local notebook changes)
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

## Eksperimen centralized per-client

Eksperimen centralized melatih satu model independen untuk setiap client
(satu client untuk setiap provinsi). Tidak ada pertukaran bobot atau agregasi
FedAvg pada tahap ini. Dengan dataset saat ini, lima model dilatih: Banten,
DI Yogyakarta, Jawa Barat, Jawa Tengah, dan Jawa Timur.

```powershell
python -m experiments.run_centralized
```

Setiap client memiliki sequence, temporal split, scaler, checkpoint, history,
prediksi, dan metrics sendiri. Ringkasan metrics tersedia di
`results/centralized/all_client_metrics.csv`.

Kolom metrics membedakan `training_mae`, `training_rmse`, `testing_mae`, dan
`testing_rmse`. Nama model mengikuti konfigurasi, misalnya `rnn_1` sampai
`rnn_5`; nama tersebut juga tercatat pada file metrics dan prediksi.

## Tuning centralized dengan Optuna

Jalankan tuning dari environment `torch-flwr`:

```powershell
python -m experiments.tune_centralized
```

Setiap trial menggunakan hyperparameter yang sama untuk kelima client, tetapi
masing-masing client tetap dilatih dengan modelnya sendiri. Objective Optuna
adalah rata-rata validation MAE kelima client; data test tidak digunakan selama
tuning. Hasil tuning disimpan di `results/optuna/`, termasuk `best_config.yaml`,
`best_params.yaml`, `trials.csv`, dan database study.

Konfigurasi ruang pencarian berada pada bagian `tuning` di
`configs/baseline.yaml`. Konfigurasi terbaik dapat digunakan untuk menjalankan
kembali centralized dan kemudian menjadi konfigurasi arsitektur yang sama pada
client dan global model decentralized.

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
