# Studi Komparatif Generative Models — Deep Learning D6421

Kenji Lawrence, NIM 2802463440, kelas LB01.

Repositori GitHub: https://github.com/FireClow/Assignment-DeepLearning-Generative-Models

Laporan: `TUGAS_DL_SESI07_2802463440_KENJI_LAWRENCE.pdf`. Naskah yang dapat diedit ada di `laporan/laporan.md`.

Perbandingan enam model generatif pada Fashion-MNIST, ditambah satu MLP classifier yang dipakai untuk menguji batas fitur diskriminatif saat dipakai sebagai generator.

## Project Overview

Keenam model dilatih pada split yang sama. Vanilla AE, Sparse AE, dan VAE belajar merekonstruksi gambar. Vanilla GAN dan DCGAN belajar memetakan noise menjadi gambar. DDPM belajar memprediksi noise, lalu menghasilkan gambar lewat 1.000 langkah reverse diffusion. MLP 784 → 256 → 64 → 10 dilatih untuk klasifikasi. Bobot dan latent classifier kemudian diuji dengan transpose, decoder terpisah, dan sampling prior.

## Models

- Vanilla AE: encoder 784 → 256 → 128 → 64, decoder simetris dengan sigmoid, loss MSE.
- Sparse AE: arsitektur yang sama ditambah ReLU pada latent dan penalti L1 dengan λ = 0,001.
- VAE: latent diagonal Gaussian 64 dimensi, ELBO = BCE yang dijumlah per piksel + KL ke N(0, I).
- Vanilla GAN: generator dan discriminator fully connected, noise 100, output Tanh, loss BCE-with-logits.
- DCGAN: generator dan discriminator konvolusional untuk 28×28, noise 100.
- DDPM: U-Net kecil, T = 1000, jadwal beta linear 0,0001–0,02, target prediksi epsilon, loss MSE.
- MLP Classifier: 784 → 256 → 64 → 10, CrossEntropy. Decoder terpisah 64 → 256 → 784 dilatih saat classifier dibekukan.

## Dataset

Fashion-MNIST, 70.000 gambar grayscale 28×28. Indeks resmi train dan test digabung, lalu dibagi 56.000 / 7.000 / 7.000 dengan seed 42.

Digest split:

`963816f417b1122c50c162765938984622f50669a4401651192ff30320d55b12`

File indeks: `data/splits/fashion_mnist_seed42.json`. Piksel loader berada di [0, 1]. GAN dan DDPM memetakan sekali ke [-1, 1] di dalam trainer masing-masing.

## Requirements

Dependensi terkunci di `requirements.txt`, termasuk PyTorch `2.15.0+cu132` dan torchvision `0.29.2+cu132`.

## Installation

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Perintah Python di bawah memakai interpreter proyek:

```bash
.venv/bin/python scripts/check_environment.py
```

## Execution

Siapkan data dan cek split:

```bash
.venv/bin/python scripts/check_data.py
```

Perintah training berikut menulis ulang checkpoint dan metrics tahap terkait. Jangan dijalankan jika hasil yang sudah tersimpan ingin dipertahankan.

```bash
.venv/bin/python experiments/run_autoencoders.py
.venv/bin/python experiments/run_gans.py
.venv/bin/python experiments/run_ddpm.py
.venv/bin/python experiments/run_classifier.py
```

`run_classifier.py` juga melatih frozen decoder. Decoder saja dapat dijalankan setelah fitur `z_MLP` tersimpan:

```bash
.venv/bin/python experiments/run_frozen_decoder.py
```

Perintah ini hanya membaca checkpoint:

```bash
.venv/bin/python visualization/figure_01_reconstruction.py
.venv/bin/python visualization/figure_02_generation.py
.venv/bin/python experiments/run_latent_interpolation.py
.venv/bin/python experiments/measure_autoencoder_latency.py
.venv/bin/python experiments/update_figure_03.py
.venv/bin/python -m pytest tests/test_splits.py tests/test_autoencoders.py tests/test_gans.py tests/test_ddpm.py tests/test_classifier.py -q
```

## Results

| Model | Latent | Test MSE | PSNR | SSIM | Latency batch 64 |
|---|---|---|---|---|---|
| Vanilla AE | 64 | 0,008685 | 21,66 dB | 0,784 | 0,1044 ms |
| Sparse AE | 64 | 0,009865 | 21,05 dB | 0,761 | 0,1089 ms |
| VAE | 64 | 0,015776 | 18,73 dB | 0,683 | 0,1179 ms |
| Vanilla GAN | 100 | N/A | N/A | N/A | 0,144 ms |
| DCGAN | 100 | N/A | N/A | N/A | 1,471 ms |
| DDPM | ruang gambar | N/A | N/A | N/A | 9.543 ms untuk 1.000 langkah |

MLP test accuracy 88,67%. Frozen decoder test MSE 0,016528, PSNR 18,63 dB, SSIM 0,667. Test noise-prediction MSE DDPM 0,035983. Angka itu bukan reconstruction MSE.

FID untuk semua model adalah N/A. Bobot `inception_v3_google-0cc3c7bd.pth` tidak ada di cache lokal, tidak diunduh, dan tidak ada metrik lain yang diberi label FID.

Latency AE, Sparse AE, dan VAE memakai batch 64, warmup 10, 100 iterasi, dan `cuda.synchronize`. VAE memakai mean posterior, bukan sampling. Latency GAN memakai warmup 10 dan 30 iterasi. Latency DDPM adalah satu lintasan 1.000 langkah, warmup 1, tiga iterasi. Angka-angka itu tidak memakai protokol yang sama.

Figur:

- `outputs/figures/figure_01_reconstruction.png`
- `outputs/figures/figure_02_generation.png`
- `outputs/figures/figure_03_classifier_proof.png`
- `outputs/figures/figure_04_latent_interpolation.png`
- `outputs/figures/figure_05_training_curves.png`
- `outputs/figures/ddpm_denoising_trajectory.png`

Tabel angka: `outputs/metrics/model_comparison.csv`.

## Reproducibility

Seed 42 mengunci split, shuffle training, noise tetap GAN, dan sampling prior MLP. Konfigurasi ada di `configs/data.yaml`, `configs/autoencoder_config.py`, `configs/gan_config.py`, `configs/ddpm_config.py`, dan `configs/classifier_config.py`. Checkpoint terbaik atau final berada di `outputs/checkpoints/`. History epoch berada di `outputs/logs/*_history.json`.

Interpolasi AE memakai dua gambar test kelas berbeda yang sama untuk Vanilla AE, Sparse AE, dan VAE. VAE memakai mean posterior. Interpolasi GAN memakai dua vektor N(0, I) berdimensi 100 dari seed 42, di ruang noise, bukan hasil encoder.

## Limitations

FID tidak dihitung. Vanilla AE, Sparse AE, VAE, dan DDPM masih membaik pada epoch terakhir, jadi training tidak dinyatakan konvergen. GAN tidak punya reconstruction MSE, PSNR, atau SSIM. Noise MSE DDPM mengukur prediksi epsilon, bukan piksel. Latency diukur pada RTX 4060 Laptop dan tidak dipindahkan ke mesin lain.

## Repository Structure

- `configs/` konfigurasi data dan tiap keluarga model.
- `data/` loader Fashion-MNIST dan indeks split.
- `models/` autoencoder, GAN, diffusion, dan classifier.
- `training/` loop training dan checkpoint.
- `evaluation/` metrik rekonstruksi, FID status, latency, dan analisis classifier.
- `experiments/` skrip training dan evaluasi.
- `visualization/` figur laporan.
- `tests/` unit test.
- `outputs/` checkpoint, log, metrics, dan figur.
