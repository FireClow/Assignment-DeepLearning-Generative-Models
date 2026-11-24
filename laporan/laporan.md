# Studi Komparatif Empiris Model Generatif pada Fashion-MNIST

Nama: Kenji Lawrence

NIM: 2802463440

Kelas: LB01

Mata kuliah: Deep Learning (D6421)

Dataset: Fashion-MNIST

Random seed: 42

Repositori GitHub: https://github.com/FireClow/Assignment-DeepLearning-Generative-Models

Naskah ini adalah sumber yang dapat diedit. PDF paginasi dihasilkan oleh `laporan/build_pdf.py`.

## Abstrak

Studi ini membandingkan enam model generatif pada satu split Fashion-MNIST yang dikunci. Vanilla AE, Sparse AE, dan VAE belajar merekonstruksi citra. Vanilla GAN dan DCGAN memetakan noise menjadi citra. DDPM dilatih untuk memprediksi noise, lalu mengambil sampel lewat 1.000 langkah mundur. MLP 784 → 256 → 64 → 10 dilatih sampai akurasi test 88,67 persen, kemudian bobot dan latent-nya diuji dengan transpose, decoder beku, dan sampling prior.

Pada 7.000 citra test, Vanilla AE memberi MSE 0,008685, PSNR 21,66 dB, dan SSIM 0,784. Sparse AE sedikit lebih lemah, dan VAE paling lemah di antara ketiganya karena objektifnya adalah ELBO, bukan MSE. Sampel GAN tetap beragam dan tidak memicu flag kestabilan. DDPM masih menurunkan validation noise MSE pada epoch 30. FID tidak dihitung karena bobot InceptionV3 tidak ada di cache dan tidak diunduh. AE, Sparse AE, VAE, dan DDPM belum terbukti konvergen pada batas epoch.

## II. Landasan teori dan metodologi

Fashion-MNIST berisi 70.000 citra grayscale 28×28 dari sepuluh kelas pakaian. Indeks resmi train dan test digabung, lalu dibagi 56.000, 7.000, dan 7.000 dengan generator privat seed 42. Digest SHA-256 split adalah `963816f417b1122c50c162765938984622f50669a4401651192ff30320d55b12`. Loader menerapkan ToTensor sehingga piksel berada di [0, 1]. Batch training 128. Test set hanya dipakai setelah checkpoint dipilih. GAN dan DDPM memetakan sekali ke [−1, 1] di dalam trainer.

Vanilla AE: 784 → 256 ReLU → 128 ReLU → z 64 tanpa aktivasi, lalu decoder simetris dengan sigmoid.

`L_AE = mean((x - x_hat)^2)`

Sparse AE menambah ReLU pada z. Penalti implementasi adalah rata-rata nilai absolut, bukan jumlah L1 mentah.

`L_sparse = mean((x - x_hat)^2) + 0,001 * mean(|z|)`

VAE memakai kepala mu dan logvar. Rekonstruksi adalah BCE yang dijumlah per 784 piksel lalu dirata-ratakan. KL tertutup menuju N(0, I), beta = 1.

`L_VAE = mean(sum_pixels BCE) + KL`

`KL = -0,5 * mean(sum_j (1 + logvar_j - mu_j^2 - exp(logvar_j)))`

Adam autoencoder 0,001, maksimum 30 epoch, patience 5. Validation loss masih turun pada epoch 30.

GAN memakai noise 100 dimensi, keluaran Tanh, dan logit discriminator. BCE-with-logits tidak dibagi dua lagi. Generator non-saturating. Saat langkah generator, discriminator tetap pada train mode agar BatchNorm DCGAN memakai statistik batch.

`L_D = BCEWithLogits(D(x), 1) + BCEWithLogits(D(G(z)), 0)`

`L_G = BCEWithLogits(D(G(z)), 1)`

Adam GAN 0,0002, beta (0,5, 0,999), 50 epoch. DCGAN: 1×1 → 7×7×128 → 14×14×64 → 28×28×1. Discriminator meratakan 6272 fitur.

DDPM memakai U-Net 2.640.961 parameter, T = 1000, beta linear 0,0001 sampai 0,02. Loss adalah MSE prediksi epsilon, bukan MSE piksel.

`x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`

`L_DDPM = mean((epsilon_hat - epsilon)^2)`

MLP memakai CrossEntropy pada logit, sehingga objektifnya mendekati P(y|x). Transpose tidak memakai bias:

`z_hat = ReLU(W3^T y_c)`, `h_hat = ReLU(W2^T z_hat)`, `x_hat = Sigmoid(W1^T h_hat)`

`x_hat = Sigmoid(W1^T ReLU(W2^T z_MLP))`

Diagram tensor ada di `laporan/diagram_arsitektur.png`.

## III. Pembuktian MLP

Checkpoint terbaik epoch 15. Akurasi train 92,71 persen, validation 89,61 persen, test 88,67 persen. Macro F1 0,886. Trouser F1 0,987. Shirt F1 0,701. Matriks confusion tidak disimpan sebagai gambar.

W1 256×784, W2 64×256, dan W3 10×64 tidak persegi, dan ReLU tidak invertible, jadi transpose bukan inverse. Top-down menghasilkan bercak. Latent transpose menghasilkan pola kotak. Frozen decoder masih membentuk siluet: MSE 0,016528, PSNR 18,63 dB, SSIM 0,667, terhadap Vanilla AE 0,008685, 21,66 dB, dan 0,784. Detail intra-kelas hilang, bentuk kasar bertahan. Enam dari 64 dimensi prior memiliki simpangan baku nol. Enam belas sampel diagonal, seed 42, rata-rata piksel 0,278, bukan pakaian utuh.

## IV. Tabel metrik

Sumber: `outputs/metrics/model_comparison.csv`.

| Model | Dim | s/epoch | Total s | Latency ms | MSE | PSNR | SSIM | FID | Catatan |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Vanilla AE | 64 | 3,91 | 117,84 | 0,1044 | 0,00868499 | 21,65924 | 0,78396 | N/A | terbaik di epoch 30/30 |
| Sparse AE | 64 | 3,95 | 118,97 | 0,1089 | 0,00986527 | 21,04701 | 0,76079 | N/A | terbaik di epoch 30/30 |
| VAE | 64 | 4,24 | 127,77 | 0,1179 | 0,01577550 | 18,73336 | 0,68282 | N/A | ELBO, epoch 30/30 |
| Vanilla GAN | 100 | 2,15 | 109,41 | 0,1439 | N/A | N/A | N/A | N/A | D loss 1,380 |
| DCGAN | 100 | 3,35 | 169,13 | 1,4705 | N/A | N/A | N/A | N/A | akurasi D sekitar 0,77 |
| DDPM | citra | 28,82 | 865,66 | 9542,6607 | N/A | N/A | N/A | N/A | noise MSE, bukan MSE piksel |

Latency AE: warmup 10, 100 iterasi, VAE memakai mu. GAN: warmup 10, 30 iterasi. DDPM: warmup 1, 3 iterasi, 1.000 langkah. FID N/A karena bobot InceptionV3 tidak diunduh.

## V. Figur

- Figur 1: `outputs/figures/figure_01_reconstruction.png`
- Figur 2: `outputs/figures/figure_02_generation.png`
- Trajektori: `outputs/figures/ddpm_denoising_trajectory.png`
- Figur 3: `outputs/figures/figure_03_classifier_proof.png`
- Figur 4: `outputs/figures/figure_04_latent_interpolation.png`
- Figur 5: `outputs/figures/figure_05_training_curves.png`

AE merekonstruksi siluet dengan tepi lembut. DCGAN dan DDPM lebih tegas. Transpose classifier gagal menjadi dekoder, sedangkan frozen decoder masih kasar tetapi mengenali bentuk. Interpolasi AE dari Dress ke T-shirt/top halus. Interpolasi GAN berada di ruang noise dan memperlihatkan artefak saat objek berganti.

## VI. Q1–Q6

Q1. Urutan fidelitas adalah Vanilla AE, Sparse AE, lalu VAE. mean(|z|) Sparse AE turun dari 0,696 ke 0,300, tetapi suku terbobot kecil dibanding MSE. FID tidak tersedia.

Q2. Loss piksel menghaluskan tepi karena menghukum rata-rata. Loss adversarial menolak citra yang terlalu lembut. Sparse AE menurunkan magnitudo latent tanpa studi peta fitur terpisah.

Q3. Vanilla GAN berakhir dekat 2 ln 2 pada D loss dan ln 2 pada G loss. DCGAN menjaga akurasi discriminator sekitar 0,77 tanpa flag dan tanpa sampel yang menyempit menjadi satu kelas. DDPM lebih tenang pada objektif noise, tetapi sampling-nya 9,54 detik. Kualitas dan mode coverage belum diukur FID.

Q4. Jalur AE yang diamati tidak menunjukkan latent void. Artefak ada pada pergantian objek Vanilla GAN dan distorsi endpoint DCGAN. Endpoint GAN bukan citra ter-encode.

Q5. Classifier mendekati P(y|x), membuang variasi intra-kelas, memakai bobot non-invertible, dan tidak memiliki prior generatif. Frozen decoder membuktikan sisa informasi bentuk, bukan kemampuan generasi.

Q6. Tidak ada uji cacat manufaktur, jadi rekomendasi anomali hanya inferensi dari MSE pakaian normal: Vanilla AE adalah baseline yang lebih rapat. Latency di bawah 30 ms diukur pada RTX 4060 Laptop, bukan ponsel. Untuk aset visual, DDPM terlihat lebih utuh pada Figur 2 dan jauh lebih lambat. DCGAN adalah draf yang cepat. Tidak ada pemenang FID.
