"""Paginate the D6421 report. Rendering uses fpdf2, which is not an experiment dependency.

    .venv/bin/python laporan/build_pdf.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TUGAS_DL_SESI07_2802463440_KENJI_LAWRENCE.pdf"
DIAGRAM = ROOT / "laporan" / "diagram_arsitektur.png"
FIG = ROOT / "outputs" / "figures"
FONT = Path("/usr/share/fonts/truetype/dejavu")


def save_diagram(path: Path) -> None:
    panels = [
        (
            "Vanilla AE",
            "x [B,1,28,28]\n784 -> 256 ReLU -> 128 ReLU -> z 64\n64 -> 128 ReLU -> 256 ReLU -> 784 Sigmoid\nLoss: mean((x - x_hat)^2)\nAdam 1e-3, maks 30 epoch, patience 5",
        ),
        (
            "Sparse AE",
            "Encoder sama, ditambah ReLU pada z\nz >= 0, dimensi 64\nLoss: MSE + 0,001 * mean(|z|)\nDecoder sigmoid, piksel (0, 1)\nAdam 1e-3, maks 30 epoch, patience 5",
        ),
        (
            "VAE",
            "784 -> 256 -> 128, lalu mu dan logvar\nz = mu + eps * exp(0,5 logvar)\nDecoder 64 -> 128 -> 256 -> 784 Sigmoid\nLoss: BCE dijumlah per piksel + KL\nAdam 1e-3, beta = 1, maks 30 epoch",
        ),
        (
            "MLP dan decoder beku",
            "784 -> 256 ReLU -> 64 ReLU -> 10 logit\nCrossEntropy, Adam 1e-3, patience 7\nTranspose hanya memakai W, tanpa bias\nDecoder terpisah: 64 -> 256 ReLU -> 784\nClassifier di-freeze saat decoder dilatih",
        ),
        (
            "Vanilla GAN",
            "z [B,100] ~ N(0, I)\n256 -> 512 -> 1024 -> 784 Tanh\nD: 784 -> 512 -> 256 -> 1 logit\nDropout 0,3 pada D, tanpa sigmoid\nAdam 2e-4, beta (0,5, 0,999), 50 epoch",
        ),
        (
            "DCGAN",
            "z [B,100,1,1]\nConvT 7x7 -> 7x7x128\nConvT 4x4 s2 -> 14x14x64 -> 28x28x1 Tanh\nD: Conv s2 28->14->7, flatten 6272 -> logit\nBatchNorm, LeakyReLU 0,2, 50 epoch",
        ),
        (
            "DDPM",
            "U-Net epsilon, bukan citra\n1x28 -> 64x28 -> 128x14 -> 128x7\nnaik lagi ke 1x28, GroupNorm, SiLU\nT = 1000, beta linear 1e-4 .. 0,02\nLoss: MSE(epsilon_hat, epsilon)\nAdam 2e-4, maks 30 epoch, patience 5",
        ),
    ]
    figure, axes = plt.subplots(4, 2, figsize=(8.2, 10.4))
    axes_flat = list(axes.ravel())
    for axis, (title, body) in zip(axes_flat, panels):
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        patch = FancyBboxPatch(
            (0.03, 0.04),
            0.94,
            0.92,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.0,
            edgecolor="#1f2933",
            facecolor="#f7f4ee",
        )
        axis.add_patch(patch)
        axis.text(0.08, 0.86, title, fontsize=12, fontweight="bold", va="top", color="#1f2933")
        axis.text(0.08, 0.72, body, fontsize=9.5, va="top", family="DejaVu Sans", color="#1f2933")
    axes_flat[-1].axis("off")
    figure.suptitle("Dimensi tensor sesuai implementasi", fontsize=13, y=0.995)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


class Report(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(40, 40, 40)
        self.cell(0, 6, f"Kenji Lawrence  |  2802463440  |  LB01  |  {self.page_no()}", align="C")

    def h1(self, text: str) -> None:
        self.set_text_color(20, 20, 20)
        self.set_font("DejaVu", "B", 13)
        self.multi_cell(0, 6.2, text, align="L")
        self.ln(1)

    def h2(self, text: str) -> None:
        self.set_font("DejaVu", "B", 11)
        self.multi_cell(0, 5.4, text, align="L")
        self.ln(0.6)

    def body(self, text: str) -> None:
        self.set_font("DejaVu", "", 10)
        self.multi_cell(0, 4.55, text, align="L")
        self.ln(1.15)

    def formula(self, text: str) -> None:
        self.set_fill_color(244, 241, 234)
        self.set_font("DejaVu", "", 9)
        self.multi_cell(0, 4.3, text, fill=True, align="L")
        self.ln(1.1)

    def caption(self, text: str) -> None:
        self.set_font("DejaVu", "I", 8.5)
        self.multi_cell(0, 3.8, text, align="L")
        self.ln(1.2)

    def picture(self, path: Path, width: float) -> None:
        self.image(path, w=width)


def build() -> None:
    save_diagram(DIAGRAM)
    pdf = Report(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(16, 14, 16)
    pdf.add_font("DejaVu", "", str(FONT / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", str(FONT / "DejaVuSans-Bold.ttf"))
    pdf.add_font("DejaVu", "I", str(FONT / "DejaVuSans-Oblique.ttf"))
    pdf.set_title("Studi Komparatif Empiris Model Generatif")
    pdf.set_author("Kenji Lawrence")

    pdf.add_page()
    pdf.h1("Studi Komparatif Empiris Model Generatif pada Fashion-MNIST")
    pdf.body(
        "Nama: Kenji Lawrence\n"
        "NIM: 2802463440\n"
        "Kelas: LB01\n"
        "Mata kuliah: Deep Learning (D6421)\n"
        "Dataset: Fashion-MNIST\n"
        "Random seed: 42\n"
        "Repositori GitHub: BELUM TERSEDIA — MENUNGGU REPOSITORY PUBLIK"
    )
    pdf.h2("Abstrak")
    pdf.body(
        "Studi ini membandingkan enam model generatif pada satu split Fashion-MNIST yang dikunci. "
        "Vanilla AE, Sparse AE, dan VAE belajar merekonstruksi citra. Vanilla GAN dan DCGAN memetakan "
        "noise menjadi citra. DDPM dilatih untuk memprediksi noise, lalu mengambil sampel lewat 1.000 "
        "langkah mundur. MLP 784-256-64-10 dilatih sampai akurasi test 88,67 persen, kemudian bobot dan "
        "latent-nya diuji dengan transpose, decoder beku, dan sampling prior."
    )
    pdf.body(
        "Pada 7.000 citra test, Vanilla AE memberi MSE 0,008685, PSNR 21,66 dB, dan SSIM 0,784. Sparse AE "
        "sedikit lebih lemah, dan VAE paling lemah di antara ketiganya karena objektifnya adalah ELBO, "
        "bukan MSE. Sampel GAN tetap beragam dan tidak memicu flag kestabilan. DDPM masih menurunkan "
        "validation noise MSE pada epoch 30. FID tidak dihitung karena bobot InceptionV3 tidak ada di cache "
        "dan tidak diunduh. AE, Sparse AE, VAE, dan DDPM belum terbukti konvergen pada batas epoch."
    )

    pdf.h1("II. Landasan teori dan metodologi")
    pdf.h2("Data dan reproduksibilitas")
    pdf.body(
        "Fashion-MNIST berisi 70.000 citra grayscale 28x28 dari sepuluh kelas pakaian. Indeks resmi train "
        "dan test digabung, lalu dibagi 56.000, 7.000, dan 7.000 dengan generator privat seed 42. "
        "Loader menerapkan ToTensor sehingga piksel berada di [0, 1]. Training di-shuffle dengan generator "
        "ber-seed. Validation dan test tidak di-shuffle. Batch training 128. Test set hanya dipakai setelah "
        "checkpoint dipilih. GAN dan DDPM memetakan sekali ke [-1, 1] di dalam trainer, lalu figur mengembalikan "
        "citra ke [0, 1]."
    )
    pdf.body("Digest SHA-256 dari indeks split:")
    pdf.formula("963816f417b1122c50c162765938984622f50669a4401651192ff30320d55b12")
    pdf.h2("Autoencoder")
    pdf.body(
        "Vanilla AE memetakan 784 ke 256, 128, lalu latent 64 tanpa aktivasi. Decoder memetakan 64 ke 128, "
        "256, dan 784 dengan sigmoid. Implementasi memakai mean squared error pada seluruh piksel dan batch, "
        "bukan jumlah kuadrat tanpa normalisasi."
    )
    pdf.formula("L_AE = mean((x - x_hat)^2)")
    pdf.body(
        "Sparse AE menambah ReLU setelah lapisan 64 sehingga z tidak negatif. Penalti yang disimpan adalah "
        "rata-rata nilai absolut, lalu dikalikan lambda 0,001. Bentuk ini setara dengan L1 yang dinormalisasi, "
        "bukan jumlah L1 mentah."
    )
    pdf.formula("L_sparse = mean((x - x_hat)^2) + 0,001 * mean(|z|)")
    pdf.body(
        "VAE memakai backbone sampai 128, lalu dua kepala mu dan logvar. logvar dijepit ke [-20, 20]. "
        "Rekonstruksi adalah binary cross-entropy yang dijumlah pada 784 piksel, lalu dirata-ratakan pada "
        "batch. KL adalah bentuk tertutup ke N(0, I), dengan beta = 1. ELBO sekitar 240 tidak dibandingkan "
        "langsung dengan MSE AE sekitar 0,01. Perbandingan memakai MSE, PSNR, dan SSIM pada citra hasil decode."
    )
    pdf.formula(
        "L_VAE = mean(sum_pixels BCE(x, x_hat)) + KL\n"
        "KL = -0,5 * mean(sum_j (1 + logvar_j - mu_j^2 - exp(logvar_j)))"
    )
    pdf.body(
        "Ketiga model memakai Adam dengan learning rate 0,001, maksimum 30 epoch, dan patience 5 pada "
        "validation loss. Patience tidak pernah memutus training karena validation loss masih turun pada epoch 30."
    )

    pdf.add_page()
    pdf.h2("GAN, DDPM, dan MLP")
    pdf.body(
        "Vanilla GAN dan DCGAN memakai noise 100 dimensi dari N(0, I) dan keluaran Tanh. Discriminator "
        "mengeluarkan logit. Loss adalah BCE-with-logits rata-rata batch. Loss discriminator adalah jumlah "
        "dua suku, bukan rata-rata yang dibagi dua lagi. Generator memakai objektif non-saturating. Saat "
        "langkah generator, parameter discriminator dibekukan tetapi modul tetap pada train mode agar "
        "BatchNorm DCGAN memakai statistik batch palsu, bukan running statistics."
    )
    pdf.formula(
        "L_D = BCEWithLogits(D(x), 1) + BCEWithLogits(D(G(z)), 0)\n"
        "L_G = BCEWithLogits(D(G(z)), 1)"
    )
    pdf.body(
        "Adam GAN memakai learning rate 0,0002 dan beta (0,5, 0,999) selama 50 epoch, tanpa early stopping. "
        "DCGAN generator berjalan 1x1 ke 7x7, 14x14, lalu 28x28. Discriminator turun 28 ke 14 ke 7 dan "
        "meratakan 128 x 7 x 7 = 6272 sebelum logit."
    )
    pdf.body(
        "DDPM memakai U-Net 2.640.961 parameter dengan base channel 64, embedding waktu sinusoidal 256, "
        "GroupNorm, dan SiLU. Keluaran adalah epsilon, tanpa tanh atau sigmoid. Jadwal beta linear dari "
        "0,0001 sampai 0,02 pada T = 1000. Indeks waktu berjalan 0 sampai 999. Sampling mulai dari x_T dan "
        "menurun sampai t = 0. Noise posterior ditambahkan hanya jika t > 0."
    )
    pdf.formula(
        "x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon\n"
        "L_DDPM = mean((epsilon_hat(x_t, t) - epsilon)^2)\n"
        "mu = (1 / sqrt(alpha_t)) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * epsilon_hat)"
    )
    pdf.body(
        "MLP memetakan 784 ke 256, 64, dan 10 logit dengan ReLU pada dua lapisan tersembunyi. Ada bias pada "
        "forward pass. Objektifnya CrossEntropy, yaitu log-likelihood kelas, sehingga model mendekati P(y|x). "
        "Latent z_MLP diambil setelah ReLU kedua dan berdimensi 64. Proyeksi transpose hanya memakai W1, W2, "
        "dan W3. Bias tidak ikut, dan pseudoinverse tidak dipakai."
    )
    pdf.formula(
        "z_hat = ReLU(W3^T y_c),   h_hat = ReLU(W2^T z_hat),   x_hat = Sigmoid(W1^T h_hat)\n"
        "Latent transpose: x_hat = Sigmoid(W1^T ReLU(W2^T z_MLP))"
    )
    pdf.body(
        "Pada layout PyTorch [out, in], transpose untuk vektor baris menjadi perkalian kanan: y @ W3, "
        "z @ W2, dan h @ W1. Optimizer MLP dan decoder beku adalah Adam 0,001, maksimum 50 epoch, patience 7. "
        "Seluruh pelatihan berjalan pada NVIDIA GeForce RTX 4060 Laptop."
    )

    pdf.add_page()
    pdf.h2("Diagram arsitektur")
    pdf.picture(DIAGRAM, 178)
    pdf.caption(
        "Gambar 1. Skema ketujuh jaringan. Angka channel dan ukuran spasial mengikuti modul yang dilatih, "
        "bukan diagram generik."
    )

    pdf.add_page()
    pdf.h1("III. Pembuktian batas MLP classifier")
    pdf.body(
        "Checkpoint terbaik adalah epoch 15, dengan validation loss 0,30218. Training berhenti pada epoch 22 "
        "karena patience 7. Akurasi checkpoint pada train 92,71 persen, validation 89,61 persen, dan test "
        "88,67 persen. Macro precision, recall, dan F1 test adalah 0,88750, 0,88646, dan 0,88647. "
        "Trouser paling kuat dengan F1 0,987. Shirt paling lemah dengan F1 0,701. Matriks confusion tidak "
        "disimpan sebagai berkas terpisah. Angka per kelas ada di classifier_metrics.json."
    )
    pdf.body(
        "Transpose top-down dari sepuluh vektor one-hot menghasilkan bercak abu-abu, bukan pakaian. "
        "W1 berukuran 256x784, W2 64x256, dan W3 10x64. Matriks ini persegi panjang, sehingga W transpose "
        "bukan inverse. ReLU juga membuang setengah garis bilangan. Karena bias tidak ikut diproyeksikan, "
        "operasi ini bahkan bukan jalur maju yang dibalik."
    )
    pdf.body(
        "Latent transpose pada delapan citra test yang sama menghasilkan pola kotak kontras tinggi. Siluet "
        "aslinya tidak kembali. Frozen decoder 64-256-784, yang dilatih 50 epoch saat classifier dibekukan, "
        "masih membentuk siluet. MSE test-nya 0,016528, PSNR 18,63 dB, dan SSIM 0,667, dibandingkan Vanilla AE "
        "0,008685, 21,66 dB, dan 0,784 pada 7.000 citra yang sama. Decoder MLP sekitar 1,9 kali lebih besar "
        "MSE-nya. Bentuk kasar bertahan. Tulisan pada tas dan logo pada kaos hilang lebih banyak. Latent "
        "klasifikasi menyimpan sebagian informasi bentuk, lalu membuang detail spasial dan variasi intra-kelas."
    )
    pdf.body(
        "Prior diambil dari mean dan simpangan baku test z_MLP. Enam dari 64 dimensi memiliki simpangan baku "
        "nol dan dinaikkan ke 1e-6. Enam belas sampel diagonal, seed 42, menghasilkan citra dengan rata-rata "
        "piksel 0,278 dan simpangan baku 0,378. Hasilnya bercak kontras tinggi, bukan pakaian utuh. Prior "
        "diagonal mengabaikan korelasi antar dimensi dan daerah yang terpotong ReLU. Classifier tidak dilatih "
        "ulang pada eksperimen ini, dan ukuran checkpoint Vanilla AE tidak berubah."
    )

    pdf.add_page()
    pdf.h1("IV. Tabel komparasi metrik")
    pdf.body(
        "Nilai di bawah disalin dari outputs/metrics/model_comparison.csv. Reconstruction MSE, PSNR, dan SSIM "
        "berlaku untuk model yang memang dilatih merekonstruksi. Sel kosong pada berkas ditulis N/A."
    )
    rows = [
        ["Model", "Dim", "s/ep", "Total s", "Lat. ms", "MSE", "PSNR", "SSIM", "FID", "Catatan"],
        ["Vanilla AE", "64", "3,91", "117,84", "0,1044", "0,008685", "21,66", "0,784", "N/A", "30/30"],
        ["Sparse AE", "64", "3,95", "118,97", "0,1089", "0,009865", "21,05", "0,761", "N/A", "30/30"],
        ["VAE", "64", "4,24", "127,77", "0,1179", "0,015776", "18,73", "0,683", "N/A", "ELBO 30/30"],
        ["Vanilla GAN", "100", "2,15", "109,41", "0,1439", "N/A", "N/A", "N/A", "N/A", "D 1,380"],
        ["DCGAN", "100", "3,35", "169,13", "1,4705", "N/A", "N/A", "N/A", "N/A", "acc D 0,77"],
        ["DDPM", "citra", "28,82", "865,66", "9542,66", "N/A", "N/A", "N/A", "N/A", "noise 30/30"],
    ]
    widths = [26, 12, 12, 16, 18, 20, 14, 14, 12, 32]
    pdf.set_font("DejaVu", "B", 7)
    pdf.set_fill_color(232, 228, 220)
    for index, row in enumerate(rows):
        pdf.set_font("DejaVu", "B" if index == 0 else "", 7)
        for width, value in zip(widths, row):
            pdf.cell(width, 6.2, value, border=1, align="C", fill=index == 0)
        pdf.ln()
    pdf.ln(2)
    pdf.body(
        "Angka penuh di CSV untuk MSE, PSNR, dan SSIM adalah 0,00868499 / 21,65923977 / 0,78396062 pada "
        "Vanilla AE, 0,00986527 / 21,04700809 / 0,76079118 pada Sparse AE, dan 0,01577550 / 18,73335882 / "
        "0,68281512 pada VAE. Latency AE diukur ulang dari checkpoint: batch 64, eval, inference mode, "
        "warmup 10, 100 iterasi, cuda.synchronize, CUDA RTX 4060 Laptop. VAE pada pengukuran itu memakai "
        "mu lalu decode, tanpa sampling. Rata-rata, median, dan simpangan baku ada di autoencoder_latency.json."
    )
    pdf.body(
        "Latency Vanilla GAN 0,1439 ms dan DCGAN 1,4705 ms memakai warmup 10 dan 30 iterasi. Latency DDPM "
        "9.542,66 ms adalah satu lintasan 1.000 langkah, warmup 1, tiga iterasi. Tiga protokol itu tidak "
        "sama, jadi kolom latency bukan satu perlombaan yang homogen. FID seluruh baris N/A. Berkas bobot "
        "inception_v3_google-0cc3c7bd.pth tidak ada di cache, tidak diunduh, dan tidak diganti metrik lain. "
        "Test noise MSE DDPM 0,035983 mengukur prediksi epsilon, bukan piksel, dan tidak dimasukkan ke kolom MSE."
    )
    pdf.body(
        "Validation loss AE, penalti sparse, ELBO VAE, dan noise MSE DDPM masih terbaik pada epoch terakhir. "
        "Selesainya jadwal epoch tidak dijadikan bukti konvergen. GAN menyelesaikan 50 epoch tanpa flag "
        "non-finite, near-white, near-black, low-variation, atau pemisahan discriminator di atas 0,98. "
        "Itu juga tidak dinyatakan sebagai konvergen."
    )

    pdf.add_page()
    pdf.h1("V. Visualisasi")
    pdf.h2("Figur 1. Rekonstruksi")
    pdf.picture(FIG / "figure_01_reconstruction.png", 178)
    pdf.caption(
        "Figur 1. Delapan citra test pertama, lalu rekonstruksi Vanilla AE, Sparse AE, dan VAE. "
        "Kelas berurutan: Dress, T-shirt/top, T-shirt/top, Ankle boot, Bag, Trouser, Shirt, T-shirt/top."
    )
    pdf.body(
        "Siluet ketiga autoencoder mengikuti citra asli. Tepi tetap lebih halus daripada citra sumber, "
        "terutama pada tulisan tas dan lipatan baju. VAE paling kehilangan kontras. Perbedaan Vanilla AE "
        "dan Sparse AE kecil, sesuai selisih MSE yang juga kecil."
    )

    pdf.add_page()
    pdf.h2("Figur 2. Sampel generatif")
    pdf.picture(FIG / "figure_02_generation.png", 176)
    pdf.caption("Figur 2. Sampel VAE, Vanilla GAN, DCGAN, dan DDPM dari checkpoint yang sudah dilatih.")
    pdf.picture(FIG / "ddpm_denoising_trajectory.png", 178)
    pdf.caption(
        "Trajektori satu rantai DDPM pada tingkat 1000, 875, 750, 625, 500, 375, 250, 125, dan 0. "
        "Tingkat 0 adalah hasil langkah terakhir, tanpa noise tambahan."
    )
    pdf.body(
        "VAE menghasilkan pakaian yang utuh tetapi lembut. Vanilla GAN lebih berbutir dan tetap memuat "
        "beberapa kelas. DCGAN menajamkan sepatu, celana, dan baju. DDPM memberi tepi paling jelas di antara "
        "empat panel. Trajektori menunjukkan struktur pakaian muncul bertahap dari noise, bukan pada satu langkah."
    )

    pdf.add_page()
    pdf.h2("Figur 3. Batas classifier")
    pdf.picture(FIG / "figure_03_classifier_proof.png", 168)
    pdf.caption(
        "Figur 3. Citra asli, transpose top-down, latent transpose, Vanilla AE, frozen decoder, "
        "dan 16 sampel prior. Baris citra memakai delapan contoh test yang sama."
    )
    pdf.body(
        "AE dan frozen decoder membaca citra yang sama. AE lebih tajam. Frozen decoder mempertahankan "
        "kategori visual secara kasar. Prior tidak menyusun pakaian yang utuh. Transpose, baik dari one-hot "
        "maupun dari z_MLP, tidak menjadi dekoder."
    )

    pdf.add_page()
    pdf.h2("Figur 4. Interpolasi linear")
    pdf.picture(FIG / "figure_04_latent_interpolation.png", 176)
    pdf.caption(
        "Figur 4. z(alpha) = (1 - alpha) z_A + alpha z_B pada alpha 0,0 sampai 1,0. "
        "Tiga baris atas berbagi endpoint Dress dan T-shirt/top. Dua baris bawah menginterpolasi noise N(0, I), seed 42."
    )
    pdf.body(
        "Pada AE, Sparse AE, dan VAE, dress memendek menjadi kaos tanpa sobekan geometri. Jalur ini terlihat "
        "terisi, sehingga laporan ini tidak mengklaim latent void pada pasangan tersebut. VAE memakai mu, "
        "bukan sampel acak. Vanilla GAN berpindah dari bentuk gelap berbutir ke sepatu, dengan kolom 0,8 yang "
        "pecah. DCGAN bertahan lebih lama pada bentuk alas kaki, lalu endpoint akhirnya melengkung. Karena "
        "endpoint GAN adalah noise, baris itu tidak bisa dibaca sebagai morph dari dress ke kaos."
    )

    pdf.add_page()
    pdf.h2("Figur 5. Kurva training")
    pdf.picture(FIG / "figure_05_training_curves.png", 178)
    pdf.caption(
        "Figur 5. Kurva loss pada skala masing-masing. Panel bawah mengulang satu trajektori DDPM. "
        "Penalti sparse dan KL tidak dipaksa pada sumbu MSE."
    )
    pdf.body(
        "Kurva AE dan DDPM masih menurun di ujung jadwal. Vanilla GAN berakhir dekat keseimbangan "
        "non-saturating: D loss 1,380 terhadap 2 ln 2 sekitar 1,386, dan G loss 0,717 terhadap ln 2 sekitar "
        "0,693. Akurasi real dan fake epoch 50 adalah 0,496 dan 0,561. DCGAN tidak jatuh ke loss generator "
        "nol. Akurasi discriminator tetap di sekitar 0,77, sementara sampel epoch 50 tetap beragam. "
        "Tidak ada bukti visual mode collapse pada run yang dicatat."
    )

    pdf.h1("VI. Jawaban Q1 sampai Q6")
    pdf.h2("Q1. Fidelitas rekonstruksi")
    pdf.body(
        "Urutan MSE test dari yang terkecil adalah Vanilla AE 0,008685, Sparse AE 0,009865, lalu VAE 0,015776. "
        "PSNR dan SSIM mengikuti urutan yang sama: 21,66 dB dan 0,784; 21,05 dB dan 0,761; 18,73 dB dan 0,683. "
        "Rata-rata |z| Sparse AE turun dari 0,696 pada epoch 1 menjadi 0,300 pada epoch 30. Karena lambda hanya "
        "0,001, suku penalti terbobot sekitar 0,0003, kecil dibanding MSE sekitar 0,01. Penalti menekan magnitudo "
        "latent, tetapi fidelitas piksel tetap dekat dengan Vanilla AE. FID tidak dapat dibandingkan. Bobot "
        "InceptionV3 tidak tersedia, dan tidak ada angka pengganti yang diberi label FID."
    )
    pdf.h2("Q2. Blur, tepi, dan sparsity")
    pdf.body(
        "MSE dan BCE menghukum rata-rata piksel. Jika beberapa tepi sama-sama masuk akal, rata-rata bersyarat "
        "itu jatuh di antara tepi-tepi tersebut, sehingga rekonstruksi AE dan VAE terlihat lembut. Loss "
        "adversarial tidak mencocokkan piksel satu per satu. Generator yang kabur mudah ditolak discriminator, "
        "sehingga DCGAN pada Figur 2 menampilkan kontur yang lebih tegas. Sparse AE tidak membentuk peta fitur "
        "terpisah di laporan ini. Bukti yang ada adalah penurunan mean(|z|) dan sedikit kenaikan MSE. Representasi "
        "menjadi lebih kecil nilainya, bukan serta-merta lebih mudah dibaca sebagai bagian pakaian."
    )

    pdf.add_page()
    pdf.h2("Q3. Dinamika GAN, DDPM, dan trilemma")
    pdf.body(
        "Vanilla GAN bergerak ke daerah loss yang dekat dengan keseimbangan BCE non-saturating, dengan akurasi "
        "discriminator di sekitar separuh. DCGAN menjaga akurasi discriminator lebih tinggi, sekitar 0,77 pada "
        "citra real maupun palsu, dan G loss akhir 1,330. Keduanya menyelesaikan 50 epoch dengan daftar flag kosong. "
        "Sampel tidak menyempit menjadi satu kelas, jadi mode collapse tidak dinyatakan. DDPM punya objektif "
        "supervised pada epsilon. Validation noise MSE turun dari 0,0553 pada epoch 1 ke 0,03646 pada epoch 30 "
        "dan masih menjadi nilai terbaik. Optimisasi ini lebih tenang daripada permainan minimax, tetapi sampel "
        "memerlukan seluruh rantai mundur."
    )
    pdf.body(
        "Pada mesin yang sama, satu forward batch 64 memakan 0,118 ms untuk VAE, 0,144 ms untuk Vanilla GAN, "
        "dan 1,47 ms untuk DCGAN. DDPM membutuhkan 9,54 detik karena 1.000 langkah. Kualitas visual DDPM pada "
        "Figur 2 lebih rapi, kecepatannya jauh lebih rendah, dan cakupan mode tidak diukur dengan FID. Trilemma "
        "terlihat pada sumbu kecepatan versus jumlah langkah. Sumbu kualitas dan cakupan mode belum tertutup angka."
    )
    pdf.h2("Q4. Interpolasi")
    pdf.body(
        "Endpoint AE adalah citra test posisi 0, Dress, indeks pooled 29848, dan posisi 1, T-shirt/top, indeks "
        "11863. Alpha 0 dan 1 sama dengan decode endpoint. Transisi tiga autoencoder halus dan buram. Tidak ada "
        "lubang atau anggota badan yang putus pada pasangan ini, termasuk pada AE deterministik. Klaim latent void "
        "tidak didukung oleh jalur yang diamati. Perbedaan VAE di sini adalah endpoint mu yang deterministik, "
        "bukan jaminan bahwa seluruh ruang latent-nya terisi. Jalur GAN berada di ruang noise. Artefak yang nyata "
        "ada pada Vanilla GAN saat objek berganti dan tekstur pecah, serta pada DCGAN saat sepatu akhir melengkung."
    )
    pdf.h2("Q5. Mengapa latent classifier bukan generator")
    pdf.body(
        "CrossEntropy menyesuaikan P(y|x). Informasi yang tidak membantu memisahkan kelas boleh hilang. Itu "
        "terlihat sebagai information bottleneck: frozen decoder masih mengenal siluet, tetapi kalah dari AE "
        "pada detail. W transpose bukan inverse karena ukuran matriks tidak persegi dan ReLU tidak invertible. "
        "Hasil top-down dan latent transpose menunjukkan kegagalan itu secara langsung. Ruang z_MLP juga tidak "
        "punya prior yang dilatih. Sampling Gaussian diagonal di sekitar statistik test menghasilkan bercak, "
        "bukan manifold pakaian. Jadi z_MLP menyimpan ciri pembeda, bukan distribusi citra P(x)."
    )
    pdf.h2("Q6. Rekomendasi yang masih berupa inferensi")
    pdf.body(
        "Untuk deteksi anomali manufaktur, eksperimen ini tidak memakai citra cacat. Inferensinya hanya dari "
        "reconstruction error pada pakaian normal. Vanilla AE memberi baseline MSE lebih rendah daripada Sparse AE. "
        "Sensitivitas penalti sparse terhadap cacat belum diukur, jadi Sparse AE tidak dinyatakan lebih peka."
    )
    pdf.body(
        "Untuk avatar interaktif di bawah 30 ms, angka yang ada diukur pada RTX 4060 Laptop untuk batch 64, "
        "bukan pada ponsel. VAE, Vanilla GAN, dan DCGAN berada jauh di bawah 30 ms pada GPU itu. Fakta itu tidak "
        "membuktikan anggaran mobile. DDPM 9,54 detik jelas di luar anggaran waktu nyata. Jika harus memilih "
        "kandidat untuk diukur ulang di perangkat target, urutan praktisnya adalah satu forward VAE atau DCGAN, "
        "lalu pengukuran baru."
    )
    pdf.body(
        "Untuk aset visual, perbandingan DCGAN dan DDPM terbatas pada gambar dan biaya sampling. FID tidak ada. "
        "Pada Figur 2, DDPM terlihat lebih tajam dan lebih utuh, dengan harga 1.000 evaluasi jaringan. DCGAN "
        "menghasilkan draf dalam 1,47 ms pada GPU ini. Pilihan industri yang jujur dari bukti ini adalah DCGAN "
        "untuk iterasi cepat dan DDPM untuk kualitas visual yang lebih teliti, bukan pemenang FID."
    )

    pdf.output(OUT)
    print(f"pages={pdf.page} path={OUT}")


if __name__ == "__main__":
    build()
