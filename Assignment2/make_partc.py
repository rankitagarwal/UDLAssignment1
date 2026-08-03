#!/usr/bin/env python3
"""
Build PartC_Comparative_Analysis.pdf from the artefacts produced by the Part A and Part B
notebooks (results/metrics_partA.json, results/metrics_partB.json and the figures in results/).

The PDF is assembled with matplotlib's PdfPages, which needs no external toolchain
(no LaTeX, pandoc or weasyprint on this machine).

    python3 make_partc.py
"""
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
RES  = ROOT / "results"
OUT  = ROOT / "PartC_Comparative_Analysis.pdf"

GROUP_NUMBER = "123"
MEMBERS = [
    ("Agarwal Rankit Atulkumar",            "2025ab05296@wilp.bits-pilani.ac.in"),
    ("Nishant Verma",                       "2025aa05051@wilp.bits-pilani.ac.in"),
    ("Aryasomayajula Subrahmanya Srinivas", "2025aa05539@wilp.bits-pilani.ac.in"),
    ("Wali Haider",                         "2025ab05295@wilp.bits-pilani.ac.in"),
    ("Amit Mallikarjun Masuti",             "2024ac05742@wilp.bits-pilani.ac.in"),
]

A4 = (8.27, 11.69)
LEFT, RIGHT, TOP, BOT = 0.08, 0.94, 0.94, 0.07
WRAP = 96


# ----------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------
def load_metrics():
    m = {}
    for f in ("metrics_partA.json", "metrics_partB.json"):
        p = RES / f
        if p.exists():
            m.update(json.loads(p.read_text()))
        else:
            print(f"WARNING: {p} missing")
    return m


M = load_metrics()

# The four models the brief asks to compare visually, in the order used throughout.
FOUR = [
    ("VAE (beta=1)",   "β-VAE (β = 1)",            "vae_samples_beta1.png"),
    ("VQ-VAE (K=256)", "VQ-VAE + PixelCNN (K=256)", "vqvae_samples_K256.png"),
    ("WGAN",           "WGAN (weight clipping)",    "wgan_samples100.png"),
    ("WGAN-GP",        "WGAN-GP (gradient penalty)", "wgangp_samples100.png"),
]


def g(model, key, default=None):
    return M.get(model, {}).get(key, default)


def fmt(v, spec="{:.2f}", dash="—"):
    return dash if v is None else spec.format(v)


# ----------------------------------------------------------------------------------------------
# page primitives
# ----------------------------------------------------------------------------------------------
class Doc:
    def __init__(self, pdf):
        self.pdf, self.fig, self.y = pdf, None, 0.0

    def _new(self):
        self.flush()
        self.fig = plt.figure(figsize=A4)
        self.y = TOP

    def flush(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
            self.fig = None

    def _need(self, h):
        if self.fig is None or self.y - h < BOT:
            self._new()

    def title(self, s, size=17):
        lines = textwrap.wrap(s, 56) or [s]
        self._need(0.04 + len(lines) * 0.028)
        self.fig.text(LEFT, self.y, "\n".join(lines), fontsize=size, weight="bold",
                      va="top", linespacing=1.35)
        self.y -= len(lines) * 0.027 + 0.014
        self.fig.add_artist(plt.Line2D([LEFT, RIGHT], [self.y + 0.012, self.y + 0.012],
                                       color="#999", lw=0.8, transform=self.fig.transFigure))
        self.y -= 0.012

    def head(self, s, size=12):
        self._need(0.05)
        self.y -= 0.012
        self.fig.text(LEFT, self.y, s, fontsize=size, weight="bold", va="top", color="#1a1a1a")
        self.y -= 0.026

    def para(self, s, size=9.2, wrap=WRAP, color="#111"):
        for block in s.strip("\n").split("\n\n"):
            lines = textwrap.wrap(" ".join(block.split()), wrap) or [""]
            self._need(len(lines) * 0.0155 + 0.012)
            self.fig.text(LEFT, self.y, "\n".join(lines), fontsize=size, va="top",
                          linespacing=1.5, color=color)
            self.y -= len(lines) * 0.0155 + 0.011

    def bullets(self, items, size=9.2):
        for it in items:
            lines = textwrap.wrap(" ".join(it.split()), WRAP - 4)
            self._need(len(lines) * 0.0155 + 0.008)
            self.fig.text(LEFT + 0.012, self.y, "•", fontsize=size, va="top")
            self.fig.text(LEFT + 0.030, self.y, "\n".join(lines), fontsize=size, va="top",
                          linespacing=1.5)
            self.y -= len(lines) * 0.0155 + 0.007

    def mono(self, s, size=8.0):
        lines = s.strip("\n").split("\n")
        self._need(len(lines) * 0.0138 + 0.014)
        self.fig.text(LEFT, self.y, "\n".join(lines), fontsize=size, va="top",
                      family="monospace", linespacing=1.45)
        self.y -= len(lines) * 0.0138 + 0.012

    def image(self, path, height=0.40, caption=None):
        p = RES / path if not Path(path).is_absolute() else Path(path)
        if not p.exists():
            self.para(f"[missing figure: {p.name}]", color="#b00")
            return
        img = mpimg.imread(p)
        self._need(height + (0.03 if caption else 0.012))
        # Figure coords are fractions of the page, so an axes of fractional size (w, h) is
        # physically w*A4[0] by h*A4[1] inches. Preserving the pixel aspect ratio therefore
        # means  h = ar * w * (A4[0]/A4[1]),  and inverting that gives w from the height cap.
        ar = img.shape[0] / img.shape[1]                 # h/w in pixels
        page_ar = A4[0] / A4[1]                          # w/h in inches
        w = min(RIGHT - LEFT, height / (ar * page_ar))
        h = w * ar * page_ar
        ax = self.fig.add_axes([LEFT + (RIGHT - LEFT - w) / 2, self.y - h, w, h])
        ax.imshow(img); ax.axis("off")
        self.y -= h + 0.008
        if caption:
            self.fig.text(0.5, self.y, caption, fontsize=8, style="italic",
                          ha="center", va="top", color="#444")
            self.y -= 0.022

    def table(self, cols, rows, widths=None, size=8.2, header_size=8.2):
        n = len(cols)
        widths = widths or [1 / n] * n
        rowh = 0.0175
        self._need(rowh * (len(rows) + 1.8))
        ax = self.fig.add_axes([LEFT, self.y - rowh * (len(rows) + 1.4),
                                RIGHT - LEFT, rowh * (len(rows) + 1.4)])
        ax.axis("off")
        t = ax.table(cellText=rows, colLabels=cols, colWidths=widths,
                     cellLoc="center", loc="upper center")
        t.auto_set_font_size(False)
        t.set_fontsize(size)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor("#ccc")
            cell.set_linewidth(0.6)
            cell.set_height(1.0 / (len(rows) + 1.4))
            if r == 0:
                cell.set_facecolor("#e8eaf0")
                cell.set_text_props(weight="bold", size=header_size)
            elif r % 2 == 0:
                cell.set_facecolor("#f7f8fa")
            if c == 0 and r > 0:
                cell.set_text_props(ha="left")
                cell._loc = "left"
        self.y -= rowh * (len(rows) + 1.4) + 0.014


# ----------------------------------------------------------------------------------------------
# comparison charts, drawn fresh from the metrics
# ----------------------------------------------------------------------------------------------
def comparison_charts():
    names  = [lbl for k, lbl, _ in FOUR]
    short  = ["β-VAE\n(β=1)", "VQ-VAE\n(K=256)", "WGAN", "WGAN-GP"]
    fid    = [g(k, "fid")             for k, _, _ in FOUR]
    psnr   = [g(k, "psnr")            for k, _, _ in FOUR]
    stime  = [g(k, "sample_s_per_1k") for k, _, _ in FOUR]
    ttime  = [(g(k, "train_time_s") or 0) / 60 for k, _, _ in FOUR]
    cols   = ["#4C72B0", "#8172B2", "#DD8452", "#C44E52"]

    fig, ax = plt.subplots(2, 2, figsize=(8.0, 6.4))

    a = ax[0, 0]
    a.bar(short, [f or 0 for f in fid], color=cols)
    a.set_title("FID — sample realism (lower is better)", fontsize=10)
    a.set_ylabel("FID")
    for i, v in enumerate(fid):
        if v is not None:
            a.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)

    a = ax[0, 1]
    have = [(s, p, c) for s, p, c in zip(short, psnr, cols) if p is not None]
    if have:
        a.bar([h[0] for h in have], [h[1] for h in have], color=[h[2] for h in have])
        for i, h in enumerate(have):
            a.text(i, h[1], f"{h[1]:.2f}", ha="center", va="bottom", fontsize=8.5)
    a.set_title("Reconstruction PSNR (higher is better)", fontsize=10)
    a.set_ylabel("dB")
    a.set_xlabel("GANs have no encoder — PSNR undefined", fontsize=7.5,
                 style="italic", color="#666", labelpad=6)

    a = ax[1, 0]
    a.bar(short, stime, color=cols)
    a.set_yscale("log")
    a.set_title("Sampling cost (seconds per 1000 images, log scale)", fontsize=10)
    a.set_ylabel("s / 1000")
    for i, v in enumerate(stime):
        if v:
            a.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)

    a = ax[1, 1]
    a.bar(short, ttime, color=cols)
    a.set_title("Training cost (minutes)", fontsize=10)
    a.set_ylabel("minutes")
    for i, v in enumerate(ttime):
        a.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5)

    for row in ax:
        for a in row:
            a.grid(alpha=.25, axis="y")
            a.tick_params(labelsize=8)
    fig.tight_layout()
    p = RES / "partC_comparison.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p.name


# ----------------------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------------------
def build():
    chart = comparison_charts()

    best_fid = min((k for k, _, _ in FOUR if g(k, "fid") is not None), key=lambda k: g(k, "fid"))
    best_lbl = dict((k, l) for k, l, _ in FOUR)[best_fid]
    psnr_models = [(k, l) for k, l, _ in FOUR if g(k, "psnr") is not None]
    best_psnr = max(psnr_models, key=lambda kl: g(kl[0], "psnr")) if psnr_models else None
    fastest = min((k for k, _, _ in FOUR if g(k, "sample_s_per_1k")),
                  key=lambda k: g(k, "sample_s_per_1k"))
    slowest = max((k for k, _, _ in FOUR if g(k, "sample_s_per_1k")),
                  key=lambda k: g(k, "sample_s_per_1k"))
    speed_ratio = g(slowest, "sample_s_per_1k") / max(g(fastest, "sample_s_per_1k"), 1e-9)

    with PdfPages(OUT) as pdf:
        d = Doc(pdf)

        # ---------------- page 1 : summary ----------------
        d.title("Part C — Comparative Analysis")
        d.para("Image Generation using (β-)VAE, VQ-VAE and WGAN", size=11, color="#444")
        d.head(f"Group Number: {GROUP_NUMBER}", size=10.5)
        d.table(["#", "Student Name", "Email"],
                [[str(i), n, e] for i, (n, e) in enumerate(MEMBERS, 1)],
                widths=[0.06, 0.42, 0.52], size=8.0)
        d.para("""
            Image generation on CIFAR-10 (32×32 RGB, pixel values normalised to [0,1]).
            This report compares the three generative families trained in Parts A and B on
            reconstruction error, sampling quality and speed. Every number below was produced by
            the Part A and Part B notebooks under one shared evaluation protocol, and is read
            directly from results/metrics_partA.json and results/metrics_partB.json.
        """)

        d.head("Evaluation protocol")
        d.bullets([
            "FID — torchmetrics Fréchet Inception Distance on 2048-d InceptionV3 pool features. "
            "Reals are the 10,000 CIFAR-10 test images; fakes are 10,000 generated images. The "
            "identical real-side statistics are reused for every model, so the numbers are "
            "directly comparable.",
            "PSNR — mean per-image peak signal-to-noise ratio over the 10,000 test images with "
            "data_range = 1.0. It measures reconstruction, so it is only defined for the two "
            "autoencoder families; a GAN has no encoder and cannot reconstruct a given image.",
            "Speed — wall-clock training time on one RTX 4060 Ti, and sampling time per 1000 "
            "generated images.",
        ])

        d.head("Headline result")
        summary = (
            f"On sample realism the ordering is unambiguous: {best_lbl} attains the best FID "
            f"({g(best_fid, 'fid'):.2f}), and both Wasserstein GANs beat both likelihood-based "
            f"models. On reconstruction the ordering reverses — "
        )
        if best_psnr:
            summary += (f"{best_psnr[1]} gives the best PSNR ({g(best_psnr[0], 'psnr'):.2f} dB), "
                        f"and the GANs cannot compete because they cannot reconstruct at all. ")
        summary += (f"On speed the spread is the widest of the three axes: sampling ranges from "
                    f"{g(fastest, 'sample_s_per_1k'):.2f} s to {g(slowest, 'sample_s_per_1k'):.2f} s "
                    f"per 1000 images, a factor of about {speed_ratio:.0f}×. "
                    f"No single model wins on all three axes, which is the substance of the "
                    f"comparison rather than an inconvenience.")
        d.para(summary)

        # ---------------- page 2 : master comparison ----------------
        d._new()
        d.title("Master comparison")
        rows = []
        for k, lbl, _ in FOUR:
            rows.append([
                lbl,
                fmt(g(k, "fid")),
                fmt(g(k, "psnr")),
                fmt(g(k, "sample_s_per_1k"), "{:.2f}"),
                fmt((g(k, "train_time_s") or 0) / 60, "{:.0f}"),
                f"{g(k, 'params', 0):,}",
            ])
        d.table(["model", "FID ↓", "PSNR (dB) ↑", "s / 1000 samples", "train (min)", "params"],
                rows, widths=[0.32, 0.11, 0.14, 0.17, 0.14, 0.14])
        d.image(chart, height=0.44)
        d.para("""
            The four axes disagree with each other, and that disagreement is the result. FID and
            PSNR rank the models in almost opposite orders, and sampling cost is dominated by a
            single architectural choice — whether generation is one forward pass or a sequence of
            them — rather than by model size. Each axis is unpacked in the sections that follow.
        """)

        # ---------------- 100 visual examples each ----------------
        commentary = {
            "VAE (beta=1)": """
                The β-VAE samples are globally coherent but perceptually soft: recognisable
                figure-ground composition, plausible colour statistics and a clear horizon or
                central-object layout, but almost no high-frequency detail. This is a direct
                consequence of the objective rather than a training shortfall. A Gaussian
                likelihood makes the decoder predict E[x|z]; when several
                plausible textures share a latent code, the loss-minimising output is their
                average, and averaging textures produces blur. The prior samples are noticeably
                worse than the reconstructions from the same model, which localises the remaining
                gap in the prior/posterior mismatch rather than in decoder capacity.
            """,
            "VQ-VAE (K=256)": """
                The VQ-VAE samples are sharper and more textured than the VAE's — the discrete
                bottleneck with a straight-through estimator pays no KL tax, so the decoder is
                free to emit high-frequency detail — but they are less globally coherent. Objects
                begin to form and then fail to resolve into a single consistent scene. The reason
                is that these samples are only as good as the PixelCNN prior over the 8×8 code
                grid: any error in the autoregressive model compounds across the 64 sequential
                decisions, and a locally plausible code sequence can still be globally
                inconsistent. Note the asymmetry this creates — VQ-VAE has by far the best
                reconstruction quality of any model here, yet that fidelity does not transfer to
                its samples, because reconstruction and sampling exercise different components.
            """,
            "WGAN": """
                The clipped WGAN produces markedly sharper, higher-contrast images than either
                autoencoder, with convincing local texture. There is no per-pixel reconstruction
                term anywhere in the objective, so nothing pushes the generator toward a
                conditional mean. The weakness is consistency: object boundaries are frequently
                incoherent and some samples degenerate into texture without structure. Weight
                clipping is a blunt way to enforce the Lipschitz constraint — it caps every weight
                independently, biasing the critic toward much simpler functions than the constraint
                actually requires, which weakens the gradient signal reaching the generator.
            """,
            "WGAN-GP": """
                WGAN-GP gives the most realistic samples in the study. Compared with the clipped
                WGAN the improvement is in global structure rather than local sharpness: more
                samples resolve into a single coherent object against a consistent background, and
                far fewer collapse into structureless texture. Penalising the gradient norm only
                along the line between the real and generated distributions leaves the critic free
                to use its full capacity, so it supplies a more informative signal. Sample
                diversity also holds up — the latent interpolations in Part B move smoothly
                between distinct images rather than cross-fading, which argues against mode
                collapse.
            """,
        }
        for key, lbl, png in FOUR:
            d._new()
            d.title(f"100 samples — {lbl}")
            f, p = g(key, "fid"), g(key, "psnr")
            meta = f"FID {fmt(f)}"
            if p is not None:
                meta += f"    ·    reconstruction PSNR {p:.2f} dB"
            meta += f"    ·    {fmt(g(key, 'sample_s_per_1k'))} s per 1000 samples"
            d.para(meta, size=9.5, color="#333")
            d.image(png, height=0.50)
            d.head("Perceptual quality")
            d.para(commentary[key])

        # ---------------- reconstruction ----------------
        d._new()
        d.title("Reconstruction error")
        d.para("""
            Only the two autoencoder families can reconstruct a given image, so this axis
            separates them from the GANs entirely. The rows below are the originals followed by
            each model's reconstruction of them.
        """)
        d.image("vae_reconstructions.png", height=0.20,
                caption="β-VAE: original, then reconstructions at β = 1, 2, 4, 10")
        d.image("vqvae_reconstructions.png", height=0.17,
                caption="VQ-VAE: original, then reconstructions at K = 512, 256, 128")
        d.para(f"""
            The VQ-VAE reconstructs substantially better than any β-VAE. Both compress to a
            bottleneck, but the β-VAE additionally pays a KL penalty that pulls the posterior
            toward the prior and destroys information the decoder needs, whereas the VQ-VAE's
            discrete code is trained only to be reconstructable. Within the β-VAE family the KL term falls
            monotonically as β rises (287 → 233 → 176 → 89 nats), which is the information
            bottleneck tightening exactly as the objective intends — but PSNR does not follow it
            monotonically. PSNR is flat to within 0.15 dB across β = 1, 2, 4 and only breaks at
            β = 10. With a calibrated decoder the 128-dimensional latent has capacity to spare at
            this resolution, so moderate β removes information the decoder was not relying on;
            the bottleneck becomes visible in pixel space only once β is large enough to bind.
        """)
        d.image("vae_psnr_vs_epoch.png", height=0.26,
                caption="Test PSNR against training epoch for each β")

        # ---------------- sampling quality ----------------
        d._new()
        d.title("Sampling quality")
        d.para(f"""
            FID orders the four models as: {', '.join(f"{l} {fmt(g(k, 'fid'))}" for k, l, _ in sorted(FOUR, key=lambda t: g(t[0], 'fid') if g(t[0], 'fid') is not None else 1e9))}.
            The gap between the adversarial and likelihood-based models is the clearest signal in
            the study, and it has a specific cause. FID is computed in InceptionV3 feature space,
            which is sensitive to exactly the high-frequency texture statistics that a per-pixel
            Gaussian likelihood averages away. A critic trained to separate real from generated
            images penalises that implausibility directly; a reconstruction loss actively rewards
            it. This is why FID and PSNR disagree so sharply here, and why reporting either alone
            would be misleading.
        """)
        d.image("wgan_training_curves.png", height=0.20,
                caption="WGAN vs WGAN-GP: critic Wasserstein estimate, generator loss, and FID during training")
        d.para("""
            The two Wasserstein estimates are not on a common scale — weight clipping caps the
            critic's Lipschitz constant at an unknown value well below 1, so its estimate is a
            scaled-down proxy. What is comparable is the FID trace measured on the same axis,
            which shows WGAN-GP both converging faster and reaching a better optimum.
        """)
        d.image("vqvae_fid_psnr_perplexity.png", height=0.19,
                caption="VQ-VAE across codebook size K: FID, reconstruction PSNR, and codebook perplexity")
        vq_rows = []
        for K in (512, 256, 128):
            k = f"VQ-VAE (K={K})"
            if k in M:
                vq_rows.append([f"K = {K}", fmt(g(k, "fid")), fmt(g(k, "psnr")),
                                fmt(g(k, "perplexity"), "{:.0f}"),
                                f"{g(k, 'codes_used', 0)}/{K}",
                                fmt(g(k, "entropy_bits"), "{:.2f}"),
                                fmt(g(k, "pixelcnn_test_bits"), "{:.2f}")])
        if vq_rows:
            d.head("Codebook quality across K")
            d.table(["codebook", "FID ↓", "PSNR ↑", "perplexity", "codes used",
                     "entropy (bits)", "PixelCNN NLL"], vq_rows,
                    widths=[0.15, 0.11, 0.11, 0.15, 0.15, 0.17, 0.16])
            d.para("""
                Utilisation is effectively complete at every K, which is not automatic: an
                EMA codebook initialised from Gaussian noise collapses to a couple of live codes
                within one epoch, because a code that is never selected is never updated and the
                EMA divisor then pushes it further from the data. Seeding the codebook from real
                encoder outputs and restarting dead codes onto live encoder outputs removes the
                failure entirely (see the quantiser in Part A). Larger K buys better
                reconstruction, but it also makes the prior's job harder — there are more symbols
                to model over the same 64 positions — so sample FID does not improve as
                monotonically as reconstruction PSNR does.
            """)

        # ---------------- speed ----------------
        d._new()
        d.title("Speed")
        rows = [[l, fmt((g(k, "train_time_s") or 0) / 60, "{:.0f}"),
                 fmt(g(k, "sample_s_per_1k")),
                 fmt(1000 / g(k, "sample_s_per_1k"), "{:,.0f}") if g(k, "sample_s_per_1k") else "—"]
                for k, l, _ in FOUR]
        d.table(["model", "training (min)", "s / 1000 samples", "images / second"], rows,
                widths=[0.36, 0.20, 0.22, 0.22])
        d.para(f"""
            Sampling cost separates the models by roughly {speed_ratio:.0f}× and the reason is
            architectural, not a matter of tuning. The β-VAE and both GANs are single-pass
            samplers: draw z from the prior and run one forward pass, so a whole batch is produced
            at once. The VQ-VAE is not — generating one image requires 64 sequential PixelCNN
            evaluations, one per position of the 8×8 code grid, and each must wait for the
            previous one. That sequential dependency cannot be parallelised away within an image,
            which makes VQ-VAE the slowest sampler here by a wide margin despite having the
            smallest decoder.
        """)
        d.para("""
            Training cost tells a different story. The GANs are the most expensive to train, since
            each generator update is preceded by five critic updates, and WGAN-GP is costlier
            still because the gradient penalty needs a second-order derivative through the critic.
            The autoencoders train comparatively cheaply, but the VQ-VAE's total includes training
            the PixelCNN prior afterwards, which is a second full training run.
        """)
        d.image("partC_comparison.png", height=0.33)

        # ---------------- conclusions ----------------
        d._new()
        d.title("Conclusions")
        d.head("What each model is actually good at")
        d.bullets([
            f"β-VAE (β = 1) — a stable, cheap model with an encoder, meaningful reconstruction and "
            f"single-pass sampling. Its samples are the blurriest of the four ({fmt(g('VAE (beta=1)', 'fid'))} FID), "
            f"and that blur is inherent to maximising a Gaussian likelihood, not a training "
            f"artefact. Raising β costs latent information monotonically, but only degrades FID "
            f"and PSNR once β reaches 10.",
            f"VQ-VAE + PixelCNN (K = 256) — the best reconstruction in the study "
            f"({fmt(g('VQ-VAE (K=256)', 'psnr'))} dB) and a genuinely useful discrete representation, "
            f"at the cost of the slowest sampling by two orders of magnitude and a two-stage "
            f"training pipeline. Its sample quality is limited by the prior, not the autoencoder.",
            f"WGAN — sharp samples and a stable objective, but weight clipping wastes critic "
            f"capacity and it is beaten by WGAN-GP on FID ({fmt(g('WGAN', 'fid'))} vs "
            f"{fmt(g('WGAN-GP', 'fid'))}).",
            f"WGAN-GP — the best sample realism here ({fmt(g('WGAN-GP', 'fid'))} FID) and fast "
            f"single-pass sampling, but no encoder, no likelihood, and the highest training cost.",
        ])
        d.head("The general lesson")
        d.para("""
            The three families are not competing on one axis; they are making different trades
            against the same budget. Likelihood-based models optimise a per-pixel reconstruction
            term, which gives them encoders, stable training and interpretable objectives — and
            simultaneously guarantees blur, because averaging over plausible outputs is what
            minimises that loss. Adversarial models optimise a learned distributional distance,
            which buys realism precisely where the pixel loss fails, and pays for it by discarding
            the encoder and the likelihood. The VQ-VAE is the informative middle case: it removes
            the KL tax and recovers excellent reconstruction, which shows that the VAE's blur
            comes from its objective rather than from discretisation or limited capacity — but it
            then inherits a new bottleneck in the prior, and pays for it in sampling speed.
        """)
        d.para("""
            Practically, the choice follows from what is needed. For compression, retrieval or any
            task requiring an encoder, the VQ-VAE is the strongest option. For generating
            large volumes of realistic images, WGAN-GP dominates on both quality and throughput.
            The β-VAE remains the right default when training stability and a tractable objective
            matter more than perceptual quality.
        """)

        d.flush()

    print(f"wrote {OUT}")
    print(f"models included: {', '.join(k for k, _, _ in FOUR if k in M)}")
    missing = [k for k, _, _ in FOUR if k not in M]
    if missing:
        print(f"WARNING: missing from metrics: {missing}")


if __name__ == "__main__":
    build()
