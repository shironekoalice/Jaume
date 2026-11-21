"""Generate publication figures for the Jaume-Memory report.

All numbers are transcribed from the experimental records in docs/memory-*.md.
No simulated data: every plotted value is traceable to a logged run.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C = {"mean": "#8c8c8c", "cls": "#a6a6a6", "attn": "#4c72b0",
     "attnmlp": "#2f4f7f", "latent": "#c44e52"}
BAR = "#4c72b0"
NEG = "#c44e52"
POS = "#55a868"


def fig1_jms():
    """JM-S: five pooling arms on SICK-R, 3 seeds."""
    arms = ["B0\nMean", "B1\nCLS", "B2\nAttention", "B2mlp\nAttn+MLP", "B3\nLatent"]
    mean = [0.6089, 0.5903, 0.6078, 0.6091, 0.5923]
    std = [0.0039, 0.0037, 0.0067, 0.0045, 0.0035]
    colors = [C["mean"], C["cls"], C["attn"], C["attnmlp"], C["latent"]]

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    x = np.arange(len(arms))
    ax.bar(x, mean, yerr=std, color=colors, capsize=3,
           error_kw={"lw": 1, "ecolor": "#333333"})
    ax.axhline(mean[3], ls="--", lw=0.9, color="#2f4f7f", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=7)
    ax.set_ylabel("SICK-R Spearman")
    ax.set_ylim(0.555, 0.625)
    ax.annotate("B3 − B2mlp = −0.0168\n(≈3× seed SD)",
                xy=(4, 0.5975), xytext=(3.05, 0.619),
                fontsize=7, color=NEG, ha="center",
                arrowprops=dict(arrowstyle="->", color=NEG, lw=0.9))
    ax.set_title("JM-S: latent slots vs. parameter-matched pooling", fontsize=8.5)
    fig.savefig(os.path.join(OUT, "fig1_jms_sickr.pdf"))
    fig.savefig(os.path.join(OUT, "fig1_jms_sickr.png"))
    plt.close(fig)


def fig2_slot_diversity():
    """Slot diversity vs. score: the falsification of the differentiation hypothesis."""
    names = ["B3\nv1", "C1\ncompetition", "C2\n+iter/GRU", "C3\n+diversity",
             "C4\n16 slots", "C5\nattn pool", "C6\nno GRU"]
    cos = [0.872, 0.825, 0.977, 0.259, 0.985, 0.976, 0.909]
    score = [0.5923, 0.5736, 0.5694, 0.5639, 0.5669, 0.5635, 0.5954]

    fig, ax = plt.subplots(figsize=(3.9, 2.7))
    ax.scatter(cos, score, s=52, color=BAR, zorder=3, edgecolor="white", linewidth=0.7)
    for i, (c, s) in enumerate(zip(cos, score), start=1):
        ax.annotate(str(i), (c, s), color="white", fontsize=6.2,
                    ha="center", va="center", zorder=4, fontweight="bold")
    ax.axhline(0.6091, ls="--", lw=0.9, color="#2f4f7f")
    ax.text(1.03, 0.6106, "B2mlp control (0.6091)", fontsize=6.5,
            color="#2f4f7f", ha="right")
    ax.set_xlabel("inter-slot cosine  (lower = more differentiated)")
    ax.set_ylabel("SICK-R Spearman")
    ax.set_xlim(0.12, 1.06)
    ax.set_ylim(0.558, 0.618)
    legend = ("1 B3 v1   2 C1 competition   3 C2 +iter/GRU   4 C3 +diversity\n"
              "5 C4 16 slots   6 C5 attn pool   7 C6 no GRU")
    ax.text(0.0, -0.34, legend, transform=ax.transAxes, fontsize=6.3,
            va="top", linespacing=1.5)
    ax.set_title("Slot differentiation does not predict quality", fontsize=8.5)
    fig.savefig(os.path.join(OUT, "fig2_slot_diversity.pdf"))
    fig.savefig(os.path.join(OUT, "fig2_slot_diversity.png"))
    plt.close(fig)


def fig3_synthetic():
    """Synthetic cross-chunk task: the only positive result, with the fair concat control."""
    arms = ["reset", "shuffled", "concat\n(unfair)", "concat\n(fair)", "carried"]
    vals = [0.161, 0.162, 0.461, 0.491, 0.492]
    errs = [0.003, 0.018, 0.006, 0.007, 0.012]
    colors = [C["mean"], C["mean"], "#9fb4d4", "#9fb4d4", C["latent"]]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(arms))
    ax.bar(x, vals, yerr=errs, color=colors, capsize=3,
           error_kw={"lw": 1, "ecolor": "#333333"})
    ax.axhline(0.167, ls=":", lw=1.0, color="#666666")
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 0.62)
    ax.annotate("carried ≈ fair concat (+0.001)", xy=(3.75, 0.500),
                xytext=(1.55, 0.585), fontsize=7,
                arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8))
    ax.text(len(arms) - 0.45, 0.183, "chance 0.167", fontsize=6.5,
            color="#666666", ha="right")
    ax.set_title("Synthetic cross-chunk task (3 seeds)", fontsize=8.5)
    fig.savefig(os.path.join(OUT, "fig3_synthetic.pdf"))
    fig.savefig(os.path.join(OUT, "fig3_synthetic.png"))
    plt.close(fig)


def fig4_probe_validity():
    """Probe validity: model score vs lexical-overlap ceiling."""
    probes = ["conflict", "isolation", "order", "far_retr.", "distractor", "aggregation"]
    ceiling = [1.00, 0.63, 0.53, 0.30, 0.23, 0.00]
    best = [1.00, 0.43, 0.60, 0.53, 0.63, 0.87]

    fig, ax = plt.subplots(figsize=(3.6, 2.0))
    y = np.arange(len(probes))
    ax.barh(y + 0.19, ceiling, height=0.36, color="#c9c9c9", label="lexical ceiling")
    ax.barh(y - 0.19, best, height=0.36, color=BAR, label="best model")
    for i, (c, b) in enumerate(zip(ceiling, best)):
        if c >= b:
            ax.text(max(c, b) + 0.02, i, "invalid", va="center",
                    fontsize=6.5, color=NEG, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(probes)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.30)
    ax.set_ylim(len(probes) - 0.45, -0.6)
    ax.set_xlabel("accuracy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.22),
              ncol=2, frameon=False)
    ax.set_title("Four of six probes are solvable without memory", fontsize=8.5)
    fig.savefig(os.path.join(OUT, "fig4_probe_validity.pdf"))
    fig.savefig(os.path.join(OUT, "fig4_probe_validity.png"))
    plt.close(fig)


def fig5_state_decay():
    """Relative state change and inter-block cosine vs. chunk index."""
    chunks = [1, 3, 5, 7, 9, 11]
    rel = [0.3176, 0.2219, 0.1581, 0.1261, 0.1024, 0.0979]
    cos = [0.9770, 0.9934, 0.9970, 0.9978, 0.9988, 0.9992]

    fig, ax1 = plt.subplots(figsize=(3.4, 2.3))
    ax1.plot(chunks, rel, "o-", color=NEG, lw=1.4, ms=4,
             label=r"$|\Delta s| / |s_{i-1}|$")
    ax1.set_xlabel("chunk index $i$")
    ax1.set_ylabel(r"relative state change", color=NEG)
    ax1.tick_params(axis="y", labelcolor=NEG)
    ax1.set_ylim(0.06, 0.35)

    ax2 = ax1.twinx()
    ax2.plot(chunks, cos, "s--", color="#2f4f7f", lw=1.2, ms=3.5,
             label=r"$\cos(s_{i-1}, s_i)$")
    ax2.set_ylabel("consecutive-state cosine", color="#2f4f7f")
    ax2.tick_params(axis="y", labelcolor="#2f4f7f")
    ax2.set_ylim(0.97, 1.001)
    ax2.spines["right"].set_visible(True)

    ax1.set_title("Additive update dilutes new chunks")
    fig.savefig(os.path.join(OUT, "fig5_state_decay.pdf"))
    fig.savefig(os.path.join(OUT, "fig5_state_decay.png"))
    plt.close(fig)


def fig6_gate_g():
    """Forced gate value sweep: no monotonic benefit."""
    g = [0.10, 0.30, 0.50, 0.70, 0.90, 0.99]
    acc = [0.0820, 0.1148, 0.0656, 0.0984, 0.0902, 0.0738]
    chance, se = 0.0847, 0.0252

    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.plot(g, acc, "o-", color=BAR, lw=1.3, ms=4, label="accuracy@1")
    ax.axhline(chance, ls="-", lw=0.9, color="#666666", label="chance")
    ax.fill_between([0.05, 1.02], chance - 2 * se, chance + 2 * se,
                    color="#cccccc", alpha=0.45, lw=0, label=r"chance $\pm 2$SE")
    ax.set_xlabel("forced gate value $g$")
    ax.set_ylabel("accuracy@1")
    ax.set_xlim(0.05, 1.02)
    ax.set_ylim(0, 0.17)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              ncol=3, fontsize=7)
    ax.set_title("Gate value is not the bottleneck")
    fig.savefig(os.path.join(OUT, "fig6_gate_sweep.pdf"))
    fig.savefig(os.path.join(OUT, "fig6_gate_sweep.png"))
    plt.close(fig)


def fig7_ft_destroys():
    """Fine-tuning sweep: separation collapses at every learning rate."""
    lrs = ["untrained", "1e-5", "3e-5", "1e-4"]
    sep = [0.021, 0.0044, 0.0032, 0.0]
    colors = [POS] + [NEG] * 3

    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    x = np.arange(len(lrs))
    ax.bar(x, sep, color=colors)
    for i, v in enumerate(sep):
        ax.text(i, v + 0.0006, f"{v:+.4f}", ha="center", fontsize=7)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(lrs)
    ax.set_xlabel("learning rate (full fine-tuning, 1 epoch)")
    ax.set_ylabel(r"separation $\cos(\mathrm{gold})-\cos(\mathrm{other})$")
    ax.set_ylim(-0.002, 0.026)
    ax.set_title("Full fine-tuning destroys pretrained signal")
    fig.savefig(os.path.join(OUT, "fig7_finetune_sep.pdf"))
    fig.savefig(os.path.join(OUT, "fig7_finetune_sep.png"))
    plt.close(fig)


def fig8_nine_runs():
    """Nine preregistered runs: carried − static, per split."""
    data = {
        "split0": [-0.0029, -0.0136, -0.0507],
        "split1": [-0.0251, -0.0261, -0.0354],
        "split2": [-0.0109, -0.0139, -0.0099],
    }

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for i, (split, vals) in enumerate(data.items()):
        xs = i + (np.arange(len(vals)) - 1) * 0.18
        ax.scatter(xs, vals, s=30, color=BAR, zorder=3)
        ax.hlines(np.mean(vals), i - 0.3, i + 0.3, color=NEG, lw=1.6, zorder=2)
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["split 0", "split 1", "split 2"])
    ax.set_ylabel(r"carried $-$ static  (set F1)")
    ax.set_ylim(-0.058, 0.012)
    ax.text(2.42, -0.052, "9/9 negative\nmean −0.0210", fontsize=7,
            ha="right", color=NEG)
    ax.set_title("Nine preregistered runs (3 splits × 3 seeds)")
    fig.savefig(os.path.join(OUT, "fig8_nine_runs.pdf"))
    fig.savefig(os.path.join(OUT, "fig8_nine_runs.png"))
    plt.close(fig)


def fig9_architecture():
    """Schematic of the audited state pathway and where information is lost."""
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.axis("off")

    boxes = [
        (0.03, 0.66, "chunk $t$\nhidden states"),
        (0.03, 0.22, "state $s_{t-1}$"),
        (0.31, 0.66, "cross-attn\nread $r_t$"),
        (0.31, 0.22, "additive\nupdate"),
        (0.60, 0.22, "gate $g$\n$\\approx 0.52$ const."),
        (0.60, 0.66, "pool\n$\\mathrm{mean}$"),
    ]
    for x, y, t in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.19, 0.20, fill=True,
                                   facecolor="#eef2f8", edgecolor="#4c72b0", lw=1))
        ax.text(x + 0.095, y + 0.10, t, ha="center", va="center", fontsize=7.5)

    def arrow(x1, y1, x2, y2, style="-|>", color="#444444", lw=1.0, ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls))

    arrow(0.22, 0.80, 0.31, 0.75)          # chunk -> read
    arrow(0.22, 0.32, 0.31, 0.32)          # prev state -> update
    arrow(0.405, 0.66, 0.405, 0.42)        # read -> update
    arrow(0.50, 0.32, 0.60, 0.32)          # update -> gate
    arrow(0.695, 0.42, 0.695, 0.66)        # gate -> pool
    arrow(0.50, 0.34, 0.60, 0.74, ls="--", color="#c44e52")  # update -> pool

    ax.text(0.505, 0.575, "new_slots", fontsize=6.5, color=NEG)
    ax.text(0.03, 0.02,
            "collapse in update:   read cos 0.765  $\\rightarrow$  state cos 0.990\n"
            "gate degenerates:      sd 0.132 $\\rightarrow$ 0.006 (weights moved 141%)\n"
            "output path:           uses new_slots; carry is saved but never read",
            fontsize=7, color="#333333", va="bottom", linespacing=1.5)

    ax.set_xlim(0, 0.82)
    ax.set_ylim(0, 0.95)
    fig.savefig(os.path.join(OUT, "fig9_pathway.pdf"))
    fig.savefig(os.path.join(OUT, "fig9_pathway.png"))
    plt.close(fig)


if __name__ == "__main__":
    for fn in (fig1_jms, fig2_slot_diversity, fig3_synthetic, fig4_probe_validity,
               fig5_state_decay, fig6_gate_g, fig7_ft_destroys, fig8_nine_runs,
               fig9_architecture):
        fn()
        print("ok", fn.__name__)
    print("figures ->", OUT)
