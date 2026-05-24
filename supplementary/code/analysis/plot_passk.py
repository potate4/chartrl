"""Generate Pass@K curve figures for the paper (ChartQA + ChartFC)."""
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

K = [1, 2, 3, 4]

# ChartQA Pass@K (unbiased estimator) — has Base
CHARTQA = {
    "Base":        [0.5245, 0.6733, 0.7375, 0.7740],
    "GRPO":        [0.6490, 0.7490, 0.7905, 0.8160],
    "GRPO+HCPC":   [0.6285, 0.7457, 0.7990, 0.8280],
    "NSR":         [0.5765, 0.7170, 0.7760, 0.8080],
    "NSR+HCPC":    [0.5755, 0.7083, 0.7660, 0.8000],
}

# ChartFC Pass@K — no Base (was not evaluated on ChartFC)
CHARTFC = {
    "GRPO":        [0.6610, 0.8207, 0.8890, 0.9220],
    "GRPO+HCPC":   [0.6245, 0.7990, 0.8745, 0.9180],
    "NSR":         [0.5840, 0.7610, 0.8405, 0.8840],
    "NSR+HCPC":    [0.6125, 0.7867, 0.8560, 0.8940],
}

STYLE = {
    "Base":      dict(color="#888888", linestyle=":",  marker="o", linewidth=1.4),
    "GRPO":      dict(color="#1f77b4", linestyle="-",  marker="s", linewidth=1.6),
    "GRPO+HCPC": dict(color="#d62728", linestyle="-",  marker="o", linewidth=2.0),
    "NSR":       dict(color="#2ca02c", linestyle="--", marker="^", linewidth=1.4),
    "NSR+HCPC":  dict(color="#9467bd", linestyle="--", marker="D", linewidth=2.0),
}

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT = Path(__file__).resolve().parents[2] / "paper draft"


def make_fig(data, ylim, outname, legend_loc="lower right"):
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    for name, ys in data.items():
        ax.plot(K, ys, label=name, markersize=4.5,
                markeredgewidth=0, **STYLE[name])
    ax.set_xlabel("$K$ (number of generations)")
    ax.set_ylabel("Pass@$K$")
    ax.set_xticks(K)
    ax.set_xlim(0.85, 4.15)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc=legend_loc, handlelength=2.0,
              borderpad=0.3, labelspacing=0.25)
    plt.tight_layout(pad=0.2)
    plt.savefig(OUT / outname, bbox_inches="tight", pad_inches=0.02)
    print(f"saved: {OUT / outname}")
    plt.close(fig)


make_fig(CHARTQA, (0.5, 0.85), "fig_passk.pdf")
make_fig(CHARTFC, (0.55, 0.94), "fig_passk_chartfc.pdf")
