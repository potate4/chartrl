"""Generate Pass@K curve figure for the paper."""
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ChartQA Pass@K (unbiased estimator), from analysis/paper_stats_output.txt
DATA = {
    "Base":        [0.5245, 0.6733, 0.7375, 0.7740],
    "GRPO":        [0.6490, 0.7490, 0.7905, 0.8160],
    "GRPO+HCPC":   [0.6285, 0.7457, 0.7990, 0.8280],
    "NSR":         [0.5765, 0.7170, 0.7760, 0.8080],
    "NSR+HCPC":    [0.5755, 0.7083, 0.7660, 0.8000],
}
K = [1, 2, 3, 4]

STYLE = {
    "Base":      dict(color="#888888", linestyle=":",  marker="o", linewidth=1.4),
    "GRPO":      dict(color="#1f77b4", linestyle="-",  marker="s", linewidth=1.6),
    "GRPO+HCPC": dict(color="#d62728", linestyle="-",  marker="o", linewidth=2.0),
    "NSR":       dict(color="#2ca02c", linestyle="--", marker="^", linewidth=1.4),
    "NSR+HCPC":  dict(color="#9467bd", linestyle="--", marker="D", linewidth=1.4),
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

fig, ax = plt.subplots(figsize=(3.2, 2.3))

for name, ys in DATA.items():
    ax.plot(K, ys, label=name, markersize=4.5, markeredgewidth=0, **STYLE[name])

ax.set_xlabel("$K$ (number of generations)")
ax.set_ylabel("Pass@$K$")
ax.set_xticks(K)
ax.set_xlim(0.85, 4.15)
ax.set_ylim(0.5, 0.85)
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.legend(loc="lower right", handlelength=2.0, borderpad=0.3, labelspacing=0.25)

plt.tight_layout(pad=0.2)

# Save PDF to paper draft folder
out = Path(__file__).resolve().parents[2] / "paper draft" / "fig_passk.pdf"
plt.savefig(out, bbox_inches="tight", pad_inches=0.02)
print(f"saved: {out}")
