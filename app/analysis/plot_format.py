"""Overall format-compliance per method, ID vs OOD.

Single figure: x = method (5 ticks), y = overall format compliance (%),
with one line for ChartQA (ID) and one for ChartFC (OOD). The drop
between the two lines at each method shows how that method generalizes
on the format dimension.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1] / "final_outputs"
OUT  = Path(__file__).resolve().parents[2] / "paper draft"


def fully_compliant_pct(path):
    n = 0; ok = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            fc = r.get("format_compliance", {}) or {}
            n += 1
            if fc.get("fully_compliant"):
                ok += 1
    return 100 * ok / n if n else 0.0


METHODS = ["Base", "GRPO", "GRPO+HCPC", "NSR", "NSR+HCPC"]

CHARTQA = {
    "Base":      ROOT / "base_chartqa"         / "per_sample.jsonl",
    "GRPO":      ROOT / "grpo_chartqa"         / "per_sample.jsonl",
    "GRPO+HCPC": ROOT / "hcpc_chartqa"         / "per_sample.jsonl",
    "NSR":       ROOT / "nsr_baseline_chartqa" / "per_sample.jsonl",
    "NSR+HCPC":  ROOT / "nsr_hcpc_chartqa"     / "per_sample.jsonl",
}
CHARTFC = {
    # Base not evaluated on ChartFC; insert None so the line skips it.
    "Base":      None,
    "GRPO":      ROOT / "grpo_chartfc"         / "per_sample.jsonl",
    "GRPO+HCPC": ROOT / "hcpc_chartfc"         / "per_sample.jsonl",
    "NSR":       ROOT / "nsr_baseline_chartfc" / "per_sample.jsonl",
    "NSR+HCPC":  ROOT / "nsr_hcpc_chartfc"     / "per_sample.jsonl",
}

qa_y = [fully_compliant_pct(CHARTQA[m]) for m in METHODS]
fc_y = [fully_compliant_pct(CHARTFC[m]) if CHARTFC[m] else None for m in METHODS]

# For plotting, drop the None entry from ChartFC line.
fc_xs = [i for i, y in enumerate(fc_y) if y is not None]
fc_ys = [y for y in fc_y if y is not None]

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

fig, ax = plt.subplots(figsize=(3.3, 2.4))
xs = list(range(len(METHODS)))

ax.plot(xs, qa_y, color="#1f77b4", marker="o", linewidth=1.8,
        markersize=5.5, markeredgewidth=0, label="ChartQA (in-distribution)")
ax.plot(fc_xs, fc_ys, color="#d62728", marker="s", linewidth=1.8,
        markersize=5.5, markeredgewidth=0, label="ChartFC (OOD)")

# Annotate each point with the percentage
for x, y in zip(xs, qa_y):
    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=7, color="#1f77b4")
for x, y in zip(fc_xs, fc_ys):
    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                xytext=(0, -11), ha="center", fontsize=7, color="#d62728")

ax.set_xticks(xs)
ax.set_xticklabels(METHODS, fontsize=8)
ax.set_ylabel("format compliance (%)")
ax.set_ylim(0, 110)
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.legend(loc="lower left", handlelength=2.0,
          borderpad=0.3, labelspacing=0.25)
plt.tight_layout(pad=0.2)
plt.savefig(OUT / "fig_format.pdf", bbox_inches="tight", pad_inches=0.02)
print(f"saved: {OUT / 'fig_format.pdf'}")
print()
print("ChartQA format %:", dict(zip(METHODS, [round(y, 1) for y in qa_y])))
print("ChartFC format %:", {m: round(y, 1) for m, y in zip(METHODS, fc_y) if y is not None})
