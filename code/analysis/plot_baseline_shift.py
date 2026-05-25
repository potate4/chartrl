"""Plot training-time group mean reward to verify the baseline-shift mechanism.

Uses TRL's trainer_state.json log_history, which captures the per-step `reward`
field that TRL actually uses to compute GRPO advantages (mean of R across the
K rollouts in the group). This is the right quantity for the baseline-shift
mechanism claim.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

NSR_STATE  = Path(r"C:\Users\sumai\DATA\RESEARCH\THESIS\chartrl\app\outputs\model_nsr_baselin\outputs\nsr_baseline\run_20260515_115514\trl_output\checkpoint-2000\trainer_state.json")
HCPC_STATE = Path(r"C:\Users\sumai\DATA\RESEARCH\THESIS\chartrl\app\outputs\model_nsr_hcpc\outputs\nsr_hcpc\run_20260515_114827\trl_output\checkpoint-2000\trainer_state.json")


def load_history(path):
    with open(path) as f:
        s = json.load(f)
    return s["log_history"]


nsr_hist = load_history(NSR_STATE)
nh_hist  = load_history(HCPC_STATE)

nsr_steps = [r["step"] for r in nsr_hist]
nsr_reward = [r["reward"] for r in nsr_hist]
nsr_std = [r["reward_std"] for r in nsr_hist]

nh_steps  = [r["step"] for r in nh_hist]
nh_reward = [r["reward"] for r in nh_hist]
nh_std = [r["reward_std"] for r in nh_hist]


def smooth(xs, window=5):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window // 2)
        hi = min(len(xs), i + window // 2 + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


# Compute step-by-step gap
common = sorted(set(nsr_steps) & set(nh_steps))
nsr_at = {r["step"]: r["reward"] for r in nsr_hist}
nh_at  = {r["step"]: r["reward"] for r in nh_hist}
nsr_std_at = {r["step"]: r["reward_std"] for r in nsr_hist}
nh_std_at  = {r["step"]: r["reward_std"] for r in nh_hist}
gaps = [nh_at[s] - nsr_at[s] for s in common]
std_gaps = [nh_std_at[s] - nsr_std_at[s] for s in common]
print(f"TRL-logged per-step reward gap (NSR+HCPC minus NSR):")
print(f"  mean gap: {np.mean(gaps):+.4f}")
print(f"  median gap: {np.median(gaps):+.4f}")
print(f"  fraction of steps where NSR+HCPC > NSR: {sum(1 for g in gaps if g > 0)/len(gaps):.1%}")
print()
print(f"TRL-logged per-step reward_std gap (NSR+HCPC minus NSR):")
print(f"  mean: {np.mean(std_gaps):+.4f}")
print(f"  median: {np.median(std_gaps):+.4f}")
print()
print(f"Late-training average (steps >= 500):")
late_gap = [nh_at[s] - nsr_at[s] for s in common if s >= 500]
late_std_gap = [nh_std_at[s] - nsr_std_at[s] for s in common if s >= 500]
print(f"  mean reward gap: {np.mean(late_gap):+.4f}")
print(f"  mean std gap: {np.mean(late_std_gap):+.4f}")
print(f"  fraction positive: {sum(1 for g in late_gap if g > 0)/len(late_gap):.1%}")


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

fig, ax = plt.subplots(figsize=(3.3, 2.3))

ax.plot(nsr_steps, smooth(nsr_reward), color="#2ca02c", linestyle="--",
        linewidth=1.5, label=r"NSR: $\bar R$")
ax.plot(nh_steps,  smooth(nh_reward),  color="#9467bd", linestyle="-",
        linewidth=2.0, label=r"NSR+HCPC: $\bar R$")

ax.set_xlabel("training step")
ax.set_ylabel(r"group mean reward $\bar R$")
ax.set_xlim(0, max(max(nsr_steps), max(nh_steps)))
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.legend(loc="lower right", handlelength=2.0,
          borderpad=0.3, labelspacing=0.25)
plt.tight_layout(pad=0.2)

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT / "fig_baseline_shift.pdf",
            bbox_inches="tight", pad_inches=0.02)
print(f"saved figure to {OUT}")
