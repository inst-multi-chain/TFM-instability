#!/usr/bin/env python3
# Plot Ri vs Gi with native 'o' / 'x' markers (unchanged look),
# log-scaled Gi (ticks labeled as 10^k), and theoretical boundary.
# Only the scatter points are rasterized (high DPI) to keep PDF small.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ======== paths ========
CSV_PATH = "nn_phase_map_validation_Spike.csv"         # <-- replace with your CSV
OUT_PDF  = "nn_phase_map_prediction.pdf"
RI_PERCENTILE = 0.995
RI_MAX = 5.0

# ======== style: larger fonts, no title ========
plt.rcParams.update({
    "font.size": 28,
    "axes.labelsize": 22,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 19,
    "pdf.compression": 9,
})


# ======== load data ========
df = pd.read_csv(CSV_PATH)

for col in ["Gi", "Ri"]:
    if col not in df.columns:
        raise ValueError(f"CSV must contain column '{col}'")

df["Gi"] = pd.to_numeric(df["Gi"], errors="coerce")
df["Ri"] = pd.to_numeric(df["Ri"], errors="coerce")

# Empirical convergence: prefer boolean column 'converged', fallback to 'final_load'
if "converged" in df.columns:
    def to_bool(x):
        if isinstance(x, bool): return x
        if isinstance(x, (int, np.integer)): return x != 0
        if isinstance(x, str): return x.strip().lower() in {"true","1","yes","y","t"}
        return False
    df["empirical"] = df["converged"].apply(to_bool)
elif "final_load" in df.columns:
    df["final_load"] = pd.to_numeric(df["final_load"], errors="coerce")
    df["empirical"] = (df["final_load"] - 0.5).abs() < 1e-5
else:
    raise ValueError("Need 'converged' or 'final_load' in CSV to derive empirical convergence")

# Theoretical stability region: Ri < 1 and Gi*(1+Ri) < 2
df["theory"] = (df["Ri"] < 1) & ((df["Gi"] * (1 + df["Ri"])) < 2)

# Consistency classification
df = df[np.isfinite(df["Gi"]) & np.isfinite(df["Ri"]) & (df["Gi"] > 0)].copy()
df["consistent"] = (df["empirical"] == df["theory"])

# Separate inconsistent cases into False Positive and False Negative
greens = df[df["consistent"]]    # consistent → green circle
false_positive = df[~df["consistent"] & df["theory"] & ~df["empirical"]]  # predicted stable but actually unstable → red cross
false_negative = df[~df["consistent"] & ~df["theory"] & df["empirical"]]  # predicted unstable but actually stable → orange triangle
n_green = len(greens)
n_fp = len(false_positive)
n_fn = len(false_negative)

# ======== plot ========
fig, ax = plt.subplots(figsize=(7, 5.4))

# Gi on log scale, ticks shown as 10^k
ax.set_yscale('log')
ax.yaxis.set_major_locator(mtick.LogLocator(base=10.0))
ax.yaxis.set_major_formatter(mtick.LogFormatterMathtext(base=10.0))

ri_vals = df["Ri"].to_numpy()
finite_ri = ri_vals[np.isfinite(ri_vals)]
x_min = float(np.min(finite_ri))
x_max = float(np.max(finite_ri))
x_plot_max = x_max
if RI_MAX is not None:
    x_plot_max = min(x_plot_max, RI_MAX)
elif 0.0 < RI_PERCENTILE < 1.0:
    perc = float(np.quantile(finite_ri, RI_PERCENTILE))
    if perc < x_plot_max:
        x_plot_max = perc
n_clipped = int(np.sum(finite_ri > x_plot_max))

x_pad = 0.02 * (x_plot_max - x_min if x_plot_max > x_min else 1.0)
x_left = max(0.0, x_min - x_pad)
x_right = x_plot_max + x_pad
ax.set_xlim(x_left, x_right)

# --- scatter with native markers; rasterize only these collections ---
# Use high savefig dpi so rasterized points are crisp when zooming
scatter_kwargs = dict(s=20, alpha=0.9, rasterized=True)

if n_green > 0:
    ax.scatter(greens["Ri"], greens["Gi"], marker='o',
               c="#2ca02c", edgecolors='none', **scatter_kwargs, label=None)
if n_fp > 0:
    ax.scatter(false_positive["Ri"], false_positive["Gi"], marker='x',
               c="#d62728", linewidths=1.2, **scatter_kwargs, label=None)
if n_fn > 0:
    ax.scatter(false_negative["Ri"], false_negative["Gi"], marker='^',
               c="#ff7f0e", edgecolors='none', s=30, alpha=0.9, rasterized=True, label=None)

# --- theoretical boundary ---
ax.axvline(1.0, color="gray", lw=2, ls="--", alpha=0.6)

# Labels (English), legend upper-left with counts
ax.set_xlabel("Ri")
ax.set_ylabel("Gi")

xticks = [tick for tick in ax.get_xticks() if tick >= x_left - 1e-6]
if not any(abs(t - 1.0) < 1e-6 for t in xticks):
    xticks.append(1.0)
    xticks = sorted(xticks)
ax.set_xticks(xticks)

xlim = ax.get_xlim()
ri = np.linspace(max(1e-6, xlim[0]), xlim[1], 1000)
gi = 2.0 / (1.0 + ri)
gi[gi <= 0] = np.nan
ax.plot(ri, gi, color="blue", lw=2, label="Gi = 2/(1+Ri)")

from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0],[0], marker='o', color='none', markerfacecolor="#2ca02c",
           markeredgecolor='none', markersize=10, label=f"Consistent ({n_green})"),
    Line2D([0],[0], marker='x', color="#d62728", markersize=10,
           markeredgewidth=1.2, label=f"False Positive ({n_fp})"),
    Line2D([0],[0], marker='^', color='none', markerfacecolor="#ff7f0e",
           markeredgecolor='none', markersize=10, label=f"False Negative ({n_fn})"),
    Line2D([0],[0], color="blue", lw=2, label="Gi = 2/(1+Ri)"),
]
ax.legend(handles=legend_handles, loc="upper right", frameon=False)

if n_clipped > 0:
    ax.text(
        0.98,
        0.05,
        f"+{n_clipped} pts beyond shown Ri",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=18,
        color="#555555",
    )

ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_PDF, dpi=600)   # high DPI → crisp rasterized points
print(f"Saved: {OUT_PDF}  (consistent: {n_green}, false positive: {n_fp}, false negative: {n_fn})")
