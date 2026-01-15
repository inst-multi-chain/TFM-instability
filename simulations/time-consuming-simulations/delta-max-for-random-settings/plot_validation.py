#!/usr/bin/env python3
"""
Plot Delta Max Validation

Scatter plot with x = empirical δ_max, y = calculated δ_max, plus y=x reference.
Only the scatter points are rasterized to keep PDF size small.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional

# ======== plotting defaults ========
plt.rcParams.update({
    "font.size": 22,
    "axes.labelsize": 22,
    "axes.titlesize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "pdf.compression": 9,
})

# ======== column guesses ========
EMP_CANDIDATES = [
    "empirical_delta_max",
    "emp_delta_max",
    "delta_emp",
    "emp_dmax",
    "empirical_dmax",
    "empirical",
    "delta_max_empirical",
]

CALC_CANDIDATES = [
    "calculated_delta_max",
    "calc_delta_max",
    "delta_calc",
    "calc_dmax",
    "theory_delta_max",
    "predicted_delta_max",
    "calculated",
    "model_delta_max",
]


def pick_col(df: pd.DataFrame, preferred: Optional[str], candidates: List[str], fallback_idx: int) -> str:
    """Pick column matching preferred/candidates or fallback to numeric-like column."""
    if preferred and preferred in df.columns:
        return preferred
    for cand in candidates:
        if cand in df.columns:
            return cand
    numeric_like = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if np.isfinite(series).sum() > 0:
            numeric_like.append(col)
    if len(numeric_like) > fallback_idx:
        return numeric_like[fallback_idx]
    raise ValueError(
        "Unable to pick column; please specify --emp-col/--calc-col explicitly."
    )


def plot_validation(
    csv_path: str = "delta_max_validation.csv",
    output_path: str = "delta_max_validation.pdf",
    emp_col: Optional[str] = None,
    calc_col: Optional[str] = None,
) -> None:
    """Load CSV and generate empirical vs calculated δ_max scatter plot."""
    df = pd.read_csv(csv_path)

    emp = pick_col(df, emp_col, EMP_CANDIDATES, 0)
    calc = pick_col(df, calc_col, CALC_CANDIDATES, 1)

    x = pd.to_numeric(df[emp], errors="coerce").to_numpy()
    y = pd.to_numeric(df[calc], errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size == 0:
        raise ValueError("No finite data points to plot.")

    fig, ax = plt.subplots(figsize=(7, 5.4))

    vmin = float(np.min([x.min(), y.min()]))
    vmax = float(np.max([x.max(), y.max()]))
    pad = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
    lo, hi = vmin - pad, vmax + pad
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.scatter(x, y, s=20, alpha=0.9, color="tab:blue", rasterized=True)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=2, color="gray", alpha=0.8)
    ax.text(
        0.8,
        0.8,
        "y = x",
        transform=ax.transAxes,
        rotation=45,
        rotation_mode="anchor",
        color="black",
        ha="center",
        va="center",
        fontsize=20,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, boxstyle="round,pad=0.2"),
    )

    ax.set_xlabel(r"Empirical $\delta_{\max}$")
    ax.set_ylabel(r"Calculated $\delta_{\max}$")

    denom = np.where(np.abs(x) > 1e-12, np.abs(x), np.nan)
    avg_gap_pct = float(np.nanmean(np.abs(y - x) / denom) * 100.0)
    # ax.text(
    #     0.5,
    #     1.02,
    #     f"Average gap = {avg_gap_pct:.1f}%",
    #     transform=ax.transAxes,
    #     ha="center",
    #     va="bottom",
    #     fontsize=20,
    # )

    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600)
    print(f"Saved: {output_path}  (x='{emp}', y='{calc}', n={x.size})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Delta Max Validation")
    parser.add_argument("--input", type=str, default="delta_max_validation.csv", help="Input CSV path")
    parser.add_argument("--output", type=str, default="delta_max_validation.pdf", help="Output PDF path")
    parser.add_argument("--emp-col", type=str, default=None, help="Empirical column name override")
    parser.add_argument("--calc-col", type=str, default=None, help="Calculated column name override")
    args = parser.parse_args()

    plot_validation(args.input, args.output, args.emp_col, args.calc_col)


if __name__ == "__main__":
    main()
