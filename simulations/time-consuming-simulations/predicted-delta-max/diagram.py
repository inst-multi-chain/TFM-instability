#!/usr/bin/env python3
"""
Compare empirical δ_max vs NN-predicted δ_max for each configuration.

Visual language mirrors the critical-delta safety chart:
    - Solid bar = predicted δ_max.
    - Hatched extension = empirical gap above the prediction.
    - If the NN overshoots empirical, the extension flips color/hatch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.size": 28,
    "axes.labelsize": 42,
    "xtick.labelsize": 36,
    "ytick.labelsize": 36,
    "legend.fontsize": 32,
    "pdf.compression": 9,
})


@dataclass
class DeltaPair:
    config: str
    empirical: float
    predicted: float


def load_empirical(path: str) -> pd.Series:
    df = pd.read_csv(path)
    required = {"config_name", "critical_delta"}
    if not required.issubset(df.columns):
        raise ValueError(f"Empirical CSV must contain {required}.")
    series = df.set_index("config_name")["critical_delta"].astype(float)
    if series.empty:
        raise ValueError("Empirical CSV has no data.")
    return series


def load_predicted(path: str) -> pd.Series:
    df = pd.read_csv(path)
    required = {"config", "delta", "stable"}
    if not required.issubset(df.columns):
        raise ValueError(f"Predicted CSV must contain {required}.")
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
    df = df[np.isfinite(df["delta"])]
    if df.empty:
        raise ValueError("Predicted CSV has no usable rows.")

    summary = {}
    for config, group in df.groupby("config"):
        group = group.sort_values("delta")
        first_unstable = group[group["stable"] == False]
        if not first_unstable.empty:
            summary[config] = float(first_unstable["delta"].iloc[0])
        else:
            summary[config] = float(group["delta"].max())
    return pd.Series(summary)


def build_pairs(empirical: pd.Series, predicted: pd.Series) -> list[DeltaPair]:
    common_set = set(empirical.index) & set(predicted.index)
    if not common_set:
        raise ValueError("No overlapping configurations between the two CSVs.")

    # Preferred ordering: Polkadot first, then Cosmos together
    preferred_order = [
        "Asset Hub",
        "Bifrost",
        "Hydration",
        "Moonbeam",
        "Cosmos Hub",
        "Osmosis",
    ]
    ordered = [name for name in preferred_order if name in common_set]
    # Append any remaining configs (unlikely) in sorted order
    ordered.extend(sorted(common_set - set(ordered)))

    return [
        DeltaPair(name, float(empirical[name]), float(predicted[name]))
        for name in ordered
    ]


def plot(pairs: list[DeltaPair], output: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 10.8))
    x = np.arange(len(pairs))
    width = 0.75

    predicted_color = "#1F6EBA"
    surplus_color = "#7DAF70"

    max_val = max(max(p.empirical, p.predicted) for p in pairs)
    label_pad = max_val * 0.015

    for idx, pair in enumerate(pairs):
        pred = pair.predicted
        emp = pair.empirical

        ax.bar(
            idx,
            pred,
            width,
            color=predicted_color,
            edgecolor="black",
            linewidth=1.6,
            label="_nolegend_",
        )

        ax.bar(
            idx,
            emp - pred,
            width,
            bottom=pred,
            color=surplus_color,
            edgecolor="darkgreen",
            linewidth=1.6,
            hatch="///",
            alpha=0.85,
            label="_nolegend_",
        )

        ax.text(
            idx,
            pred + label_pad,
            f"{pred:.3f}",
            ha="center",
            va="bottom",
            fontsize=30,
            color="#0f3a6d",
            fontweight="bold",
        )
        ax.text(
            idx,
            emp + label_pad,
            f"{emp:.3f}",
            ha="center",
            va="bottom",
            fontsize=30,
            color="#265624",
            fontweight="bold",
        )

    ax.set_ylabel(r"$\delta_{max}$")
    ax.set_xticks(x)
    ax.set_xticklabels([p.config for p in pairs], rotation=20, ha="right")
    ax.set_xlim(-0.5, len(pairs) - 0.5)
    ax.set_ylim(0, max_val * 1.1)
    ax.grid(axis="y", alpha=0.25)

    legend_handles = [
        Patch(facecolor=predicted_color, edgecolor="black", linewidth=1.6, label="Predicted $\\delta_{max}$"),
        Patch(facecolor=surplus_color, edgecolor="darkgreen", linewidth=1.6, hatch="///", label="Empirical surplus"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False, ncol=3)

    fig.tight_layout()
    fig.savefig(output, dpi=600)
    print(f"Saved: {output}  ({len(pairs)} configurations)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot empirical vs predicted delta_max")
    parser.add_argument(
        "--empirical",
        default="../critical-delta-experiment/critical_delta_results.csv",
        help="Path to critical delta experiment results CSV",
    )
    parser.add_argument(
        "--predicted",
        default="nn_predicted_critical_deltas.csv",
        help="Path to NN predicted delta results CSV",
    )
    parser.add_argument(
        "--output",
        default="delta_max_empirical_vs_predicted.pdf",
        help="Output PDF filename",
    )
    args = parser.parse_args()

    empirical = load_empirical(args.empirical)
    predicted = load_predicted(args.predicted)
    pairs = build_pairs(empirical, predicted)
    plot(pairs, args.output)


if __name__ == "__main__":
    main()
