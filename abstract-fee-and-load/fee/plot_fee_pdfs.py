
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _hours(series: pd.Series) -> pd.Series:
    return (series - series.dt.floor("D")).dt.total_seconds() / 3600.0


def _rolling_trend(series: pd.Series, window: int = 25) -> pd.Series:
    """Centered rolling median to reveal trend while preserving spikes."""
    return series.rolling(window=window, center=True, min_periods=1).median()


def plot_chain(df: pd.DataFrame, title: str, out_path: str, add_trend: bool = False) -> None:
    df = df.copy().sort_values("timestamp")
    df["hour"] = _hours(df["timestamp"])
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    # Top: base fee scatter (all points)
    axes[0].scatter(
        df["hour"],
        df["base_fee_gwei"],
        label="base fee",
        color="tab:blue",
        s=5,
        alpha=0.7,
        rasterized=True,
    )
    if add_trend:
        axes[0].plot(
            df["hour"],
            _rolling_trend(df["base_fee_gwei"]),
            color="black",
            linewidth=2.0,
            alpha=0.6,
            label="_nolegend_",  # keep legend unchanged
        )
    axes[0].set_ylabel("Base fee", fontsize=22)
    axes[0].set_title(title, fontsize=22)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=20)
    axes[0].tick_params(axis="both", labelsize=20)

    eps = 1e-6
    axes[1].scatter(
        df["hour"],
        df["gas_price_gwei"] + eps,
        label="gas price",
        color="tab:red",
        s=5,
        alpha=0.6,
        rasterized=True,
    )
    if add_trend:
        axes[1].plot(
            df["hour"],
            _rolling_trend(df["gas_price_gwei"] + eps),
            color="black",
            linewidth=2.0,
            alpha=0.6,
            label="_nolegend_",
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Hour of day (0–24)", fontsize=22)
    axes[1].set_ylabel("Gas price", fontsize=22)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=20)
    axes[1].tick_params(axis="both", labelsize=20)
    axes[1].set_xlim(0, 24)
    axes[1].set_xticks([0, 6, 12, 18, 24])

    plt.subplots_adjust(left=0.16, right=0.98, top=0.92, bottom=0.12, hspace=0.2)
    os.makedirs("plots", exist_ok=True)
    plt.savefig(out_path, format="pdf", dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    eth = pd.read_csv("data/eth_fee_range_2025-12-23.csv", parse_dates=["timestamp"])
    moon = pd.read_csv("data/moonbeam_fee_range.csv", parse_dates=["timestamp"])

    plot_chain(eth, "Ethereum (24h window)", "plots/fee-eth.pdf", add_trend=True)
    plot_chain(moon, "Moonbeam (24h window)", "plots/fee-moonbeam.pdf", add_trend=False)

if __name__ == "__main__":
    main()
