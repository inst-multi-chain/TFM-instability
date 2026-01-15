
import argparse
import csv
import datetime as dt
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def _parse_ts(val: str) -> dt.datetime:
    return dt.datetime.fromisoformat(val.replace("Z", "+00:00")).astimezone(dt.timezone.utc)


def _read_eth(path: Path) -> List[Tuple[dt.datetime, float]]:
    rows: List[Tuple[dt.datetime, float]] = []
    per_minute: Dict[dt.datetime, Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ts_raw = row.get("timestamp", "")
            gas_used_raw = row.get("gas_used", "")
            gas_limit_raw = row.get("gas_limit", "")
            if not ts_raw or not gas_used_raw:
                continue
            try:
                ts = _parse_ts(ts_raw)
                gas_used = float(gas_used_raw)
                gas_limit = float(gas_limit_raw) if gas_limit_raw else 60_000_000.0
            except Exception:
                continue
            minute_key = ts.replace(second=0, microsecond=0)
            agg = per_minute.setdefault(minute_key, {"gas_used": 0.0, "gas_limit": 0.0})
            agg["gas_used"] += gas_used
            agg["gas_limit"] += gas_limit
    for ts_min, vals in sorted(per_minute.items()):
        if vals["gas_limit"] <= 0:
            continue
        load = vals["gas_used"] / vals["gas_limit"]
        rows.append((ts_min, load))
    return rows


def _read_osmo(path: Path) -> List[Tuple[dt.datetime, float, float]]:
    rows: List[Tuple[dt.datetime, float, float]] = []
    per_minute: Dict[dt.datetime, Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ts_raw = row.get("timestamp", "")
            g_used_raw = row.get("gas_used", "")
            g_wanted_raw = row.get("gas_wanted", "")
            if not ts_raw or (not g_used_raw and not g_wanted_raw):
                continue
            try:
                ts = _parse_ts(ts_raw)
                g_used = float(g_used_raw or 0)
                g_wanted = float(g_wanted_raw or 0)
                max_gas = float(row.get("max_gas", "") or 300_000_000.0)
            except Exception:
                continue
            minute_key = ts.replace(second=0, microsecond=0)
            agg = per_minute.setdefault(minute_key, {"used": 0.0, "wanted": 0.0, "limit": 0.0})
            agg["used"] += g_used
            agg["wanted"] += g_wanted
            agg["limit"] += max_gas
    for ts_min, vals in sorted(per_minute.items()):
        if vals["limit"] <= 0:
            continue
        load_used = vals["used"] / vals["limit"]
        load_wanted = vals["wanted"] / vals["limit"] if vals["limit"] else 0.0
        rows.append((ts_min, load_used, load_wanted))
    return rows


def _time_limits(ts_list: List[dt.datetime]) -> Tuple[dt.datetime, dt.datetime]:
    if not ts_list:
        now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0, tzinfo=dt.timezone.utc)
        return now, now + dt.timedelta(hours=2)
    start = min(ts_list)
    end = max(ts_list)
    if start == end:
        end = start + dt.timedelta(hours=2)
    pad = max(dt.timedelta(minutes=1), (end - start) * 0.05)
    return start - pad, end + pad


def _plot_eth(rows: Iterable[Tuple[dt.datetime, float]], out_path: str) -> None:
    times, gas_used = zip(*rows) if rows else ([], [])
    x_min, x_max = _time_limits(list(times))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(times, gas_used, color="tab:blue", linewidth=1.4, alpha=0.7, label="load (used)")
    ax.scatter(times, gas_used, color="tab:blue", s=14, alpha=0.7, rasterized=True)
    ax.axhline(0.5, color="tab:red", linestyle="--", linewidth=2.2, label="target 0.5")
    ax.set_xlim(mdates.date2num(x_min), mdates.date2num(x_max))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(""))
    ax.set_xticks([])
    ax.set_xlabel("Time", fontsize=22)
    ax.set_ylabel("Load", fontsize=22)
    ax.set_title("Ethereum load (per-minute)", fontsize=24)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(loc="upper right", fontsize=22)
    plt.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.12)
    os.makedirs("plots", exist_ok=True)
    plt.savefig(out_path, format="pdf", dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_osmo(rows: Iterable[Tuple[dt.datetime, float, float]], out_path: str) -> None:
    if rows:
        times, g_used, g_wanted = zip(*rows)
    else:
        times, g_used, g_wanted = [], [], []
    x_min, x_max = _time_limits(list(times))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(times, g_used, color="tab:blue", linewidth=1.4, alpha=0.7, label="load (used)")
    ax.scatter(times, g_used, color="tab:blue", s=14, alpha=0.7, rasterized=True)
    ax.plot(times, g_wanted, color="tab:orange", linewidth=1.4, alpha=0.6, label="load (wanted)")
    ax.scatter(times, g_wanted, color="tab:orange", s=14, alpha=0.6, rasterized=True)
    ax.axhline(0.25, color="tab:red", linestyle="--", linewidth=2.2, label="target 0.25")
    ax.set_xlim(mdates.date2num(x_min), mdates.date2num(x_max))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(""))
    ax.set_xticks([])
    ax.set_xlabel("Time", fontsize=22)
    ax.set_ylabel("Load", fontsize=22)
    ax.set_title("Osmosis load (per-minute)", fontsize=24)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(loc="upper right", fontsize=22)
    plt.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.12)
    os.makedirs("plots", exist_ok=True)
    plt.savefig(out_path, format="pdf", dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def _default_eth() -> Path:
    candidates = [
        Path("load/data/Eth_hourly_2025/eth_load_2025-12-23_13-15.csv"),
        Path("data/eth_load_2025-12-23.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit("No default ETH CSV found; pass --eth-csv.")


def _default_osmo() -> Path:
    candidates = [
        Path("load/data/Osmosis_hourly_2025/osmosis_blocks_2025-10-10_2100_2300.csv"),
        Path("load/data/Osmosis_hourly_2025/osmosis_blocks_2025-10-10_2100_2200.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit("No default Osmosis CSV found; pass --osmo-csv.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot hourly gas usage for ETH and Osmosis.")
    parser.add_argument("--eth-csv", type=Path, default=None, help="ETH hourly CSV (timestamp, gas_used columns).")
    parser.add_argument(
        "--osmo-csv", type=Path, default=None, help="Osmosis hourly CSV (timestamp, gas_used, gas_wanted columns)."
    )
    args = parser.parse_args()

    eth_csv = args.eth_csv or _default_eth()
    osmo_csv = args.osmo_csv or _default_osmo()

    eth_rows = _read_eth(eth_csv)
    osmo_rows = _read_osmo(osmo_csv)

    if not eth_rows:
        print(f"[warn] no ETH rows parsed from {eth_csv}")
    if not osmo_rows:
        print(f"[warn] no Osmosis rows parsed from {osmo_csv}")

    _plot_eth(eth_rows, "plots/gas-eth.pdf")
    _plot_osmo(osmo_rows, "plots/gas-osmosis.pdf")


if __name__ == "__main__":
    main()
