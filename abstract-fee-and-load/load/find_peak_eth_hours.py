
import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import List, Optional, Tuple


def _parse_ts(val: str) -> Optional[dt.datetime]:
    if not val:
        return None
    try:
        # Ensure UTC; ISO strings from exporter already have offset
        ts = dt.datetime.fromisoformat(val.replace("Z", "+00:00"))
        return ts.astimezone(dt.timezone.utc)
    except Exception:
        return None


def _load_hours(csv_path: Path, target_date: Optional[dt.date]) -> Tuple[List[int], dt.date]:
    hours = [0] * 24
    detected_date: Optional[dt.date] = target_date
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ts = _parse_ts(row.get("timestamp", ""))
            if ts is None:
                continue
            d = ts.date()
            if detected_date is None:
                detected_date = d
            if d != detected_date:
                continue
            try:
                gas_used = int(row.get("gas_used", "") or 0)
            except Exception:
                continue
            h = ts.hour
            if 0 <= h < 24:
                hours[h] += gas_used
    if detected_date is None:
        raise SystemExit("No timestamps found in CSV.")
    return hours, detected_date


def _find_peak_pair(hours: List[int]) -> Tuple[int, int]:
    best_idx = 0
    best_total = -1
    for i in range(23):
        total = hours[i] + hours[i + 1]
        if total > best_total:
            best_total = total
            best_idx = i
    return best_idx, best_total


def main() -> None:
    parser = argparse.ArgumentParser(description="Find most active consecutive 2-hour window by gas_used.")
    parser.add_argument("--csv", required=True, help="Path to eth_load_<date>.csv")
    parser.add_argument(
        "--date",
        help="Target date (UTC) YYYY-MM-DD. If omitted, inferred from first timestamp.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    target_date = dt.date.fromisoformat(args.date) if args.date else None
    hours, detected_date = _load_hours(csv_path, target_date)

    start_hour, total = _find_peak_pair(hours)
    print(f"date={detected_date} peak_window={start_hour:02d}:00-{start_hour+2:02d}:00 total_gas_used={total}")
    print("hourly_gas_used=" + ",".join(str(v) for v in hours))


if __name__ == "__main__":
    main()
