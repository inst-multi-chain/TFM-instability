

import csv
import datetime as dt
import os
import re
import sys
import types
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import base64
import decimal

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Minimal pandas shim to satisfy range_load_extract imports (pandas not available here).
def _parse_iso_ts(val: str) -> dt.datetime:
    val = val.replace("Z", "+00:00")
    m = re.match(r"(.+?)([+-]\d\d:\d\d)$", val)
    if not m:
        return dt.datetime.fromisoformat(val)
    main, tz = m.groups()
    if "." in main:
        base, frac = main.split(".", 1)
        frac = (frac + "000000")[:6]
        main = f"{base}.{frac}"
    return dt.datetime.fromisoformat(f"{main}{tz}")


def _to_datetime(value, utc: bool = False, unit: str = None):
    if isinstance(value, dt.datetime):
        ts = value
    elif isinstance(value, (int, float)) and unit == "s":
        ts = dt.datetime.fromtimestamp(value, tz=dt.timezone.utc)
    elif isinstance(value, str):
        ts = _parse_iso_ts(value)
    else:
        raise TypeError(f"unsupported to_datetime value: {value}")
    return ts.astimezone(dt.timezone.utc) if utc and ts.tzinfo else ts


class _Series(list):
    pass


class _DataFrame(list):
    def __init__(self, *args, **kwargs):
        super().__init__()


pd_stub = types.SimpleNamespace(
    to_datetime=_to_datetime,
    Timestamp=dt.datetime,
    Series=_Series,
    DataFrame=_DataFrame,
)
sys.modules.setdefault("pandas", pd_stub)

from load import range_load_extract as r

VERBOSE = os.environ.get("VERBOSE", "1") not in ("", "0")


BLOCK_TIME_HINT = float(os.environ.get("BLOCK_TIME_HINT", "1.0"))
BATCH_FLUSH = 10
BASE_DENOM = os.environ.get("BASE_DENOM", "uosmo")

OSMO_RPCS = [
    "https://osmosis-rpc.publicnode.com:443",
    "https://osmosis-archive-rpc.polkachu.com",
    "https://rpc-osmosis-ia.cosmosia.notional.ventures",
    "https://osmosis-archive-rpc.bccnodes.com",
    "https://rpc-osmosis-ia.polkachu.com",
    "https://rpc.osmosis.zone",
    "https://rpc.cosmos.directory/osmosis",
    "https://osmosis-rpc.polkachu.com",
]

WINDOWS: List[Tuple[str, str]] = [
    ("2025-10-10 21:00:00", "2025-10-10 22:00:00"),
    ("2025-10-10 22:00:00", "2025-10-10 23:00:00"),
]

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "hourly_2025"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _parse_hour(ts: str) -> dt.datetime:
    return dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)


def _vlog(msg: str) -> None:
    if VERBOSE:
        print(msg, file=sys.stderr, flush=True)


def _decode_attr(val: str) -> str:
    if val is None:
        return ""
    try:
        return base64.b64decode(val).decode("utf-8")
    except Exception:
        return val


def _parse_fee_value(val: str, denom: str) -> int:
    """Parse coin string like '123uosmo,10uatom' and return amount for denom as int."""
    if not val:
        return 0
    total = 0
    parts = val.split(",")
    for p in parts:
        p = p.strip()
        if not p:
            continue
        num = ""
        i = 0
        while i < len(p) and p[i].isdigit():
            num += p[i]
            i += 1
        denom_part = p[i:]
        if denom_part != denom or not num:
            continue
        try:
            total += int(num)
        except ValueError:
            continue
    return total


def _locate_window_heights(
    session,
    rpc: str,
    latest_height: int,
    latest_ts: int,
    ts_cache: Dict,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
) -> Tuple[int, int]:
    start_ts = r._epoch(start_dt)
    end_ts = r._epoch(end_dt) - 1
    bt_hint = BLOCK_TIME_HINT
    low_start, high_start = r._height_bounds(latest_height, latest_ts, start_ts, bt_hint)
    low_end, high_end = r._height_bounds(latest_height, latest_ts, end_ts, bt_hint)
    ts_fn = r._cosmos_block_ts
    _vlog(f"[locate] window {start_dt}->{end_dt} start_ts={start_ts} end_ts={end_ts} avg_bt={bt_hint}")
    _vlog(f"[locate] start bounds={low_start}->{high_start} end bounds={low_end}->{high_end}")

    def find_ge(target_ts, low, high):
        lo = max(0, low)
        hi = min(latest_height, high)
        while lo < hi:
            mid = (lo + hi) // 2
            ts = ts_fn(session, rpc, mid, ts_cache)
            _vlog(f"[locate] GE mid={mid} ts={ts} target={target_ts} lo={lo} hi={hi}")
            if ts < target_ts:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def find_le(target_ts, low, high):
        lo = max(0, low)
        hi = min(latest_height, high)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            ts = ts_fn(session, rpc, mid, ts_cache)
            _vlog(f"[locate] LE mid={mid} ts={ts} target={target_ts} lo={lo} hi={hi}")
            if ts <= target_ts:
                lo = mid
            else:
                hi = mid - 1
        return lo

    start_h = find_ge(start_ts, low_start, high_start)
    start_ts_found = ts_fn(session, rpc, start_h, ts_cache)
    _vlog(f"[locate] start_height={start_h} ts={start_ts_found}")
    end_h = find_le(end_ts, low_end, high_end)
    end_ts_found = ts_fn(session, rpc, end_h, ts_cache)
    _vlog(f"[locate] end_height={end_h} ts={end_ts_found}")
    return start_h, end_h


def _fetch_window(
    session,
    rpc: str,
    latest_height: int,
    latest_ts: int,
    max_gas_cache: Dict[str, int],
    ts_cache: Dict,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
) -> Tuple[int, int, Iterable[Dict]]:
    start_h, end_h = _locate_window_heights(session, rpc, latest_height, latest_ts, ts_cache, start_dt, end_dt)
    if start_h > end_h:
        return start_h, end_h, []

    def _fetch_block(height: int) -> Dict:
        ts, txc = r._cosmos_block(session, rpc, height, ts_cache)
        res = r._cosmos_get(session, rpc, f"/block_results?height={height}")
        result = res["result"]
        gas_used = 0
        gas_wanted = 0
        total_fee_base = 0
        for txr in result.get("txs_results") or []:
            try:
                gas_used += int(txr.get("gas_used", "0"))
            except Exception:
                pass
            try:
                gas_wanted += int(txr.get("gas_wanted", "0"))
            except Exception:
                pass
            for ev in txr.get("events") or []:
                for attr in ev.get("attributes") or []:
                    key = _decode_attr(attr.get("key"))
                    if key != "fee":
                        continue
                    val = _decode_attr(attr.get("value"))
                    total_fee_base += _parse_fee_value(val, BASE_DENOM)
        updates = result.get("consensus_param_updates", {})
        new_max_gas = None
        if updates and "block" in updates and updates["block"] and "max_gas" in updates["block"]:
            try:
                val = int(updates["block"]["max_gas"])
                new_max_gas = val if val > 0 else None
            except Exception:
                new_max_gas = None
        if new_max_gas is not None:
            max_gas_cache["value"] = new_max_gas
        max_gas = max_gas_cache.get("value")
        load = gas_used / max_gas if max_gas else float("nan")
        fee_per_gas = (decimal.Decimal(total_fee_base) / decimal.Decimal(10**6) / gas_used) if gas_used else decimal.Decimal(0)

        return {
            "height": height,
            "timestamp": ts if isinstance(ts, dt.datetime) else _to_datetime(ts),
            "gas_used": gas_used,
            "gas_wanted": gas_wanted,
            "tx_count": txc,
            "total_fee_base": total_fee_base,
            "total_fee_osmo": float(decimal.Decimal(total_fee_base) / decimal.Decimal(10**6)),
            "fee_per_gas_osmo": float(fee_per_gas),
            "load": load,
            "max_gas": max_gas if max_gas is not None else "",
            "status": "ok",
            "error": "",
        }

    def gen():
        for h in range(start_h, end_h + 1):
            try:
                row = _fetch_block(h)
                _vlog(f"[progress] h={h} ts={row['timestamp']} gas_used={row['gas_used']}")
                yield row
            except Exception as exc:
                _vlog(f"[warn] h={h} failed: {exc}")
                yield {
                    "height": h,
                    "timestamp": "",
                    "gas_used": "",
                    "gas_wanted": "",
                    "tx_count": "",
                    "total_fee_base": "",
                    "total_fee_osmo": "",
                    "fee_per_gas_osmo": "",
                    "load": "",
                    "max_gas": "",
                    "status": "error",
                    "error": str(exc),
                }

    return start_h, end_h, gen()


def main() -> None:
    for rpc in OSMO_RPCS:
        try:
            session = r._build_session(timeout=10)
            ts_cache: Dict = {}
            latest_height, latest_ts = r._cosmos_latest(session, rpc)
            max_gas_cache: Dict[str, int] = {"value": r._cosmos_consensus_max_gas(session, rpc, latest_height)}
            _vlog(f"[info] rpc={rpc} latest_height={latest_height} latest_ts={latest_ts} bt_hint={BLOCK_TIME_HINT}")

            for start_str, end_str in WINDOWS:
                start_dt = _parse_hour(start_str)
                end_dt = _parse_hour(end_str)
                start_h, end_h, row_iter = _fetch_window(
                    session, rpc, latest_height, latest_ts, max_gas_cache, ts_cache, start_dt, end_dt
                )
                label = f"{start_str}-{end_str}"
                out_path = OUT_DIR / f"osmosis_blocks_{start_dt.date()}_{start_dt:%H00}_{end_dt:%H00}.csv"
                with out_path.open("w", newline="", encoding="utf-8") as fh:
                    fieldnames = [
                        "height",
                        "timestamp",
                        "gas_used",
                        "gas_wanted",
                        "tx_count",
                        "total_fee_base",
                        "total_fee_osmo",
                        "fee_per_gas_osmo",
                        "load",
                        "max_gas",
                        "status",
                        "error",
                    ]
                    writer = csv.DictWriter(fh, fieldnames=fieldnames)
                    writer.writeheader()
                    fh.flush()

                    blocks = 0
                    total_tx = 0
                    total_gas = 0
                    total_gas_wanted = 0
                    total_fee_base = 0
                    for row in row_iter:
                        blocks += 1
                        ts_val = row["timestamp"]
                        row_out = dict(row)
                        if ts_val:
                            row_out["timestamp"] = ts_val.isoformat()
                        writer.writerow(row_out)
                        if blocks % BATCH_FLUSH == 0:
                            fh.flush()
                        if row_out["status"] == "ok":
                            try:
                                total_tx += int(row_out["tx_count"])
                                total_gas += int(row_out["gas_used"])
                                total_gas_wanted += int(row_out["gas_wanted"])
                                total_fee_base += int(row_out["total_fee_base"])
                            except Exception:
                                pass
                        print(
                            f"{row_out['height']},{row_out.get('timestamp','')},{row_out['gas_used']},"
                            f"{row_out['gas_wanted']},{row_out['tx_count']},"
                            f"{row_out.get('total_fee_base','')},{row_out.get('total_fee_osmo','')},"
                            f"{row_out.get('fee_per_gas_osmo','')},"
                            f"{row_out['status']},{row_out['error']}"
                        )
                    fh.flush()

                print(
                    f"=== {label} heights {start_h}->{end_h} blocks={blocks} "
                    f"gas_total={total_gas} gas_wanted_total={total_gas_wanted} "
                    f"tx_total={total_tx} fee_total_base={total_fee_base} ==="
                )
                if blocks == 0:
                    print("  (no blocks found in this window)")
            return
        except Exception as exc:
            print(f"[warn] rpc failed {rpc}: {exc}", file=sys.stderr)
            continue

    sys.exit("All RPCs failed")


if __name__ == "__main__":
    main()
