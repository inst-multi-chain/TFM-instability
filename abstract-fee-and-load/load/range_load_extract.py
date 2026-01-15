
import argparse
import datetime as dt
import json
import math
import base64
import os
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


DEFAULT_ETH_RPC = "your_key"


DEFAULT_OSMOSIS_RPC = "https://osmosis-mainnet-rpc.allthatnode.com:443"
OSMOSIS_RPC_FALLBACKS = [
    DEFAULT_OSMOSIS_RPC,
    # "https://osmosis-archive-rpc.polkachu.com",
    "https://rpc-osmosis-ia.cosmosia.notional.ventures",
    "https://osmosis-archive-rpc.bccnodes.com",
    "https://rpc-osmosis-ia.polkachu.com",
    "https://rpc.osmosis.zone",
    "https://rpc.cosmos.directory/osmosis",
    "https://osmosis-rpc.polkachu.com",
]
OSMOSIS_LCD_FALLBACKS = [
    "https://osmosis-lcd.publicnode.com",
    "https://osmosis-mainnet-lcd.allthatnode.com:443",
    "https://rest.cosmos.directory/osmosis",
    "https://lcd.osmosis.zone",
    "https://osmosis-api.polkachu.com",
]


def _build_session(timeout: int) -> requests.Session:
    sess = requests.Session()
    sess.request_timeout = timeout
    return sess


def _post_json(session: requests.Session, url: str, payload) -> Dict:
    resp = session.post(url, json=payload, timeout=getattr(session, "request_timeout", 30))
    resp.raise_for_status()
    body = resp.json()
    if isinstance(body, dict) and "error" in body:
        raise RuntimeError(body["error"])
    return body


def _get_json(session: requests.Session, url: str, retries: int = 3, backoff: float = 0.6) -> Dict:
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=getattr(session, "request_timeout", 30))
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            time.sleep(backoff)


def _cosmos_get(session: requests.Session, rpc: str, path: str, validate=None) -> Dict:
    """Fetch Cosmos endpoint with simple endpoint failover."""
    base = rpc.rstrip("/")
    candidates: List[str] = []
    for url in [base] + OSMOSIS_RPC_FALLBACKS:
        u = url.rstrip("/")
        if u not in candidates:
            candidates.append(u)
    last_exc = None
    for idx, base_url in enumerate(candidates):
        try:
            data = _get_json(session, f"{base_url}{path}")
            if not data:
                raise ValueError("empty response")
            if isinstance(data, dict) and "error" in data:
                raise RuntimeError(data["error"])
            if validate and not validate(data):
                raise ValueError("invalid response shape")
            return data
        except requests.HTTPError as exc:
            code = getattr(exc.response, "status_code", None)
            retriable = code in {404, 405, 429, 500, 502, 503, 504}
            is_last = idx == len(candidates) - 1
            if retriable and not is_last:
                last_exc = exc
                continue
            raise
        except (ValueError, KeyError, TypeError) as exc:
            last_exc = exc
            continue
        except requests.RequestException as exc:
            last_exc = exc
            continue
    if last_exc:
        raise last_exc
    raise RuntimeError("Unexpected cosmos_get failure")


def _epoch(ts: dt.datetime) -> int:
    return int(ts.replace(tzinfo=dt.timezone.utc).timestamp())


def _is_valid_block_resp(data: Dict) -> bool:
    try:
        blk = data["result"]["block"]
        _ = blk["header"]["time"]
        _ = blk["data"]
        return True
    except Exception:
        return False


def _is_valid_block_resp_lcd(data: Dict) -> bool:
    try:
        blk = data["block"]
        _ = blk["header"]["time"]
        _ = blk["data"]
        return True
    except Exception:
        return False


def _block_time_hint(chain: str) -> float:
    return 12.0 if chain == "eth" else 6.0


def _height_bounds(latest_height: int, latest_ts: int, target_ts: int, avg_block_time: float) -> Tuple[int, int]:
    if latest_ts <= 0 or avg_block_time <= 0:
        return 0, latest_height
    if target_ts >= latest_ts:
        low = max(0, latest_height - 50_000)
        return low, latest_height
    diff_sec = latest_ts - target_ts
    est_back = int(diff_sec / avg_block_time)
    est_height = max(0, latest_height - est_back)
    cushion = max(50_000, int(est_back * 0.25))
    low = max(0, est_height - cushion)
    high = min(latest_height, est_height + cushion)
    if low >= high:
        high = min(latest_height, low + cushion)
    return low, high


# -------- EVM helpers --------
def _evm_block(session: requests.Session, rpc: str, block_number: int, cache: Dict[int, Dict]) -> Dict:
    if block_number in cache:
        return cache[block_number]
    payload = {
        "jsonrpc": "2.0",
        "id": block_number,
        "method": "eth_getBlockByNumber",
        "params": [hex(block_number), False],
    }
    res = _post_json(session, rpc, payload)
    block = res["result"]
    cache[block_number] = block
    return block


def _evm_latest(session: requests.Session, rpc: str) -> Tuple[int, int]:
    payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_getBlockByNumber", "params": ["latest", False]}
    res = _post_json(session, rpc, payload)
    block = res["result"]
    num = int(block["number"], 16)
    ts = int(block["timestamp"], 16)
    return num, ts


def _evm_block_ts(session: requests.Session, rpc: str, block_number: int, cache: Dict[int, Dict]) -> int:
    blk = _evm_block(session, rpc, block_number, cache)
    return int(blk["timestamp"], 16)


def _find_block_ge_ts(
    session: requests.Session,
    rpc: str,
    target_ts: int,
    latest_height: int,
    ts_fn,
    cache: Dict[int, Dict],
    low: Optional[int] = None,
    high: Optional[int] = None,
) -> int:
    lo = 0 if low is None else max(0, low)
    hi = latest_height if high is None else min(latest_height, high)
    while lo < hi:
        mid = (lo + hi) // 2
        ts = ts_fn(session, rpc, mid, cache)
        if ts < target_ts:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _find_block_le_ts(
    session: requests.Session,
    rpc: str,
    target_ts: int,
    latest_height: int,
    ts_fn,
    cache: Dict[int, Dict],
    low: Optional[int] = None,
    high: Optional[int] = None,
) -> int:
    lo = 0 if low is None else max(0, low)
    hi = latest_height if high is None else min(latest_height, high)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        ts = ts_fn(session, rpc, mid, cache)
        if ts <= target_ts:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _evm_block_load(
    session: requests.Session, rpc: str, height: int, cache: Dict[int, Dict]
) -> Tuple[pd.Timestamp, int, int, float, int, float]:
    blk = _evm_block(session, rpc, height, cache)
    ts = int(blk["timestamp"], 16)
    gas_used = int(blk["gasUsed"], 16)
    gas_limit = int(blk["gasLimit"], 16)
    base_fee_gwei = int(blk.get("baseFeePerGas", "0x0"), 16) / 1e9
    load = gas_used / gas_limit if gas_limit > 0 else float("nan")
    tx_count = len(blk.get("transactions", []))
    return pd.to_datetime(ts, unit="s", utc=True), gas_used, gas_limit, load, tx_count, base_fee_gwei


# -------- Cosmos helpers (Tendermint) --------
def _cosmos_block(
    session: requests.Session, rpc: str, height: int, ts_cache: Dict[int, Tuple[pd.Timestamp, int]]
) -> Tuple[pd.Timestamp, int]:
    if height in ts_cache:
        return ts_cache[height]
    try:
        res = _cosmos_get(session, rpc, f"/block?height={height}", _is_valid_block_resp)
        header = res["result"]["block"]["header"]
        block_data = res["result"]["block"]["data"]
    except Exception as rpc_exc:
        last_exc = rpc_exc
        for base in OSMOSIS_LCD_FALLBACKS:
            try:
                res = _get_json(session, f"{base.rstrip('/')}/cosmos/base/tendermint/v1beta1/blocks/{height}")
                if not _is_valid_block_resp_lcd(res):
                    raise ValueError("invalid LCD block response")
                header = res["block"]["header"]
                block_data = res["block"]["data"]
                break
            except Exception as lcd_exc:
                last_exc = lcd_exc
                continue
        else:
            raise RuntimeError(f"Invalid block response for height {height}") from last_exc
    ts = pd.to_datetime(header["time"], utc=True)
    txs = block_data.get("txs") or []
    tx_count = len(txs)
    ts_cache[height] = (ts, tx_count)
    return ts, tx_count


def _cosmos_latest_height(session: requests.Session, rpc: str) -> int:
    height, _ = _cosmos_latest(session, rpc)
    return height


def _cosmos_latest(session: requests.Session, rpc: str) -> Tuple[int, int]:
    try:
        res = _cosmos_get(session, rpc, "/block", _is_valid_block_resp)
        header = res["result"]["block"]["header"]
    except Exception as rpc_exc:
        last_exc = rpc_exc
        for base in OSMOSIS_LCD_FALLBACKS:
            try:
                res = _get_json(session, f"{base.rstrip('/')}/cosmos/base/tendermint/v1beta1/blocks/latest")
                if not _is_valid_block_resp_lcd(res):
                    raise ValueError("invalid LCD latest block response")
                header = res["block"]["header"]
                break
            except Exception as lcd_exc:
                last_exc = lcd_exc
                continue
        else:
            raise last_exc
    height = int(header["height"])
    ts = pd.to_datetime(header["time"], utc=True)
    return height, int(ts.timestamp())


def _cosmos_block_ts(session: requests.Session, rpc: str, height: int, cache: Dict[int, pd.Timestamp]) -> int:
    ts, _ = _cosmos_block(session, rpc, height, cache)
    return int(ts.timestamp())


def _cosmos_consensus_max_gas(session: requests.Session, rpc: str, height: int) -> Optional[int]:
    res = _cosmos_get(session, rpc, f"/consensus_params?height={height}")
    block_params = res["result"]["consensus_params"]["block"]
    val = int(block_params.get("max_gas", "-1"))
    return val if val > 0 else None


def _cosmos_block_results(session: requests.Session, rpc: str, height: int) -> Tuple[int, Optional[int], Optional[int]]:
    res = _cosmos_get(session, rpc, f"/block_results?height={height}")
    result = res["result"]
    gas_used = 0
    for txr in result.get("txs_results") or []:
        gas_used += int(txr.get("gas_used", "0"))
    updates = result.get("consensus_param_updates", {})
    new_max_gas = None
    if updates and "block" in updates and updates["block"] and "max_gas" in updates["block"]:
        val = int(updates["block"]["max_gas"])
        new_max_gas = val if val > 0 else None
    base_fee = _cosmos_base_fee(result)
    if base_fee is None:
        base_fee = _cosmos_base_fee_api(session, height)
    return gas_used, new_max_gas, base_fee


def _cosmos_base_fee(result: Dict) -> Optional[int]:
    """Extract EIP-like base fee from begin/end block events if present (Osmosis txfees module)."""
    events = (result.get("begin_block_events") or []) + (result.get("end_block_events") or [])
    for ev in events:
        attrs = ev.get("attributes") or []
        for attr in attrs:
            try:
                key = base64.b64decode(attr.get("key", "")).decode("utf-8")
                val_raw = base64.b64decode(attr.get("value", "")).decode("utf-8")
            except Exception:
                continue
            if key in {"base_fee", "eip_base_fee"}:
                try:
                    return int(val_raw)
                except ValueError:
                    continue
    return None


def _cosmos_base_fee_api(session: requests.Session, height: int) -> Optional[int]:
    """Fallback: query LCD txfees endpoint with explicit height header (x-cosmos-block-height)."""
    query_height = max(1, height - 1)  # align with base fee used when entering block height
    for base in OSMOSIS_LCD_FALLBACKS:
        url = f"{base.rstrip('/')}/osmosis/txfees/v1beta1/cur_eip_base_fee"
        headers = {"x-cosmos-block-height": str(query_height)}
        try:
            resp = session.get(url, headers=headers, timeout=getattr(session, "request_timeout", 30))
            resp.raise_for_status()
            data = resp.json()
            # Common shapes: {"base_fee":"1234"} or {"eip_base_fee":"1234"} or nested under "base_fee":"1234"
            for key in ("base_fee", "eip_base_fee"):
                if key in data:
                    try:
                        return int(data[key])
                    except (TypeError, ValueError):
                        continue
            nested = data.get("base_fee")
            if isinstance(nested, dict):
                for key in ("amount", "base_fee", "eip_base_fee"):
                    if key in nested:
                        try:
                            return int(nested[key])
                        except (TypeError, ValueError):
                            continue
        except Exception:
            continue
    return None


def _cosmos_block_load(
    session: requests.Session,
    rpc: str,
    height: int,
    ts_cache: Dict[int, Tuple[pd.Timestamp, int]],
    max_gas_cache: Dict[str, Optional[int]],
) -> Tuple[pd.Timestamp, int, Optional[int], float, int, Optional[int]]:
    ts, tx_count = _cosmos_block(session, rpc, height, ts_cache)
    gas_used, updated_max_gas, base_fee = _cosmos_block_results(session, rpc, height)
    if updated_max_gas is not None:
        max_gas_cache["value"] = updated_max_gas
    max_gas = max_gas_cache["value"]
    load = gas_used / max_gas if max_gas else float("nan")
    return ts, gas_used, max_gas, load, tx_count, base_fee


# -------- Shared routines --------
def _iter_day_heights(
    session: requests.Session,
    rpc: str,
    chain: str,
    date: dt.date,
    latest_height: int,
    latest_ts: int,
    ts_cache,
) -> Tuple[int, int]:
    start_ts = _epoch(dt.datetime.combine(date, dt.time(0, 0), tzinfo=dt.timezone.utc))
    end_ts = _epoch(dt.datetime.combine(date + dt.timedelta(days=1), dt.time(0, 0), tzinfo=dt.timezone.utc)) - 1
    bt_hint = _block_time_hint(chain)
    low_start, high_start = _height_bounds(latest_height, latest_ts, start_ts, bt_hint)
    low_end, high_end = _height_bounds(latest_height, latest_ts, end_ts, bt_hint)
    ts_fn = _evm_block_ts if chain == "eth" else _cosmos_block_ts
    start_height = _find_block_ge_ts(session, rpc, start_ts, latest_height, ts_fn, ts_cache, low_start, high_start)
    end_height = _find_block_le_ts(session, rpc, end_ts, latest_height, ts_fn, ts_cache, low_end, high_end)
    return start_height, end_height


def _fetch_full_day(
    session: requests.Session,
    rpc: str,
    chain: str,
    start_h: int,
    end_h: int,
    ts_cache,
    max_gas_cache,
    verbose: bool = False,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for h in range(start_h, end_h + 1):
        if chain == "eth":
            ts, gas_used, gas_limit, load, txc, base_fee = _evm_block_load(session, rpc, h, ts_cache)
        else:
            ts, gas_used, gas_limit, load, txc, base_fee = _cosmos_block_load(
                session, rpc, h, ts_cache, max_gas_cache
            )
        rows.append(
            {
                "timestamp": ts,
                "height": h,
                "gas_used": gas_used,
                "gas_limit": gas_limit,
                "load": load,
                "tx_count": txc,
                "base_fee": base_fee,
            }
        )
        if verbose:
            print(
                f"[block] {chain} h={h} ts={ts} gas_used={gas_used} gas_limit={gas_limit} base_fee={base_fee}",
                flush=True,
            )
    df = pd.DataFrame(rows)
    df.sort_values("height", inplace=True)
    return df


def _run_stats(loads: pd.Series, over_level: float) -> Dict:
    """Run-length stats for consecutive blocks above a level."""
    flags = (loads > over_level).astype(int)
    runs: List[int] = []
    cur = 0
    for flag in flags:
        if flag:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    if not runs:
        return {"runs_over_target": 0, "max_run_over_target": 0, "mean_run_over_target": 0.0}
    return {
        "runs_over_target": len(runs),
        "max_run_over_target": max(runs),
        "mean_run_over_target": sum(runs) / len(runs),
    }


def _metrics(loads: pd.Series, target: float, threshold: float) -> Dict:
    clean = loads.dropna()
    if clean.empty:
        return {
            "mean_load": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "p_over_target": 0.0,
            "p_over_threshold": 0.0,
            "runs_over_target": 0,
            "max_run_over_target": 0,
            "mean_run_over_target": 0.0,
        }
    err = clean - target
    mae = float(err.abs().mean())
    rmse = float(math.sqrt((err.pow(2).mean())))
    mean_load = float(clean.mean())
    p_over_target = float((clean > target).mean())
    p_over_threshold = float((clean > threshold).mean())
    run_stats = _run_stats(clean, target)
    return {
        "mean_load": mean_load,
        "mae": mae,
        "rmse": rmse,
        "p_over_target": p_over_target,
        "p_over_threshold": p_over_threshold,
        **run_stats,
    }


def _save_daily_metrics(rows: List[Dict], chain: str, start_date: dt.date, end_date: dt.date) -> Optional[str]:
    if not rows:
        return None
    daily_path = f"data/{chain}_load_metrics_daily_{start_date.isoformat()}_to_{end_date.isoformat()}.csv"
    pd.DataFrame(rows).to_csv(daily_path, index=False)
    return daily_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Load stability metrics over a UTC date range.")
    parser.add_argument("--chain", required=True, choices=["eth", "osmosis"], help="Chain to query.")
    parser.add_argument("--rpc", help="RPC endpoint (HTTP). Default depends on chain.")
    parser.add_argument("--start-date", required=True, help="UTC start date YYYY-MM-DD (inclusive).")
    parser.add_argument("--end-date", help="UTC end date YYYY-MM-DD (inclusive). Defaults to start-date.")
    parser.add_argument("--target", type=float, help="Target load. Defaults: ETH=0.5, Osmosis=0.25.")
    parser.add_argument("--threshold", type=float, default=0.9, help="Saturation threshold for probability metric.")
    parser.add_argument("--block-max-gas", type=int, help="Override max_gas for Cosmos if RPC omits it.")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout seconds.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start_date)
    end_date = dt.date.fromisoformat(args.end_date) if args.end_date else start_date
    if start_date > end_date:
        raise SystemExit("start-date must be <= end-date")

    if args.rpc:
        rpc = args.rpc.rstrip("/")
    else:
        rpc = (DEFAULT_ETH_RPC if args.chain == "eth" else DEFAULT_OSMOSIS_RPC).rstrip("/")
    os.makedirs("data", exist_ok=True)

    session = _build_session(args.timeout)
    ts_cache: Dict = {}
    max_gas_cache = {"value": args.block_max_gas}

    target = args.target
    if target is None:
        target = 0.5 if args.chain == "eth" else 0.25

    if args.chain == "eth":
        latest_height, latest_ts = _evm_latest(session, rpc)
        latest_dt = pd.to_datetime(latest_ts, unit="s", utc=True)
        if args.verbose:
            print(f"[ETH] latest block {latest_height} @ {latest_dt}")
    else:
        latest_height, latest_ts = _cosmos_latest(session, rpc)
        latest_dt = pd.to_datetime(latest_ts, unit="s", utc=True)
        if max_gas_cache["value"] is None:
            max_gas_cache["value"] = _cosmos_consensus_max_gas(session, rpc, latest_height)
        if args.verbose:
            print(f"[Cosmos] latest height {latest_height} @ {latest_dt}, max_gas={max_gas_cache['value']}")

    latest_date = latest_dt.date()
    if end_date > latest_date:
        raise SystemExit(
            f"Requested window ends after tip: end-date {end_date}, latest block date {latest_date} (height {latest_height})."
        )

    day_frames: List[pd.DataFrame] = []
    daily_metrics: List[Dict] = []

    cur = start_date
    while cur <= end_date:
        start_h, end_h = _iter_day_heights(session, rpc, args.chain, cur, latest_height, latest_ts, ts_cache)
        if start_h > end_h:
            if args.verbose:
                print(f"[warn] no blocks found for {cur}")
            cur += dt.timedelta(days=1)
            continue
        if args.verbose:
            print(f"[day] {cur} heights {start_h}->{end_h}")
        day_df = _fetch_full_day(session, rpc, args.chain, start_h, end_h, ts_cache, max_gas_cache, args.verbose)
        day_df["date"] = cur.isoformat()
        day_df["error_vs_target"] = day_df["load"] - target
        day_frames.append(day_df)

        m = _metrics(day_df["load"], target, args.threshold)
        m.update(
            {
                "chain": args.chain,
                "date": cur.isoformat(),
                "target": target,
                "threshold": args.threshold,
                "blocks": len(day_df),
                "gas_used_total": int(day_df["gas_used"].sum()),
                "tx_total": int(day_df["tx_count"].sum()),
                "avg_tx_per_block": float(day_df["tx_count"].mean()),
            }
        )
        daily_metrics.append(m)

        # Save per-day block data for quick inspection.
        day_out = f"data/{args.chain}_load_{cur.isoformat()}.csv"
        day_df.to_csv(day_out, index=False)
        cur += dt.timedelta(days=1)

    if not day_frames:
        raise SystemExit("No blocks found in the requested window.")

    all_df = pd.concat(day_frames, ignore_index=True)
    all_df.sort_values(["timestamp", "height"], inplace=True)
    loads = all_df["load"]

    metrics = _metrics(loads, target, args.threshold)
    metrics.update(
        {
            "chain": args.chain,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "target": target,
            "threshold": args.threshold,
            "blocks": len(all_df),
            "gas_used_total": int(all_df["gas_used"].sum()),
            "tx_total": int(all_df["tx_count"].sum()),
            "avg_tx_per_block": float(all_df["tx_count"].mean()),
        }
    )

    range_csv = f"data/{args.chain}_load_{start_date.isoformat()}_to_{end_date.isoformat()}.csv"
    all_df.to_csv(range_csv, index=False)

    metrics_path = f"data/{args.chain}_load_metrics_{start_date.isoformat()}_to_{end_date.isoformat()}.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    daily_metrics_path = _save_daily_metrics(daily_metrics, args.chain, start_date, end_date)

    print(f"Saved blocks: {range_csv}")
    print(f"Metrics: {metrics_path}")
    if daily_metrics_path:
        print(f"Per-day metrics: {daily_metrics_path}")


if __name__ == "__main__":
    main()
