
import argparse
import datetime as dt
import os
import time
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_ETH_RPC = "your_key"


def _build_session(timeout: int = 20) -> requests.Session:
    sess = requests.Session()
    retry = Retry(total=3, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"])
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    sess.request_timeout = timeout
    return sess


def _rpc_call(session: requests.Session, url: str, method: str, params: Sequence) -> Dict:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": list(params)}
    resp = session.post(url, json=payload, timeout=getattr(session, "request_timeout", 20))
    resp.raise_for_status()
    body = resp.json()
    if "error" in body:
        raise RuntimeError(f"RPC error on {method}: {body['error']}")
    return body["result"]


def _get_block(session: requests.Session, rpc: str, number: int) -> Dict:
    return _rpc_call(session, rpc, "eth_getBlockByNumber", [hex(number), False])


def _latest_block(session: requests.Session, rpc: str) -> Tuple[int, int]:
    latest = _rpc_call(session, rpc, "eth_getBlockByNumber", ["latest", False])
    num = int(latest["number"], 16)
    ts = int(latest["timestamp"], 16)
    return num, ts


def _find_block_by_timestamp(session: requests.Session, rpc: str, target_ts: int, latest_num: int) -> int:
    """Binary search for the first block with timestamp >= target_ts."""
    low, high = 0, latest_num
    while low < high:
        mid = (low + high) // 2
        blk = _get_block(session, rpc, mid)
        ts = int(blk["timestamp"], 16)
        if ts < target_ts:
            low = mid + 1
        else:
            high = mid
    return low


def _find_block_lte_timestamp(session: requests.Session, rpc: str, target_ts: int, latest_num: int) -> int:
    """Binary search for the last block with timestamp <= target_ts."""
    low, high = 0, latest_num
    while low < high:
        mid = (low + high + 1) // 2
        blk = _get_block(session, rpc, mid)
        ts = int(blk["timestamp"], 16)
        if ts <= target_ts:
            low = mid
        else:
            high = mid - 1
    return low


def _fee_history_range(
    session: requests.Session, rpc: str, start_block: int, end_block: int, percentile: int, verbose: bool = False
) -> pd.DataFrame:
    """Collect base fee & tip percentile for [start_block, end_block] inclusive."""
    rows: List[Dict] = []
    remaining = end_block - start_block + 1
    cursor = end_block
    chunk = 100  # conservative blockCount to avoid provider limits
    while remaining > 0:
        take = min(chunk, remaining)
        try:
            res = _rpc_call(session, rpc, "eth_feeHistory", [hex(take), hex(cursor), [percentile]])
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of bounds" in msg or "invalid block range" in msg or "block range" in msg:
                # shrink chunk and retry
                if chunk == 1:
                    raise
                chunk = max(1, chunk // 2)
                if verbose:
                    print(f"[feeHistory] provider rejected block range, reducing chunk to {chunk} and retrying...")
                continue
            raise
        oldest = int(res["oldestBlock"], 16)
        base_fees = res["baseFeePerGas"][:-1]
        rewards = res["reward"]
        for i, bf_hex in enumerate(base_fees):
            bn = oldest + i
            base_fee = int(bf_hex, 16) / 1e9
            tip = int(rewards[i][0], 16) / 1e9 if rewards[i] else 0.0
            rows.append({"block": bn, "base_fee_gwei": base_fee, "tip_gwei": tip})
        remaining -= take
        cursor = oldest - 1
        if verbose:
            print(f"[feeHistory] collected up to block {cursor}, {remaining} remaining (chunk {chunk})...")
    df = pd.DataFrame(rows)
    df.sort_values("block", inplace=True)

    # fetch timestamps
    ts_list: List[int] = []
    for bn in df["block"]:
        blk = _get_block(session, rpc, int(bn))
        ts_list.append(int(blk["timestamp"], 16))
        time.sleep(0.005)
    df["timestamp"] = pd.to_datetime(ts_list, unit="s", utc=True)
    df["gas_price_gwei"] = df["base_fee_gwei"] + df["tip_gwei"]
    return df[["timestamp", "block", "base_fee_gwei", "tip_gwei", "gas_price_gwei"]]


def _manual_block_scan(
    session: requests.Session, rpc: str, start_block: int, end_block: int, verbose: bool = False
) -> pd.DataFrame:
    """Fallback: per-block scan with full txs, compute median tip."""
    rows: List[Dict] = []
    for bn in range(start_block, end_block + 1):
        blk = _rpc_call(session, rpc, "eth_getBlockByNumber", [hex(bn), True])
        ts = int(blk["timestamp"], 16)
        base_fee = int(blk.get("baseFeePerGas", "0x0"), 16) / 1e9
        tips: List[float] = []
        for tx in blk.get("transactions", []):
            # prefer maxPriorityFeePerGas if present, else use gasPrice - base
            if "maxPriorityFeePerGas" in tx and tx["maxPriorityFeePerGas"]:
                tip_val = int(tx["maxPriorityFeePerGas"], 16) / 1e9
            else:
                gp = int(tx.get("gasPrice", "0x0"), 16) / 1e9
                tip_val = max(gp - base_fee, 0.0)
            tips.append(tip_val)
        if tips:
            tip_med = float(pd.Series(tips).median())
        else:
            tip_med = 0.0
        rows.append(
            {
                "timestamp": pd.to_datetime(ts, unit="s", utc=True),
                "block": bn,
                "base_fee_gwei": base_fee,
                "tip_gwei": tip_med,
                "gas_price_gwei": base_fee + tip_med,
            }
        )
        if verbose and (bn - start_block) % 100 == 0:
            print(f"[manual] scanned block {bn} ({bn - start_block}/{end_block - start_block} done)")
        time.sleep(0.005)
    return pd.DataFrame(rows)


def fetch_range_for_chain(
    session: requests.Session,
    rpc: str,
    start_ts: int,
    end_ts: int,
    percentile: int,
    label: str,
    verbose: bool = False,
) -> pd.DataFrame:
    latest_num, latest_ts = _latest_block(session, rpc)
    if verbose:
        print(f"[{label}] latest block {latest_num} @ {pd.to_datetime(latest_ts, unit='s', utc=True)}")
    start_block = _find_block_by_timestamp(session, rpc, start_ts, latest_num)
    end_block = _find_block_lte_timestamp(session, rpc, end_ts, latest_num)
    if start_block > end_block:
        raise SystemExit(f"[{label}] No blocks in the given window.")
    if verbose:
        print(f"[{label}] block range {start_block} -> {end_block} (count {end_block - start_block + 1})")
    try:
        df = _fee_history_range(session, rpc, start_block, end_block, percentile, verbose=verbose)
    except RuntimeError as exc:
        if verbose:
            print(f"[{label}] feeHistory failed ({exc}); falling back to manual per-block scan (slower).")
        df = _manual_block_scan(session, rpc, start_block, end_block, verbose=verbose)
    return df


def _iso_to_epoch(ts: str) -> int:
    return int(dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch fee history for ETH and optional EVM chain by time range.")
    parser.add_argument("--from-ts", required=True, help="Start time ISO8601, e.g. 2024-12-26T12:00:00Z")
    parser.add_argument("--to-ts", required=True, help="End time ISO8601, e.g. 2024-12-26T14:00:00Z")
    parser.add_argument("--eth-rpc", default=os.environ.get("ETH_RPC_URL", DEFAULT_ETH_RPC), help="ETH RPC URL.")
    parser.add_argument("--evm-rpc", help="Optional EVM RPC URL (Moonbeam/Astar/Acala etc.).")
    parser.add_argument("--evm-chain", default="evm", help="Slug for EVM output file, default evm.")
    parser.add_argument("--tip-percentile", type=int, default=50, help="Tip percentile (default 50 for median).")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout seconds.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    start_ts = _iso_to_epoch(args.from_ts)
    end_ts = _iso_to_epoch(args.to_ts)
    if end_ts <= start_ts:
        raise SystemExit("to-ts must be later than from-ts")

    session = _build_session(timeout=args.timeout)
    os.makedirs("data", exist_ok=True)

    if args.verbose:
        print(f"[ETH] window {args.from_ts} -> {args.to_ts}, RPC {args.eth_rpc}")
    eth_df = fetch_range_for_chain(
        session,
        args.eth_rpc,
        start_ts,
        end_ts,
        args.tip_percentile,
        label="ETH",
        verbose=args.verbose,
    )
    eth_path = "data/eth_fee_range.csv"
    eth_df.to_csv(eth_path, index=False)
    if args.verbose:
        print(f"[ETH] saved {len(eth_df)} rows to {eth_path}")

    if args.evm_rpc:
        slug = args.evm_chain.replace(" ", "_")
        if args.verbose:
            print(f"[EVM] window {args.from_ts} -> {args.to_ts}, RPC {args.evm_rpc}")
        evm_df = fetch_range_for_chain(
            session,
            args.evm_rpc,
            start_ts,
            end_ts,
            args.tip_percentile,
            label=slug,
            verbose=args.verbose,
        )
        evm_path = f"data/{slug}_fee_range.csv"
        evm_df.to_csv(evm_path, index=False)
        if args.verbose:
            print(f"[EVM] saved {len(evm_df)} rows to {evm_path}")

    print("Done.")


if __name__ == "__main__":
    main()
