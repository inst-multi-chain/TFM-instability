
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
OUT_DIR="${ROOT}/data/base_fee_hourly"
mkdir -p "${OUT_DIR}"

# HTTP RPC used to locate block heights (can override via RPC env).
RPC="${RPC:-https://osmosis-rpc.publicnode.com:443}"
# gRPC endpoint for base fee query (can override via GRPC_ENDPOINT env).
GRPC_ENDPOINT="${GRPC_ENDPOINT:-osmosis-grpc.publicnode.com:443}"
# Average block time hint (seconds) for binary search.
BLOCK_TIME_HINT="${BLOCK_TIME_HINT:-1.0}"

WINDOWS=(
  "2025-10-10 21:00:00,2025-10-10 22:00:00"
  "2025-10-10 22:00:00,2025-10-10 23:00:00"
)

locate_heights() {
  local start_ts="$1" end_ts="$2" rpc="$3" bt_hint="$4"
  python3 - <<'PY' "${start_ts}" "${end_ts}" "${rpc}" "${bt_hint}" "${REPO_ROOT}"
import datetime as dt
import sys
import types
from pathlib import Path

start_str, end_str, rpc, bt_hint, repo_root = sys.argv[1:]

# Minimal pandas shim with tolerant timestamp parsing (handles long fractional seconds).
def _to_datetime(val, utc=False, unit=None):
    if isinstance(val, dt.datetime):
        ts = val
    elif isinstance(val, (int, float)) and unit == "s":
        ts = dt.datetime.fromtimestamp(val, tz=dt.timezone.utc)
    elif isinstance(val, str):
        s = val.replace("Z", "+00:00")
        if "." in s:
            main, tz = s.rsplit("+", 1) if "+" in s[1:] else (s, "")
            if tz:
                tz = "+" + tz
            base, frac = main.split(".", 1)
            frac = (frac + "000000")[:6]
            s = f"{base}.{frac}{tz}"
        ts = dt.datetime.fromisoformat(s)
    else:
        ts = dt.datetime.utcfromtimestamp(0)
    return ts.astimezone(dt.timezone.utc) if utc and ts.tzinfo else ts


class _Series(list):
    pass


class _DataFrame(list):
    pass


pd_stub = types.SimpleNamespace(
    to_datetime=_to_datetime,
    Timestamp=dt.datetime,
    Series=_Series,
    DataFrame=_DataFrame,
)
sys.modules.setdefault("pandas", pd_stub)

sys.path.insert(0, str(Path(repo_root)))
from load import range_load_extract as r  # noqa: E402

start_dt = dt.datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
end_dt = dt.datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)

session = r._build_session(timeout=10)
ts_cache = {}
latest_height, latest_ts = r._cosmos_latest(session, rpc)

start_ts = r._epoch(start_dt)
end_ts = r._epoch(end_dt) - 1
bt_hint = float(bt_hint)
low_start, high_start = r._height_bounds(latest_height, latest_ts, start_ts, bt_hint)
low_end, high_end = r._height_bounds(latest_height, latest_ts, end_ts, bt_hint)
start_h = r._find_block_ge_ts(session, rpc, start_ts, latest_height, r._cosmos_block_ts, ts_cache, low_start, high_start)
end_h = r._find_block_le_ts(session, rpc, end_ts, latest_height, r._cosmos_block_ts, ts_cache, low_end, high_end)

print(f"{start_h} {end_h}")
PY
}

convert_fee() {
  local raw="$1"
  python3 - <<'PY' "${raw}"
import decimal, sys
raw = sys.argv[1]
q = decimal.Decimal(raw) / decimal.Decimal(10 ** 18)
print(q)
PY
}

for window in "${WINDOWS[@]}"; do
  IFS=',' read -r START_TS END_TS <<<"${window}"
  read -r START_H END_H < <(locate_heights "${START_TS}" "${END_TS}" "${RPC}" "${BLOCK_TIME_HINT}")

  # Sanity check
  if [[ -z "${START_H}" || -z "${END_H}" || "${START_H}" -gt "${END_H}" ]]; then
    echo "[warn] no heights for window ${START_TS}-${END_TS} (got ${START_H}-${END_H})" >&2
    continue
  fi

  start_tag="${START_TS//[: ]/_}"
  end_tag="${END_TS//[: ]/_}"
  out_file="${OUT_DIR}/osmosis_base_fee_${start_tag}_${end_tag}.csv"
  echo "height,base_fee_raw,base_fee_osmo,status,error" >"${out_file}"

  echo "=== ${START_TS} -> ${END_TS} heights ${START_H}-${END_H} ==="
  for ((h = START_H; h <= END_H; h++)); do
    status="ok"
    err=""
    raw=""
    fee=""
    # grpcurl may fail; continue on error.
    if ! raw=$(grpcurl -insecure -H "x-cosmos-block-height: ${h}" -d '{}' "${GRPC_ENDPOINT}" osmosis.txfees.v1beta1.Query/GetEipBaseFee 2>/dev/null | jq -r '.baseFee // .base_fee // empty'); then
      status="error"
      err="grpc_failed"
    elif [[ -z "${raw}" ]]; then
      status="error"
      err="missing_baseFee"
    else
      if fee=$(convert_fee "${raw}" 2>/dev/null); then
        :
      else
        status="error"
        err="convert_failed"
      fi
    fi
    echo "${h},${raw},${fee},${status},${err}" | tee -a "${out_file}"
  done
done
