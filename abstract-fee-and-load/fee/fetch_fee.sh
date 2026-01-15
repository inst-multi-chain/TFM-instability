#!/usr/bin/env bash
set -e

# Moonbeam 2025-05-29 UTC
python fetch_fee_range.py \
  --from-ts "2025-05-29T00:00:00Z" --to-ts "2025-05-30T00:00:00Z" \
  --eth-rpc "https://rpc.api.moonbeam.network" \
  --tip-percentile 50 --verbose
mv data/eth_fee_range.csv data/moonbeam_fee_range.csv

# Ethereum 2025-12-23 UTC
python fetch_fee_range.py \
  --from-ts "2025-12-23T00:00:00Z" --to-ts "2025-12-24T00:00:00Z" \
  --eth-rpc "https://mainnet.infura.io/v3/b424aba0f7c04c6bb7c1e7c4e0f869c2" \
  --tip-percentile 50 --verbose
mv data/eth_fee_range.csv data/eth_fee_range_2025-12-23.csv
