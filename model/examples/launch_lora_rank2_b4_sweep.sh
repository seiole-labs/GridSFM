#!/usr/bin/env bash
# Run the rank-2 graph-count sweep sequentially, one physical batch of 4 at a time.
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

python_bin="${PYTHON_BIN:-python}"
log_root="artifacts/training_logs/lora_rank2_b4_sweep"
mkdir -p "$log_root"

for graphs in 10 100 500 1000; do
    config="model/examples/lora_case6470_rank2_n${graphs}.yaml"
    run_name="case6470_lora_r2_a2_n${graphs}_e8_b4"
    log_path="$log_root/${run_name}_$(date -u +%Y%m%dT%H%M%SZ).log"

    printf 'Starting %s with %s\n' "$run_name" "$config" | tee "$log_path"
    PYTHONUNBUFFERED=1 PYTHONPATH=model "$python_bin" \
        model/examples/train_lora_case6470.py --config "$config" \
        2>&1 | tee -a "$log_path"
done
