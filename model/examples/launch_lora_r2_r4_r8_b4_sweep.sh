#!/usr/bin/env bash
# Run all twelve rank/graph configurations sequentially in one shell.
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

python_bin="${PYTHON_BIN:-python}"
log_root="artifacts/training_logs/lora_r2_r4_r8_b4_sweep"
sweep_id="$(date -u +%Y%m%dT%H%M%SZ)"
sweep_dir="$log_root/$sweep_id"

# Check the complete matrix before starting the first training job.
for rank in 2 4 8; do
    for graphs in 10 100 500 1000; do
        config="model/examples/lora_case6470_rank${rank}_n${graphs}.yaml"
        if [[ ! -f "$config" ]]; then
            printf 'Missing config: %s\n' "$config" >&2
            exit 1
        fi
    done
done

mkdir -p "$sweep_dir"
printf 'running\n' > "$sweep_dir/status.txt"
finish() {
    result=$?
    if (( result == 0 )); then
        printf 'complete\n' > "$sweep_dir/status.txt"
    else
        printf 'failed (exit %s)\n' "$result" > "$sweep_dir/status.txt"
    fi
}
trap finish EXIT

printf 'Sweep logs: %s\n' "$sweep_dir"
for rank in 2 4 8; do
    for graphs in 10 100 500 1000; do
        config="model/examples/lora_case6470_rank${rank}_n${graphs}.yaml"
        run_name="case6470_lora_r${rank}_a${rank}_n${graphs}_e8_b4"
        log_path="$sweep_dir/${run_name}.log"
        printf 'Starting %s with %s\n' "$run_name" "$config" | tee "$log_path"
        PYTHONUNBUFFERED=1 PYTHONPATH=model "$python_bin" \
            model/examples/train_lora_case6470.py --config "$config" \
            2>&1 | tee -a "$log_path"
        printf 'Completed %s\n' "$run_name"
    done
done
