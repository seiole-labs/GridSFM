#!/usr/bin/env bash
# Run the cost-error report only after the sequential training sweep completes.
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"
sweep_dir="${1:?pass the sweep log directory}"
python_bin="${PYTHON_BIN:-python}"

while true; do
    status="$(cat "$sweep_dir/status.txt")"
    case "$status" in
        complete) break ;;
        running) sleep 30 ;;
        *) printf 'Sweep ended without completion: %s\n' "$status" >&2; exit 1 ;;
    esac
done

printf 'running\n' > "$sweep_dir/cost_report_status.txt"
if PYTHONPATH=model "$python_bin" \
    model/examples/report_lora_sweep_cost_max.py --sweep-dir "$sweep_dir" \
    2>&1 | tee "$sweep_dir/cost_report.log"; then
    printf 'complete\n' > "$sweep_dir/cost_report_status.txt"
else
    result=$?
    printf 'failed (exit %s)\n' "$result" > "$sweep_dir/cost_report_status.txt"
    exit "$result"
fi
