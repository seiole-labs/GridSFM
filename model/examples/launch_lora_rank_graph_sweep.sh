#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

output_root="artifacts/lora_rank_graph_sweep_runs"
log_root="artifacts/training_logs/lora_rank_graph_sweep"
gpu_lock="/tmp/gridsfm_lora_gpu.lock"
mkdir -p "$output_root" "$log_root"

first_job=1
for rank in 4 8 16; do
    for graphs in 10 100 500 1000; do
        session="lora-r${rank}-n${graphs}-e8"
        run_name="case6470_lora_r${rank}_a${rank}_n${graphs}_e8"
        log_path="$log_root/r${rank}_a${rank}_n${graphs}_e8.log"
        if tmux has-session -t "$session" 2>/dev/null; then
            echo "session already exists: $session" >&2
            exit 1
        fi
        command="printf 'queued: rank=${rank} alpha=${rank} graphs=${graphs} epochs=8\\n' | tee '$log_path'; flock -x '$gpu_lock' bash -lc \"printf 'acquired_gpu_lock\\n' | tee -a '$log_path'; PYTHONUNBUFFERED=1 PYTHONPATH=model python model/examples/train_lora_case6470.py --config model/examples/lora_case6470.yaml --run-name '$run_name' --train-graphs '$graphs' --epochs 8 --rank '$rank' --alpha '$rank' --output-root '$output_root' 2>&1 | tee -a '$log_path'; run_dir=\\\$(find '$output_root' -mindepth 1 -maxdepth 1 -type d -name '*_${run_name}' | sort | tail -n 1); python model/examples/plot_lora_test_results.py --metrics \\\"\\\$run_dir/metrics.jsonl\\\" --output \\\"\\\$run_dir/test_results.png\\\" 2>&1 | tee -a '$log_path'; python model/examples/plot_lora_paper_comparison.py --metrics \\\"\\\$run_dir/metrics.jsonl\\\" --output \\\"\\\$run_dir/paper_comparison.png\\\" 2>&1 | tee -a '$log_path'\""
        tmux new-session -d -s "$session" bash -lc "$command"
        if [[ "$first_job" -eq 1 ]]; then
            sleep 3
            first_job=0
        fi
    done
done

tmux list-sessions
