# Benchmark Timing Boundaries

This document defines exactly what each reported duration measures. The current pilot uses model-resident GridSFM inference and a combined PowerModels construction plus Ipopt optimization measurement.

## Current observations

| Observation | Starts | Stops | Included | Excluded | Ohio value |
|---|---|---|---|---|---:|
| GridSFM forward | Immediately before `model(batch)` | Immediately after `model(batch)` and device synchronization | Neural forward computation, DC prior, predicted branch-flow calculation, feasibility head | Process startup, checkpoint loading, input parsing, feature preparation, batching, device transfer, CPU conversion, JSON serialization | 2.171 s |
| GridSFM request | Before loading the `.pyg.json` | After prediction tensors are detached and moved to CPU | Input JSON loading, PyG conversion, cycle/positional feature preparation, batching, device transfer, GridSFM forward, prediction tensors moved to CPU | Process startup, imports, checkpoint loading, Python-list conversion, JSON serialization, output writing | 2.401 s |
| AC-OPF solve | Before `PowerModels.instantiate_model` | After `PowerModels.optimize_model!` returns | PowerModels/JuMP nonlinear-model construction, Ipopt initialization, optimization, solver-result construction | Docker/Julia startup, input JSON parsing, GridSFM export-object construction, `.pyg.json` serialization and writing | 10.727 s |

The implementation locations are:

- GridSFM timers: [`benchmark/run_inference.py`](run_inference.py)
- AC-OPF timer: [`power_grid/US/topology_solver_pipeline/export_gridsfm_data.jl`](../power_grid/US/topology_solver_pipeline/export_gridsfm_data.jl)

## What is not currently measured

| Missing observation | Meaning |
|---|---|
| GridSFM checkpoint load | Time to read, validate, construct, and move the v1.1 model to its device |
| GridSFM cold process | Python startup + imports + checkpoint load + one complete request + result serialization |
| GridSFM serialization | Tensor-to-list conversion and prediction JSON writing |
| AC-OPF optimizer-only | Ipopt execution separated from PowerModels/JuMP model construction |
| AC-OPF complete request | Input parsing + model construction + optimization + result conversion |
| AC-OPF cold process | Docker/Julia startup + complete request + output writing |

## Synchronization

CPU operations are synchronous. CUDA operations are normally asynchronous, so GPU timing synchronizes the selected CUDA device immediately before starting and immediately after finishing the forward pass. Without these synchronizations, the host timer could stop before GPU kernels complete and under-report latency.

## Cache state

GridSFM preprocessing uses topology-related caches for cycle bases and numerical factorizations. The reported Ohio run used a new temporary cache directory, so its preprocessing portion was cold. The checkpoint was already loaded and is therefore warm/model-resident.

The forward measurement is unaffected by disk cache state because feature preparation completes before its timer starts. The request measurement is cache-sensitive and must be labelled `cold-cache` or `warm-cache` in future repeated runs.

## Which comparisons are defensible now?

The current observations describe useful components, but their boundaries are not identical:

```text
GridSFM request
  = parse + prepare + transfer + forward + CPU tensor return

AC-OPF solve
  = construct optimization model + optimize
```

The provisional ratio `10.727 / 2.401 = 4.47x` must not yet be presented as a formal end-to-end speedup because AC input parsing is excluded while GridSFM input parsing and preparation are included, and only one run was measured.

For a customer-facing comparison, report two matched pairs:

1. **Resident compute:** GridSFM forward versus Ipopt optimizer-only.
2. **Warm request:** GridSFM parse-to-prediction versus AC-OPF parse-to-solved-result.

Optionally report cold-process startup separately. Never mix cold startup into only one side of the comparison.

## Required repeated-run protocol

- Use the exact same transformed scenario.
- Record CPU/GPU, RAM/VRAM, OS, and batch size.
- Record cold-cache and warm-cache state.
- Run warmups before resident-compute measurements.
- Synchronize CUDA around every measured GPU interval.
- Repeat each observation and report median, p5, and p95 rather than a single run.
- Keep raw per-run timings in the machine-readable result.
- Report checkpoint/model loading and process startup separately, not hidden inside one method.
