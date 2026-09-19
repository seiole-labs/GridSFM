# Benchmark Timing Boundaries

This document defines exactly what each reported duration measures. GridSFM now records both a model-resident request and an externally observed cold CLI request. Historical prediction artifacts used `end_to_end_seconds` for the resident request; new artifacts use the honest names below.

## Current observations

| Observation | Starts | Stops | Included | Excluded | Ohio value |
|---|---|---|---|---|---:|
| GridSFM forward | Immediately before `model(batch)` | Immediately after `model(batch)` and device synchronization | Neural forward computation, DC prior, predicted branch-flow calculation, feasibility head | Process startup, checkpoint loading, input parsing, feature preparation, batching, device transfer, CPU conversion, JSON serialization | 2.171 s |
| GridSFM resident request | Before loading the `.pyg.json` | After prediction tensors are detached and moved to CPU | Input JSON loading, PyG conversion, cycle/positional feature preparation, batching, device transfer, GridSFM forward, prediction tensors moved to CPU | Process startup, imports, checkpoint loading, Python-list conversion, JSON serialization, output writing | Historical Ohio value: 2.401 s |
| GridSFM cold request E2E | Immediately before launching the inference worker | After the prediction file is written and the worker exits successfully | Python worker startup, imports, checkpoint load/model construction, input loading, feature preparation, batching, transfer, forward pass, CPU conversion, Python-list conversion, response construction, JSON serialization, file writing, worker shutdown | Startup of the small external timing observer; post-measurement annotation of the result with timing metadata | Not yet rerun |
| AC-OPF solve | Before `PowerModels.instantiate_model` | After `PowerModels.optimize_model!` returns | PowerModels/JuMP nonlinear-model construction, Ipopt initialization, optimization, solver-result construction | Docker/Julia startup, input JSON parsing, GridSFM export-object construction, `.pyg.json` serialization and writing | 10.727 s |

The implementation locations are:

- GridSFM forward and resident-request timers: [`benchmark/run_inference.py`](run_inference.py)
- GridSFM cold-request observer: [`benchmark/run_inference_e2e.py`](run_inference_e2e.py)
- AC-OPF timer: [`power_grid/US/topology_solver_pipeline/export_gridsfm_data.jl`](../power_grid/US/topology_solver_pipeline/export_gridsfm_data.jl)

## What is not currently measured

| Missing observation | Meaning |
|---|---|
| AC-OPF optimizer-only | Ipopt execution separated from PowerModels/JuMP model construction |
| AC-OPF complete request | Input parsing + model construction + optimization + result conversion |
| AC-OPF cold process | Docker/Julia startup + complete request + output writing |

## Synchronization

CPU operations are synchronous. CUDA operations are normally asynchronous, so GPU timing synchronizes the selected CUDA device immediately before starting and immediately after finishing the forward pass. Without these synchronizations, the host timer could stop before GPU kernels complete and under-report latency.

## What happens during each request

### GridSFM cold CLI request

1. Launch a fresh Python worker.
2. Import PyTorch and GridSFM modules.
3. Select CPU or GPU, load the checkpoint, construct the model, move it to the device, and switch to evaluation mode.
4. Read the `.pyg.json` request and construct the heterogeneous graph.
5. Build cycle, positional, stress, and other inference features, batch the graph, and transfer it to the selected device.
6. Run the model under inference mode. The forward path includes the DC prior, graph-network prediction, branch-flow calculation, and feasibility head.
7. Synchronize the device, detach outputs, move them to CPU, and convert them into response fields.
8. Serialize the prediction JSON, write it completely, and let the worker exit.

`run_inference_e2e.py` observes this whole sequence from outside the worker. `resident_request_seconds` covers only steps 4 through the CPU tensor return in step 7, because it assumes steps 1 through 3 have already happened.

### AC-OPF with PowerModels and Ipopt

1. Parse the network into the PowerModels data model.
2. Select the cold, DC-derived, or GridSFM-derived primal start values.
3. Instantiate an AC polar power-flow optimization model in JuMP: voltage magnitudes and angles, generator active/reactive power, branch-flow expressions, operating limits, AC balance equations, and the generation-cost objective.
4. Attach Ipopt and apply the fixed tolerances and warm-start experiment options.
5. Ipopt iteratively evaluates the nonlinear objective, constraints, derivatives, barrier system, and line search until it reaches a termination condition.
6. Extract the solved bus, generator, and branch state; then evaluate objective and feasibility checks and serialize the result.

The benchmark now queries JuMP's `barrier_iterations` after every Ipopt solve. Warm-start records contain `ac_ipopt_iterations`; the DC seed solve additionally contains `dc_presolve.ipopt_iterations`. Reference scenario exports store `metadata.ipopt_iterations`.

## Cache state

GridSFM preprocessing uses topology-related caches for cycle bases and numerical factorizations. The reported Ohio run used a new temporary cache directory, so its preprocessing portion was cold. The checkpoint was already loaded and is therefore warm/model-resident.

The forward measurement is unaffected by disk cache state because feature preparation completes before its timer starts. The request measurement is cache-sensitive and must be labelled `cold-cache` or `warm-cache` in future repeated runs.

## Which comparisons are defensible now?

The historical observations describe useful components, but their boundaries are not identical:

```text
GridSFM resident request
  = parse + prepare + transfer + forward + CPU tensor return

AC-OPF solve
  = construct optimization model + optimize
```

The provisional ratio `10.727 / 2.401 = 4.47x` must not yet be presented as a formal end-to-end speedup because AC input parsing is excluded while GridSFM input parsing and preparation are included, and only one run was measured.

For a customer-facing comparison, report two matched pairs:

1. **Resident compute:** GridSFM forward versus Ipopt optimizer-only.
2. **Warm request:** GridSFM parse-to-prediction versus AC-OPF parse-to-solved-result.
3. **Cold request:** externally observed process launch through a fully written response for both methods.

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
