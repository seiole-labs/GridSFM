# GridSFM Pilot Benchmark — Phased Implementation

This plan implements [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md) as a vertical slice before expanding to the complete 10-scenario pilot.

## Current status

| Phase | Status | Evidence / remaining work |
|---|---|---|
| 0 — Environment | Complete | Python: 32 passed, 6 skipped. Existing Julia/PowerModels integration pipeline passed all six round trips. GridSFM v1.1 checkpoint is installed. |
| 1 — Select topology | Complete | Persistent 30-topology pool, deterministic selector tests pass, and seed 42 selects Ohio (784 buses). |
| 2.1 — Perturbation contract | Complete | Constants are in `benchmark/perturbations.yaml`; YAML contract tests pass. |
| 2.2 — Load perturbation | Complete | Deterministic transformation tests pass 13/13. |
| 2.3–2.4 — Generator outage and branch derating | Complete | Generator tests pass 9/9 and derating tests pass 16/16. |
| 2.5–2.7 — Voltage, costs, and mixed chain | Complete | Voltage 21/21, cost 15/15, and orchestration 9/9 tests pass. |
| 2.8 — Persist real transformed sample | Complete | Reproducible Ohio transformed grid and compact metadata are saved under `artifacts/pilot/`. |
| 3 — Solve and export | Complete | Ohio is `LOCALLY_SOLVED`; the 784-bus `.pyg.json` passes independent structural, finite-value, and GridSFM-loader validation. |
| 4 — GridSFM inference | Complete | Ohio prediction is finite and dimensionally aligned: 784 buses, 96 generators, and 1,513 branches. CPU timings and minimal serving metadata are saved. |
| 5 — Compare against AC-OPF | Complete | Ohio metrics recompute byte-for-byte from the saved scenario and prediction; focused ordering, circular-angle, unit, thermal, and determinism tests pass. |
| 6–8 | Complete | The accepted `seed42_5x5` benchmark contains 25 scenarios across five topologies. |
| 9 — Warm-start contract | Complete | Added the Microsoft evidence boundary, two fixed pilot cases, symmetric timing scopes, and acceptance rules to `BENCHMARK_SPEC.md`. |
| 10 — Shared runner | Complete | One topology-agnostic paired runner executes matched cold, GridSFM-warm, and DC-warm AC-OPF with explicit timing boundaries. |
| 11A — GridSFM warm path | Complete | Raw `Pg + θ` is mapped by source ID; `Vm` and `Qg` are not injected. |
| 11B — DC warm path | Complete | DC-OPF `Pg + θ` uses the same AC/barrier path and records complete DC presolve time. |
| 12 — Full 5×5 execution | Complete | All five grids × five accepted mixed-perturbation realizations ran one warm-up plus five measured repetitions for all three methods (375 measured method runs). |
| 13 — Verification/report | Complete | All runs converged; objective, every state channel, every branch-flow channel, solver-only time, seeded total, and workflow E2E are saved and summarized. |

Phase 2 is deliberately review-gated into: contract, load, generator outage, branch derating, voltage tightening, cost shuffle, mixed orchestration, and real-sample persistence.

## Phase 0 — Environment readiness

Set up and verify:

- Julia, PowerModels, and Ipopt for AC-OPF and `.pyg.json` export.
- Python, PyTorch, PyTorch Geometric, and the local `gridsfm` package.
- GridSFM-Open v1.1 checkpoint.

**Gate:** the existing Julia component integration test and Python smoke tests pass.

## Phase 1 — Select one base topology

- Use the checked-in one-time eligibility manifest at `benchmark/topology_pool.json`.
- The manifest contains the discovered `16h` grids with 500–4,661 buses; normal benchmark runs do not rediscover eligibility.
- Select one topology using the benchmark seed.
- Save its repository-relative path and basic dimensions.

**Output:** `artifacts/pilot/selection.json`.

**Gate:** rerunning selection with the same seed returns the same topology.

## Phase 2 — Generate one mixed perturbation

- Keep all perturbation constants in `benchmark/perturbations.yaml`.
- Copy the selected base model in memory.
- Apply the mixed perturbation chain with deterministic per-transform seeds.
- Record only the base-grid path, transformed-grid path, seed, and values used by each perturbation.
- Save the transformed PowerModels JSON before solving it.

**Outputs:**

- `artifacts/pilot/<sample_id>_transformed.json`
- `artifacts/pilot/<sample_id>_metadata.json`

**Gate:** the base file is unchanged and repeated generation is byte-for-byte reproducible.

## Phase 3 — Solve and export the sample

- Solve the transformed model with PowerModels AC-OPF and Ipopt.
- Export the transformed input and AC-OPF reference solution to GridSFM `.pyg.json`.
- Keep this first slice only if the solve returns a complete solved result.

**Output:** `artifacts/pilot/<sample_id>.pyg.json`.

**Gate:** reload the file, validate required node/edge shapes, and confirm that the AC-OPF solution is complete.

## Phase 4 — Run GridSFM inference

- Load the `.pyg.json` grid input without exposing its solution to inference.
- Prepare GridSFM features and run one forward pass.
- Capture the minimal serving metadata and forward/end-to-end wall-clock times.
- Save predictions independently of the reference solution.

**Outputs:**

- `artifacts/pilot/<sample_id>_prediction.json`
- `artifacts/pilot/run_metadata.json`

**Gate:** prediction dimensions match the sample's buses, generators, and branches, and all values are finite.

## Phase 5 — Compare against AC-OPF

- Compute the metrics required by the benchmark spec.
- Calculate physics metrics directly from the GridSFM prediction and transformed limits.
- Produce a compact machine-readable result and human-readable report.

**Outputs:**

- `artifacts/pilot/result.json`
- `artifacts/pilot/summary.md`

**Gate:** independently recomputing metrics from the saved scenario and prediction reproduces the result.

## Phase 6 — Complete one-topology slice

- Generate a second independent mixed perturbation from the same untouched base topology.
- Run Phases 3–5 for it.
- Add aggregation across the two scenarios.

**Gate:** both scenarios reproduce from their seeds and the aggregate is computed only from saved per-scenario results.

## Phase 7 — Expand to five topologies

- Change selection from one to five eligible topologies without replacement.
- Generate two independent mixed scenarios per topology.
- Reuse the same generation, solve, inference, and evaluation paths without topology-specific logic.
- Continue deterministic draws when a scenario does not produce a complete AC-OPF solution.

**Outputs:** the complete `artifacts/` layout defined in the benchmark spec.

**Gate:** 10 solved scenarios are present, with two belonging to each selected topology.

## Phase 8 — Full-pipeline verification

- Run the complete benchmark from an empty artifacts directory.
- Repeat it and confirm deterministic scenario generation and metrics.
- Run the saved scenarios on CPU and later on Colab GPU without regenerating ground truth.
- Confirm that prediction accuracy is device-consistent within numerical tolerance.

**Gate:** the final CSV, JSON, and Markdown summaries agree and all links resolve to saved artifacts.

## Implementation rule

Do not begin the next phase until the current gate passes. Keep the one-sample path as the permanent smoke test for later changes to perturbation, conversion, inference, or metrics code.

## Warm-start extension

This extension implements the warm-start section of [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md). Tracks 11A and 11B may be developed in parallel only after the shared runner contract passes.

### Phase 10 — Shared paired runner

- Load one accepted transformed scenario without reading its stored solution as a seed.
- Construct one unchanged AC-OPF formulation for cold, GridSFM, and DC arms.
- Expose a seed interface containing only `Pg` and `θ` keyed by source component ID; `θ` is written to PowerModels `va`, never to voltage magnitude `vm`.
- Record seed preparation, mapping, model construction, Ipopt solve, workflow E2E, iterations, status, objective, and final violations.
- Read and persist the solver-reported `solve_time`; do not substitute the existing construction-plus-optimization wall timer for the Microsoft-published-boundary column.
- Execute the paired arms in one Julia process with one solver thread and one BLAS thread.
- Add configuration/provenance for solver versions, options, hardware, and threads.

**Gate:** a no-op/native seed reproduces the matched cold objective and constraints, and timing fields have non-overlapping documented boundaries.

### Phase 11A — GridSFM warm path

- Read the independent `*_prediction.json`; never read `pyg["solution"]` for initialization.
- Validate finite values and exact bus/generator row counts.
- Map predicted `Pg` and `θ` to PowerModels `pg` and `va` IDs without adding `V`/`vm` or `Qg`.
- Time GridSFM feature preparation, forward pass, extraction, validation, and mapping as `gridsfm_presolve_seconds`.
- Set `mu_init = 1.0` and `warm_start_bound_push = 1.0` on the approximate AC solve.
- Keep any sanitation or projection out of this first Microsoft-described arm.

**Gate:** a fixture proves the AC model receives exactly the expected `Pg`/`θ` values in `pg`/`va`, never confuses `θ` with `V`/`vm`, and receives no ground-truth or other predicted blocks.

Smoke evidence: [`artifacts_seed42_5x5/warmstart_pilot/summary.md`](artifacts_seed42_5x5/warmstart_pilot/summary.md). The reusable path-based runner is [`run_gridsfm_warmstart.jl`](power_grid/US/topology_solver_pipeline/run_gridsfm_warmstart.jl); it uses scenario ID maps rather than topology-specific assumptions, so the same interface accepts training-like and OOD artifacts in the same schema.

### Phase 11B — DC warm path

- Solve DC-OPF on the identical transformed scenario.
- Extract and map only DC `Pg` and `θ` through the same seed interface.
- Apply the identical AC formulation and approximate-start Ipopt options as Track 11A.
- Time complete DC model construction, solve, extraction, validation, and mapping as `dc_presolve_seconds`.
- Report `gridsfm_presolve_seconds + gridsfm_ac_solve_seconds` and `dc_presolve_seconds + dc_ac_solve_seconds` as the symmetric primary comparison.

**Gate:** DC seed values satisfy the DC result and the AC runner differs from Track 11A only in seed producer.

### Phase 12 — Execute the full 5×5 benchmark

- Run five accepted mixed-perturbation realizations for each of Arizona, Colorado, Indiana, Ohio, and Pennsylvania.
- Perform one unmeasured warm-up, then five serial measured repetitions for cold, GridSFM warm, and DC warm.
- Preserve every raw observation; do not replace failures or select the fastest repetition.

**Gate:** complete. All 25 cases and 375 measured method runs are present under `artifacts_seed42_5x5/warmstart_5x5/` with complete timing and provenance.

### Phase 13 — Verify and report

- Apply identical strict final checks to every converged result.
- Compare objectives and final violations with the matched cold result.
- Compute symmetric seeded-total speedup, complete workflow-E2E speedup, and the Microsoft-published timing boundary independently.
- Report GridSFM-versus-DC ratios, iteration counts, convergence, and failures.
- Keep archived cold timings visible but use matched cold measurements for the primary ratios.

**Gate:** complete. Every reported speedup corresponds to a converged final AC-OPF result and can be recomputed from the saved run JSON/CSV. The OOD extension remains a separate next phase.
