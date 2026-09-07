# PGLib topology-and-scale OOD evaluation plan

## Scope

Evaluate the released GridSFM-Open checkpoint zero-shot on six locally available PGLib MATPOWER cases:

| Case | Buses | Intended role |
|---|---:|---|
| `case6470_rte` | 6,470 | paper-aligned zero-shot OOD reference |
| `case8387_pegase` | 8,387 | first large-scale step |
| `case13659_pegase` | 13,659 | medium-large scale step |
| `case20758_epigrids` | 20,758 | large scale step |
| `case24464_goc` | 24,464 | large scale step, distinct family |
| `case78484_epigrids` | 78,484 | extreme-scale capability boundary |

These are topology-and-scale OOD under the paper's stated GridSFM-Open training maximum of 4,661 buses. They are not source-family OOD: PGLib and OPFData were used during pretraining.

## Implementation status

| Phase | Status | Evidence |
|---|---|---|
| 1. Raw MATPOWER audit | Complete | All five cases parse; all have exact quadratic GridSFM-exportable generator costs and zero unsupported generators. See `research-notes/pglib_ood_phase1_audit.json`. |
| 2–3. Source solve → PyG export → PyG re-solve proof | Four accepted; 24,464 failed; 78,484 skipped | `case6470_rte`, `case8387_pegase`, `case13659_pegase`, and `case20758_epigrids` passed. `case24464_goc` produced two `LOCALLY_SOLVED` results with objective relative difference `1.90e-9`, but maximum reactive dispatch/flow difference `0.11607` exceeded the predeclared `1e-3` gate. `case78484_epigrids` solved and exported natively after a Docker exit-137 failure; six inactive type-4 bus rows were source-aligned and structural validation passed, but PyG re-solve was skipped at user direction. |
| 4. GridSFM resource qualification | Complete for all six cases | Batch-size-one CPU inference completed through 78,484 buses. The 24k and 78k executions are exploratory because their acceptance gates did not pass or were skipped. |
| 5. Base-case OOD smoke-test evaluation | Four accepted plus two exploratory reports | Accepted reports exist through `case20758_epigrids`. `case24464_goc` is labelled `EXPLORATORY_NOT_ACCEPTED`; `case78484_epigrids` is labelled `EXPLORATORY_CAPACITY_PROBE_NOT_ROUNDTRIP_VALIDATED`. Neither is included in accepted aggregates. |

## Finding: MATPOWER `.m` ingestion is available

The existing Julia exporter at `power_grid/US/topology_solver_pipeline/export_gridsfm_data.jl` states that it accepts MATPOWER `.m` files, including PGLib cases. It calls:

```julia
net = PowerModels.parse_file(input_path; import_all=false, validate=true)
```

A read-only smoke test successfully parsed the local `pglib_opf_case4837_goc.m` with PowerModels: 4,837 buses, 332 generators, 7,765 branches, base MVA 100.

This verifies parse compatibility only. It does not yet prove solver convergence, lossless schema conversion, or feasible GridSFM inference at the target sizes.

## Mandatory gates

### Gate 0 — Input and objective-semantics audit

For every target `.m` file, parse it through the pinned PowerModels environment and record:

- bus, generator, branch, transformer, load, and shunt counts;
- connected-component count and isolated buses;
- base MVA;
- generator cost model and polynomial degree distribution;
- branches with missing/zero thermal ratings;
- parser warnings and every normalization/mutation applied by PowerModels.

The first observed parser messages on `case4837_goc` remove leading **zero** polynomial coefficients and rescale the retained coefficients to the 100-MVA per-unit base. That is objective-preserving: for example, raw `0·P_MW² + 26.16·P_MW − 59.42` becomes `0·P_pu² + 2616·P_pu − 59.42`.

**Stop condition:** the GridSFM schema carries only quadratic generator costs (`cp2`, `cp1`, `cp0`). If an audit finds non-zero higher-order or piecewise costs that cannot be represented exactly, cost comparison is invalid unless the transformed objective is explicitly made the evaluation objective *before* AC-OPF is solved. Never solve with one objective and score GridSFM using another.

### Gate 1 — One-case end-to-end conversion proof

Start with `case8387_pegase`.

1. Parse its `.m` file.
2. Solve cold AC-OPF using the pinned PowerModels + IPOPT environment.
3. Export the solved graph to GridSFM `.pyg.json`.
4. Validate schema counts, finite values, maps, and rate coverage.
5. Re-solve the exported graph with `solve_pyg_json.jl`.
6. Compare termination status, objective, bus state, dispatch, and flows against the source solve within predeclared tolerances.

**Stop condition:** no OOD inference until source-to-PyG round-trip reproduces the same optimization problem and solution sufficiently closely.

### Gate 2 — GridSFM resource qualification

Run a single-scenario, batch-size-one forward pass on the same exported case with device telemetry.

Record CPU/GPU model, available/peak GPU memory, wall time, forward time, end-to-end request time, and failure mode.

Repeat only after success, in ascending size order:

`8,387 → 13,659 → 20,758 → 24,464 → 78,484 buses`.

**Critical risk:** GridSFM-Open is documented for roughly 4k buses, while the paper's other tier, GridSFM-Premier, is the tier advertised for tens of thousands of buses. The 78k case must be treated as a capacity-boundary probe, not a promised runnable benchmark. An out-of-memory or impractical-runtime result is a valid result if reported honestly.

### Gate 3 — Base-case AC-OPF ground truth (initial OOD pilot)

The local PGLib repository contains base MATPOWER cases, not OPFData labelled scenario splits. The initial OOD pilot therefore uses each selected case **as supplied**, with no new perturbations:

1. Cold-solve the unmodified source `.m` case with AC-OPF.
2. Treat the complete solver state — objective, bus voltage/angle, generator dispatch, and branch flows — as the ground truth for GridSFM comparison.
3. Compare the source solve to the re-solved exported PyG graph to validate conversion integrity.
4. If a source case is not `LOCALLY_SOLVED`, preserve its status/logs and raise it for review. Do not apply a relaxation, repair, or other quick fix.

This is a five-topology base-case OOD pilot, not a distributional OOD benchmark. Mixed perturbation scenarios are explicitly deferred to a later, separately labelled expansion.

### Gate 4 — Zero-shot evaluation and reporting

For each accepted scenario, run GridSFM-Open without fine-tuning, then report:

- system metadata: buses, generators, branches, transformer count, base MVA;
- objective: cost MAPE and signed cost bias;
- state errors: V, theta, Pg, Qg MAE/RMSE; add predeclared normalized errors later;
- flow errors: branch P/Q MAE/RMSE;
- physics: GridSFM and AC-OPF KCL P/Q MAE and maximum; thermal loading and overload count;
- timing: AC-OPF cold solve, GridSFM forward, GridSFM request boundary, and serving-device metadata;
- exact provenance: source `.m` SHA-256, transform metadata, seed, solver/image/package versions, checkpoint SHA-256.

Aggregate at two levels: per-case mean across its accepted scenarios, then an unweighted mean across cases that actually completed. Do not average raw generator or bus vectors across scenarios.

## Explicit result labels

- `case6470_rte`: paper-backed zero-shot OOD reference.
- Other selected PGLib cases: self-generated PGLib topology-and-scale OOD stress cases.
- 78k result: GridSFM-Open extreme-scale capacity probe; a non-run due to memory/runtime is reportable and must not be converted into an accuracy comparison.

## No-go conditions

Do not present numerical accuracy for a case if any of these holds:

- the parser/converter changes the cost objective without a declared normalized objective;
- source and PyG round-trip do not match within tolerance;
- AC-OPF reference is non-converged;
- tensor shapes/edge mappings are invalid;
- GridSFM cannot execute within the declared hardware envelope.
