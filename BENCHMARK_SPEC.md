# GridSFM Benchmark Specification

## Goal

Measure GridSFM inference accuracy against AC-OPF on newly perturbed scenarios, then test whether GridSFM-seeded AC-OPF is faster than DC-OPF-seeded and cold AC-OPF while preserving the same final acceptance criteria.

## Dataset cut

- Use the `16h` GridSFM US snapshots.
- Keep topologies with 500–4,661 buses.
- Select 5 topologies without replacement using a recorded random seed.
- Generate 5 accepted mixed-perturbation scenarios from each selected base grid.
- Evaluate 25 successfully solved scenarios in total.

## Pipeline

```text
gridsfm_data/16h/*_model.json
  -> filter eligible topologies
  -> seeded sample of 5 base grids
  -> make 2 fresh copies of each base grid
  -> apply mixed perturbations to each copy
  -> solve the transformed grid with PowerModels AC-OPF + Ipopt
  -> export transformed grid and AC-OPF solution as .pyg.json
  -> run GridSFM using only the transformed grid input
  -> compare GridSFM predictions with the AC-OPF solution
  -> write per-scenario and aggregate results
```

A scenario is included only when AC-OPF returns a complete solved result. Failed attempts are recorded and replaced with the next deterministic draw.

## Mixed perturbations

Apply the transformations in this fixed order to a fresh base-grid copy:

1. Load scaling and per-load jitter.
2. Optional generator outage.
3. Optional branch-rating derating.
4. Optional bus-voltage-limit tightening.
5. Generator-cost reshuffling.

Use an independent deterministic random stream for each transformation. The implementation will define its probabilities and ranges as named constants matching the paper's in-distribution envelope.

## Inference sample metadata

Store one small metadata object beside each generated `.pyg.json`:

```json
{
  "sample_id": "texas_mixed_0001",
  "base_grid": "gridsfm_data/16h/texas_model.json",
  "transformed_grid": "artifacts/scenarios/texas_mixed_0001.pyg.json",
  "seed": 42001,
  "perturbations": {
    "load": {},
    "generator_outage": {},
    "branch_derating": {},
    "voltage_tightening": {},
    "cost_shuffle": {}
  }
}
```

Each perturbation entry contains only the values used for that transformation, including `active: false` when it was not applied. Paths are repository-relative so they can be rendered as links by reports.

## Inference serving metadata

Record once per benchmark run:

```json
{
  "cpu_model": "...",
  "ram_gb": 0,
  "gpu_model": "...",
  "gpu_memory_gb": 0,
  "os": "...",
  "batch_size": 1
}
```

Use `null` for GPU fields on CPU-only runs.

## Metrics

Report per scenario and as aggregate mean, median, and maximum where applicable:

- Cost: absolute error, MAPE, MSE, and RMSE.
- `V`, circular `theta`, `Pg`, and `Qg`: MAE, MSE, and RMSE.
- Branch `P` and `Q` flows: MAE, MSE, and RMSE.
- Physics: active/reactive KCL residual, maximum thermal loading, overloaded-branch fraction, and maximum overload.
- Timing: AC-OPF solve time, GridSFM forward time, and GridSFM end-to-end time.

Angles are reported in radians and degrees. Power values retain the dataset's per-unit convention, with converted units added only when the base MVA is available.

## Outputs

```text
artifacts/
  scenarios/       # transformed .pyg.json files and sample metadata
  predictions/     # GridSFM predictions
  results.json     # complete machine-readable results
  results.csv      # one row per scenario
  summary.md       # compact human-readable comparison
  run_metadata.json
```

Generated artifacts remain untracked by default.

## Acceptance criteria

- The same transformed input is used by AC-OPF and GridSFM.
- GridSFM cannot read the AC-OPF solution during inference.
- Sampling and perturbations reproduce exactly from their recorded seeds.
- All 10 included scenarios have complete AC-OPF reference solutions.
- Metrics can be recomputed from saved scenarios and predictions.
- A CPU run and a later Colab GPU run use the same scenario files.

## Warm-start speed pilot

### Guiding evidence and boundary

Microsoft's GridSFM white paper is the guiding reference for the first arm, but not a complete implementation recipe:

- Section 6.3 says GridSFM's forward prediction seeds PowerModels/Ipopt.
- Table 5 names the model input as a partial `(P, θ)` payload: generator active dispatch `Pg` plus bus voltage angle `θ` (stored as `va` in PowerModels), not voltage magnitude `V`.
- Microsoft's accompanying article identifies these quantities as predicted active dispatch `Pg` and voltage angle `θ`/`Va`.
- Section 6.3 specifies `mu_init = 1.0` and `warm_start_bound_push = 1.0` for both approximate GridSFM and DC starts.
- Microsoft did not release the prediction-to-PowerModels driver or document mapping and omitted-variable handling. Our arm is therefore **Microsoft-described**, not an exact reproduction.

Primary source: [GridSFM white paper, Section 6.3 and Table 5](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/05/GridFM_white_paper.pdf#page=9).

### Hypothesis

GridSFM `Pg + θ` initialization reduces AC-OPF solve time and preparation-inclusive workflow time relative to both cold AC-OPF and DC-OPF `Pg + θ` initialization, while the final Ipopt result passes the same feasibility and objective checks as cold. Here `θ` maps to PowerModels `va`; it is not GridSFM voltage magnitude `V`.

Raw GridSFM prediction error is diagnostic only. Success is determined by final AC-OPF validity and runtime.

### Two predeclared scenarios

Use two already accepted scenarios from the 25-case `seed42_5x5` set:

| Scenario | Buses | Reason | Archived cold time |
|---|---:|---|---:|
| `colorado_mixed_0002` | 651 | First accepted prediction on the smallest selected topology | 6.591 s |
| `indiana_mixed_0002` | 1,271 | First accepted prediction on the largest selected topology | 11.459 s |

The choices are fixed before warm-start results are observed. Archived cold times remain provenance records; the paired runner also measures cold under the same process, hardware, solver, and repetition protocol as the warm arms.

### Parallel pipelines

```text
GridSFM path                           DC path
scenario                               scenario
  -> GridSFM forward pass                -> DC-OPF solve
  -> map predicted Pg + θ                -> map solved Pg + θ
  -> PowerModels/Ipopt AC-OPF             -> PowerModels/Ipopt AC-OPF
  -> strict final checks                  -> strict final checks
```

Both paths must use the identical transformed network, AC formulation, Ipopt version, tolerances, iteration limit, thread count, and approximate-start options. Only seed construction differs.

### Comparison arms

| Arm | AC initialization | Seed preparation |
|---|---|---|
| Matched cold | Existing native/flat policy, explicitly recorded | None |
| GridSFM warm | Predicted `Pg + θ` only (`pg` + PowerModels `va`) | GridSFM inference and deterministic row-to-ID mapping |
| DC warm | DC-solved `Pg + θ` only (`pg` + PowerModels `va`) | DC-OPF presolve and deterministic row-to-ID mapping |

For both approximate arms, test the Microsoft-stated `mu_init = 1.0` and `warm_start_bound_push = 1.0`. Do not inject `Vm`, `Qg`, constraint multipliers, or bound multipliers in this first pilot. Record the actual defaults used for omitted variables rather than guessing them.

### Timing contract

Record three views. The first is the primary fair comparison requested for this benchmark:

1. **Seeded total — primary:** seed preparation plus solver-reported seeded AC time. GridSFM is `GridSFM presolve + seeded AC solve`; DC is `DC-OPF presolve + seeded AC solve`; cold is its AC solve time because it has no presolve.
2. **Workflow E2E:** seed preparation + AC model construction + Ipopt solve. Runtime/checkpoint initialization is excluded from all steady-state workflows and reported separately if measured.
3. **Microsoft-published boundary — reference only:** GridSFM seeded AC solve without GridSFM inference, while DC includes DC preprocessing. Preserve this asymmetric view only to relate our results to Table 5; do not use it as the primary GridSFM-versus-DC claim.

`GridSFM presolve` starts before feature preparation for the already loaded scenario and ends when mapped `Pg + θ` seed values are ready for PowerModels. It includes feature preparation, model forward/device synchronization, output extraction, validation, and row-to-ID mapping. It excludes one-time Python startup and checkpoint loading, which are recorded separately.

`DC-OPF presolve` starts before construction of the DC problem and ends when mapped DC `Pg + θ` seed values are ready. It includes DC model construction, optimization, extraction, validation, and row-to-ID mapping.

Execute cold, DC-warm, and GridSFM-warm as paired runs in the same Julia process. Pin each solve to one CPU thread and set BLAS to one thread. The paper ran different grid cases 54-way in parallel, one core per case; this two-case pilot runs serially to avoid cross-case contention and records that deviation.

Persist these raw fields rather than reconstructing timing later:

| Field | Definition |
|---|---|
| `cold_ac_solve_seconds` | Solver-reported cold AC-OPF time |
| `gridsfm_presolve_seconds` | GridSFM feature preparation through mapped `Pg + θ` seed ready |
| `gridsfm_ac_solve_seconds` | Solver-reported GridSFM-seeded AC-OPF time |
| `gridsfm_seeded_total_seconds` | `gridsfm_presolve_seconds + gridsfm_ac_solve_seconds` |
| `dc_presolve_seconds` | Complete DC-OPF seed preparation through mapped `Pg + θ` seed ready |
| `dc_ac_solve_seconds` | Solver-reported DC-seeded AC-OPF time |
| `dc_seeded_total_seconds` | `dc_presolve_seconds + dc_ac_solve_seconds` |
| `*_ac_model_build_seconds` | AC PowerModels construction component, excluded from seeded total but included in E2E |
| `*_workflow_e2e_seconds` | Complete steady-state workflow boundary defined above |

Run one unmeasured warm-up followed by five measured repetitions per case and arm. Keep execution serial and pin the same thread settings. Report median, minimum, maximum, every individual observation, and:

```text
primary_gridsfm_speedup = cold_ac_solve_seconds / gridsfm_seeded_total_seconds
primary_dc_speedup      = cold_ac_solve_seconds / dc_seeded_total_seconds
primary_gridsfm_vs_dc   = dc_seeded_total_seconds / gridsfm_seeded_total_seconds

published_grid_boundary = gridsfm_ac_solve_seconds
published_dc_boundary   = dc_seeded_total_seconds

e2e_speedup_vs_cold   = cold_workflow_e2e_seconds / warm_workflow_e2e_seconds
e2e_gridsfm_vs_dc     = dc_workflow_e2e_seconds / gridsfm_workflow_e2e_seconds
```

Never add the archived reference solve to any arm's runtime.

### Correctness and acceptance

Every warm-started result must have:

- an accepted Ipopt termination status;
- final KCL, voltage, generator, angle, and thermal violations within the cold-solve tolerances;
- finite state and objective;
- objective gap versus matched cold explicitly reported, with `0.1%` as the initial equivalence threshold;
- fallback disabled during the comparison so a failed warm start remains a visible failure.

### Pilot outputs

```text
artifacts_seed42_5x5/warmstart_pilot/
  config.json
  <sample_id>_runs.json
  results.csv
  summary.md
```

The summary must lead with seeded-total results, then show workflow E2E and the Microsoft-published boundary separately. No speedup claim is accepted when the corresponding final AC-OPF result fails validation. The paper does not state how its ten scenario times were reduced to each Table 5 grid entry, so this pilot reports all five repetitions plus their median and does not claim to reproduce that undisclosed aggregation.
