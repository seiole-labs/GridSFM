# GridSFM-assisted AC-OPF warm-start strategy

## Purpose

Use GridSFM as a learned initializer for PowerModels/Ipopt while preserving AC-OPF as the authority for final feasibility and optimality. The intended production path is:

```text
grid → GridSFM prediction → validate/sanitize → AC power-flow projection
     → AC-OPF primal initialization → strict AC-OPF → final acceptance checks
```

GridSFM does not replace AC-OPF in this workflow. It proposes a starting point that may reduce Ipopt iterations, restoration work, or convergence failures.

## Two different meanings of “warm-up”

| Term | Meaning | Relevance here |
|---|---|---|
| Model/timing warm-up | Run GridSFM a few times to load weights, populate caches, and stabilize timing | Required for fair performance measurement, but it does not help AC-OPF convergence |
| Solver warm start | Initialize AC-OPF variables near a useful solution instead of using flat/default values | The feature being designed here |

GridSFM supplies only primal variables (`θ`, `V`, `Pg`, `Qg`). It does not supply Ipopt constraint multipliers or bound duals, so this is more precisely a **learned primal initialization**, not a complete Ipopt warm start.

## What PowerModels accepts

For the polar AC formulation, PowerModels reads these fields when it creates JuMP variables:

| GridSFM value | PowerModels field | Normal cold default |
|---|---|---|
| Bus angle `θ` | `va_start` | 0 |
| Voltage magnitude `V` | `vm_start` | 1 |
| Generator active dispatch `Pg` | `pg_start` | package/model dependent |
| Generator reactive dispatch `Qg` | `qg_start` | package/model dependent |

The existing [`solve_pyg_json.jl`](power_grid/US/topology_solver_pipeline/solve_pyg_json.jl) demonstrates the mechanics, but it reads `pyg["solution"]`, which is AC-OPF ground truth. A GridSFM warm-start benchmark must read the independent `*_prediction.json` file instead. Accidentally using `pyg["solution"]` would create an oracle benchmark.

## Selective lessons from arXiv:2606.08984

The later benchmark, *Not All Warm Starts Help*, studies exact same-instance oracle restarts and DC seeds rather than GridSFM predictions. Its results therefore define diagnostics and ceilings, not expected learned-model gains. The selective review is in [`research-notes/arxiv_2606_08984_warmstart_review.md`](research-notes/arxiv_2606_08984_warmstart_review.md).

Adopt now:

- make full `Pg + Qg + Vm + Va` **primal-only** initialization the primary new candidate, because GridSFM already emits all four blocks;
- let Ipopt initialize its own constraint and bound multipliers; do not fabricate or learn partial duals;
- keep interior clipping and angle-reference alignment;
- use one predeclared policy across cases and report convergence/fallback rate beside speed;
- separate optimizer-only, build-plus-solve, and true end-to-end timing.

Test, rather than assume:

- Microsoft-described `Pg + θ` payload (`pg` + PowerModels `va`) with its published barrier settings;
- full raw and full sanitized GridSFM starts;
- `Vm + Va` as a block ablation;
- AC-PF projection, with its preparation cost included;
- barrier settings as an explicit experimental factor.

Do not adopt:

- partial or learned dual initialization;
- case-wise selection of the fastest arm as if it were a deployable policy;
- the newer paper's 47.6–50.3% oracle gains as a GridSFM forecast;
- its exact-solution, narrow-bound-push configuration for noisy predictions without testing.

## The warm-start module

Place one deep module at the seam between prediction and solver initialization. Callers should not need to understand row mappings, electrical islands, angle references, bounds, feasibility scoring, or fallback rules.

Proposed interface:

```text
prepare_warm_start(grid, prediction, strategy) → WarmStartDecision
```

`strategy` is one of:

- `raw`: direct prediction, for research comparison only;
- `sanitized`: mapped, anchored, clipped, and balanced;
- `pf_projected`: sanitized prediction followed by AC power-flow projection;
- `dc`: existing DC initializer;
- `cold`: normal flat/default initialization.

`WarmStartDecision` should contain:

```text
accepted          true/false
requested_mode    raw/sanitized/pf_projected/dc/cold
applied_mode      the mode actually used after fallback
start_values      va_start/vm_start/pg_start/qg_start by source ID
diagnostics       shape, bounds, island balance, KCL, thermal, PF status
fallback_reason   null or a stable machine-readable reason
timing_seconds    inference, sanitation, projection, total preparation
```

This interface provides leverage: every benchmark and future serving path gets the same mapping, checks, projection, and fallback behaviour. It also concentrates tests at one seam.

## Safety pipeline

### 1. Structural validation

Reject the prediction before touching the solver when any of these holds:

- bus or generator row counts do not match the ID maps;
- IDs are duplicated, missing, or mapped in a different order;
- any predicted value is NaN or infinite;
- topology/checkpoint metadata do not match the scenario;
- an energized island has demand but no active generator or slack capability.

The fallback should be DC, then cold if DC initialization fails.

### 2. Angle-reference alignment

Voltage angle has an arbitrary additive reference. For every connected island:

1. find its reference/slack bus;
2. subtract the predicted slack angle from all predicted angles in that island;
3. add the source model's required reference angle, normally zero.

Without this step, a prediction can have physically reasonable angle differences but look far from the solver's coordinate system.

### 3. Bound sanitation

- Clip `V` to an interior margin inside `[Vmin, Vmax]`.
- Clip `Pg` and `Qg` to an interior margin inside generator limits.
- Preserve offline generator status; never revive a tripped generator from a prediction.
- Replace non-finite or unmapped values with the corresponding DC/cold value rather than silently using zero.

An interior margin is preferable to placing an interior-point solver exactly on many bounds.

### 4. Per-island active-power balancing

For each island, compare predicted generation with demand plus an estimated loss allowance. Redistribute the mismatch across generators with remaining headroom, using a deterministic rule such as proportional headroom or participation factors.

If the mismatch cannot be absorbed within generator limits, reject the GridSFM start and fall back. Do not alter loads or limits to make the start fit.

### 5. AC power-flow projection

The recommended production strategy is `pf_projected`:

- use sanitized GridSFM `Pg` and generator voltage setpoints as targets;
- let slack generation absorb the remaining active-power mismatch;
- solve AC power flow for consistent `θ`, `V`, and `Qg`;
- derive branch flows from the projected state;
- pass the projected primal state to AC-OPF.

The projection is valuable because the current raw predictions have substantial KCL and thermal violations. A successful AC power flow converts global GridSFM guidance into an electrically consistent initializer before Ipopt begins optimization.

If projection fails, production should fall back to DC/cold. Raw or merely sanitized GridSFM should remain benchmark arms, not automatic production fallbacks.

### 6. Final AC-OPF acceptance

Warm starting never weakens final acceptance. Require:

- accepted solver termination status;
- finite objective and state;
- the same KCL, voltage, generator, angle, and thermal tolerances as the cold solve;
- unchanged network, limits, loads, costs, and objective;
- explicit reporting when the warm and cold starts converge to different local objectives.

If the warm-started solve fails, retry once using the standard DC/cold path and record the fallback. Never return the raw GridSFM prediction as an AC-OPF solution.

## Why high GridSFM error may still help

A starting point does not need to be operationally accurate. It only needs to place Ipopt in a better region than a flat or DC start. Approximate generator commitment patterns, voltage profiles, and angle differences can still reduce nonlinear search work after sanitation and projection.

The benefit is not guaranteed:

- high KCL violations can increase Ipopt restoration work;
- bound-heavy predictions can slow an interior-point method;
- a poor prediction can enter a worse local basin;
- model inference and projection overhead can exceed solver savings;
- AC-OPF model-construction time is unaffected by the starting point.

This is why raw, sanitized, projected, DC, and cold arms must be compared independently.

## Benchmark design

### Comparison arms

| Arm | Purpose | Production candidate? |
|---|---|---|
| Cold/flat | Current baseline | Yes, fallback |
| DC start | Strong conventional baseline | Yes, fallback |
| Microsoft-described GridSFM `Pg + θ` | Tests the explicitly documented payload (`pg` + PowerModels `va`) and barrier settings; not an exact reproduction because Microsoft did not release the driver | Candidate after local validation |
| Raw GridSFM | Measures whether unmodified predictions help or hurt | No |
| Sanitized GridSFM | Isolates the value of mapping, anchoring, clipping, and balance | Possibly |
| GridSFM + AC-PF projection | Tests the recommended strategy | Yes |
| GridSFM `Vm + Va` only | Paper-informed block ablation | No, diagnostic |
| Oracle AC-OPF start | Verifies that the solver/timing instrumentation responds to a near-solution start | Diagnostic only |

### Metrics to capture

- PowerModels construction time;
- Ipopt optimizer-only time;
- Ipopt iteration count;
- restoration-phase iteration count/time when available;
- GridSFM inference time;
- sanitation and AC-PF projection time;
- warm preparation plus optimizer time;
- complete cold-process and warm-process wall time;
- convergence and fallback rate;
- final objective and difference from the cold result;
- final KCL, bounds, voltage, thermal, and angle-limit violations;
- peak memory.

Use identical solver tolerances, limits, hardware, and thread counts. Run repeated trials after explicit warm-ups and report median, p5, and p95. Keep training-scale, accepted OOD, and exploratory scale probes separate.

## Indicative break-even points

Using the current model-resident GridSFM request and AC construction-plus-solve measurements, GridSFM preparation must save at least approximately:

| Group | GridSFM request / AC recorded time | Minimum indicative reduction before projection overhead |
|---|---:|---:|
| Training-scale mean | 1.24 / 8.58 s | 14.5% |
| Accepted OOD mean | 27.13 / 106.69 s | 25.4% |
| 24k | 46.13 / 1,914.41 s | 2.4% |
| 78k | 159.35 / 1,924.45 s | 8.3% |

These are not formal speedup thresholds because the current timers are unmatched and AC model-construction time cannot be reduced by initialization. The benchmark must add optimizer-only and matched process/request timers before making a speedup claim.

## Implementation sequence

1. Add matched timing and Ipopt-iteration instrumentation without changing solver behaviour.
2. Implement prediction-to-source-ID mapping and diagnostics only; do not solve yet.
3. Add deterministic angle alignment, clipping, and per-island balancing with unit tests.
4. Benchmark the Microsoft-described `Pg + θ` arm, full raw, and full sanitized starts against cold/DC on the small smoke case.
5. Add AC power-flow projection and explicit fallback reasons.
6. Run all arms on the 25 training-scale scenarios.
7. Run accepted OOD cases in ascending size order.
8. Attempt 24k and 78k only after the smaller comparison is stable.
9. Publish accepted comparisons only when final AC-OPF feasibility and objective checks pass.

The first implementation milestone should stop after steps 1–4. It will reveal mapping, timing, and solver-initialization problems cheaply before adding projection or running the large cases.
