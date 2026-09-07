# Selective review: arXiv 2606.08984 for GridSFM-assisted AC-OPF warm starts

Source reviewed: Babak Taheri and Daniel K. Molzahn, *Not All Warm Starts Help: Benchmarking Primal-Dual Initializations for ACOPF Algorithms*, arXiv:2606.08984v1 (2026), [HTML paper](https://arxiv.org/html/2606.08984v1). This note treats the paper as a useful benchmark study, not as validation of learned or out-of-distribution warm starts.

## Verdict

Adopt the paper's **full-primal, primal-only** emphasis and its benchmarking discipline. GridSFM already predicts all four blocks the paper studies—`Pg`, `Qg`, `Vm`, and `Va`—so the most defensible first experiment is to initialize all four while letting Ipopt initialize its own multipliers. Do not add learned duals now.

Keep the repository's sanitation and optional AC-power-flow projection as separate experimental arms. The paper did not test learned predictions, AC projection, or feasibility repair, so it neither validates nor refutes that part of [`WARMSTART_STRATEGY.md`](../WARMSTART_STRATEGY.md).

## What the paper actually tested

The study uses Ipopt on 19 PGLib-OPF cases from 5 to 30,000 buses. Its oracle experiments solve an AC-OPF, then initialize a **second solve of the same instance** with exact converged values for subsets of `Pg`, `Qg`, `Vm`, and `Va`. It separately tests primal-only starts, several primal-plus-dual coverage patterns, dual-only controls, and practical DCOPF seeds containing `Pg` and/or `Va` ([Sections III–IV](https://arxiv.org/html/2606.08984v1#S3)). It does not train or evaluate an ML model.

The central empirical results relevant here are:

- Exact full-vector primal-only initialization had a `+14.3%` median solve-time speedup as a fixed policy; primal-only starts were much more robust than partial primal-plus-dual starts ([Section VI](https://arxiv.org/html/2606.08984v1#S6)).
- If only part of the primal vector is available, combinations containing both voltage blocks were the safest oracle candidates. This is a target-priority result, not evidence that inaccurate voltage predictions will help ([Section VI-C](https://arxiv.org/html/2606.08984v1#S6.SS3)).
- Partial or inconsistent dual coverage was often harmful. Across all subsets, block-matched primal-plus-dual starts converged in `70.4%` of runs with a `-31.1%` median speedup, versus `98.5%` convergence and `+5.8%` median speedup for primal-only starts ([Section VII](https://arxiv.org/html/2606.08984v1#S7)).
- DCOPF seeding sometimes reduced the AC solve time, but its benefit was not statistically significant after adding DCOPF presolve cost to one-shot end-to-end time (`p = 0.4171`) ([Section VIII](https://arxiv.org/html/2606.08984v1#S8)).

## Original GridSFM method: documented facts only

The [GridSFM white paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/05/GridFM_white_paper.pdf) says GridSFM's forward prediction seeds PowerModels/Ipopt; Table 5 names the partial payload `(P, θ)`. It also specifies `mu_init = 1.0` and `warm_start_bound_push = 1.0`. Microsoft's [accompanying article](https://www.microsoft.com/en-us/research/blog/gridsfm-a-new-small-foundation-model-for-the-electric-grid/) identifies those quantities as predicted voltage angles and active dispatch. Microsoft has not published the model-to-solver driver, so omitted-variable handling and mapping details are unknown.

## Transfer decisions

| Decision | Technique | Application here |
|---|---|---|
| **Adopt now** | Initialize the complete primal vector | Make `Pg + Qg + Vm + Va` the primary GridSFM arm. GridSFM already emits all four, so dropping blocks saves no inference cost and can create a less coherent start. |
| **Adopt now** | Let Ipopt initialize duals | Use GridSFM as a learned primal initialization only. Do not predict or inject constraint or bound multipliers. The paper's strongest warning is against inconsistent partial primal-dual states. |
| **Adopt now** | Keep starts inside bounds | Retain reference-angle alignment, finite-value checks, and interior clipping from `WARMSTART_STRATEGY.md`. The paper explains that values at or near bounds interact poorly with Ipopt's barrier and bound-push logic ([Appendix A](https://arxiv.org/html/2606.08984v1#A1)). |
| **Adopt now** | Separate timing scopes | Record optimizer-only, model-build-plus-solve, and true E2E time. GridSFM E2E must include inference, mapping, sanitation, optional projection, model construction, and AC-OPF; it must not be compared with Ipopt-only time. |
| **Adopt now** | Report failures with speed | Report convergence/fallback rates beside median time and iterations. Do not choose a method by the median of successful runs alone; the paper demonstrates survivor bias when hard cases fail ([Section IV-C](https://arxiv.org/html/2606.08984v1#S4.SS3)). |
| **Adopt now** | Fixed-policy paired comparisons | Apply one predeclared strategy across cases and compute per-case paired speedups. Treat case-wise-best results only as diagnostic upper bounds. |
| **Defer / test separately** | `Vm + Va` partial start | Include at most as an ablation if full GridSFM initialization performs poorly. The paper's voltage priority is based on exact oracle values, not noisy OOD predictions. |
| **Defer / test separately** | AC-power-flow projection | Keep `sanitized` and `pf_projected` as distinct arms and include projection cost. The paper explicitly used raw DC outputs without AC lift or projection, so it supplies no evidence that projection will yield a net speedup ([Discussion](https://arxiv.org/html/2606.08984v1#S10)). |
| **Defer** | Full primal-plus-dual warm start | Revisit only if a future model can produce a nearly complete, mutually consistent primal-dual-barrier state and the repository can obtain reliable dual training labels. This is a materially larger research project. |
| **Reject** | Partial/block-matched learned duals | They add training targets and solver coupling while the paper finds poor reliability. GridSFM currently has no dual outputs, so there is no near-term benefit. |
| **Reject** | Copy the paper's `1e-6` warm-start pushes blindly | Those settings were used with exact same-instance oracle primal-dual values. They are not established as safe for inaccurate GridSFM predictions. Tune only in an explicit solver-option experiment. |
| **Reject** | Use the paper's `47.6–50.3%` figures as expected gains | Those are oracle/case-wise ceilings involving exact previous solutions and, for the largest figures, dual information. They are not learned-start forecasts. |
| **Reject** | Generalize the 30k result to this 78k case | The paper used a different modeling stack, hardware, formulation implementation, and maximum network size. Its timing ranks may not transfer ([Limitations](https://arxiv.org/html/2606.08984v1#S10)). |

## Feasibility and guarantees

The paper does not repair warm starts before Ipopt. It gives local bounds showing that omitted or displaced primal blocks can introduce feasibility and stationarity residuals, and that reusing mismatched bound multipliers can disturb complementarity and barrier centering ([Section III-E](https://arxiv.org/html/2606.08984v1#S3.SS5), [Appendix A](https://arxiv.org/html/2606.08984v1#A1)). These arguments explain possible harm; they do **not** guarantee convergence, speedup, feasibility recovery, or convergence to the same AC-OPF local optimum.

For this repository, high GridSFM KCL violations therefore remain a real risk, but they are not proof that a primal start is useless. The correct controls are to record the initial KCL/bound/thermal residuals, Ipopt restoration behavior, final feasibility, and final objective difference. AC-OPF supplies final accuracy only when it terminates acceptably and passes the repository's strict feasibility checks. Matching objective values alone does not prove an identical state or KKT point; the paper makes the same limitation explicit ([Section IV-E](https://arxiv.org/html/2606.08984v1#S4.SS5)).

## Training and data implications

The paper adds no training recipe, loss, sample-complexity result, OOD result, or learned feasibility guarantee. Its only directly usable ML guidance is **which outputs to prioritize** under an exact-prediction ceiling. No retraining is required for the first experiment because GridSFM already predicts the full primal vector.

If later training is changed, prioritize joint quality and coherence of all four primal blocks rather than adding dual heads. Feasibility-aware losses or projection-aware training may be sensible hypotheses, but they do not come from this paper and need their own evidence.

## Local integration observations

- [`solve_pyg_json.jl`](../power_grid/US/topology_solver_pipeline/solve_pyg_json.jl) already demonstrates mapping `Va/Vm/Pg/Qg` into PowerModels starts, but reads `pyg["solution"]`, i.e. oracle AC-OPF ground truth. A real benchmark must accept the independent GridSFM prediction artifact through an explicit interface.
- That direct strict-solve path supplies primal values without setting `warm_start_init_point=yes`, which is the appropriate first arm when no duals are supplied. By contrast, [`run_opf_relaxation.jl`](../power_grid/US/topology_solver_pipeline/run_opf_relaxation.jl) sets `warm_start_init_point=yes` globally; do not inherit that option uncritically for the learned-primal benchmark.
- The paper's baseline is native case-file initialization, whereas this repository also has an explicit flat-start path. Benchmark both the currently deployed baseline and a clearly named flat baseline; baseline choice can change the apparent benefit.
- Add a baseline-versus-baseline second solve before using the oracle arm. The paper rebuilt its model but ran both oracle solves in one process and acknowledges an unmeasured second-solve/cache advantage ([Section IV-E](https://arxiv.org/html/2606.08984v1#S4.SS5)).

## Minimal comparison to implement first

Use one small smoke case, then the existing training-scale and OOD tiers:

1. Native/current solver start.
2. Explicit flat start.
3. Existing DC start, with DC cost included in E2E.
4. Microsoft-described GridSFM `Pg + θ` start (`pg` + PowerModels `va`) with its wide-barrier settings.
5. Full raw GridSFM primal start.
6. Full sanitized GridSFM primal start.
7. Full sanitized + AC-PF-projected GridSFM start.
8. Exact full-primal oracle and baseline-to-baseline re-solve, both diagnostic only.

Hold formulation, Ipopt tolerances, iteration limits, linear solver, hardware, and thread count fixed. Unlike the paper's single-run timing, use repeated trials after declared process/model warm-up and report median plus dispersion. Capture inference, sanitation, projection, model build, Ipopt-only time, iterations/restoration, total E2E, convergence/fallback, final objective delta, and strict final violations. This preserves the paper's strongest controls while addressing its stated timing and caching limitations.

## Bottom line

The selectively transferable idea is not “use primal-dual warm starts.” It is: **give Ipopt a coherent full primal proposal, keep dual initialization internal, measure the entire workflow, and treat reliability as a first-class outcome**. That strengthens the existing strategy. The repository's AC-PF projection remains a plausible safety mechanism, but it is an unvalidated hypothesis whose cost and benefit must be measured rather than assumed.
