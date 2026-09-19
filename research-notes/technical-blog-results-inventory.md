# AC-OPF vs GridSFM: blog sketchbook

Working notes only. Source for later blog, not publication copy.

## Story

1. Test GridSFM against AC-OPF on familiar, training-scale grids.
2. Repeat on much larger PGLib grids.
3. Compare prediction error, physical feasibility, runtime, and solver iterations.
4. Ask whether GridSFM helps as an AC-OPF warm start.

## Test sets

### Almost-ID

- 25 accepted scenarios: five perturbations each for Ohio, Colorado, Arizona, Pennsylvania, Indiana.
- 651–1,271 buses; within published 500–4,661-bus training range.
- Familiar US-derived topology family. Exact scenario membership in pretraining unknown; call this **training-scale** or **almost-ID**, not certified held-out ID.
- Mixed perturbations: load scaling, generator outages, branch derating, voltage tightening, cost reshuffling.
- AC-OPF and GridSFM receive same perturbed grid.

Realized perturbations: load factor 0.8014–1.2408; outages in 3/25; derating in 2/25; voltage tightening in 1/25; cost reshuffling in 25/25.

Sources: [`perturbations.yaml`](../benchmark/perturbations.yaml), [`results.json`](../artifacts_seed42_5x5/results.json), [`summary.md`](../artifacts_seed42_5x5/summary.md).

### PGLib OOD

- Unmodified PGLib base cases.
- Topology-and-scale OOD, not source-family OOD: PGLib/OPFData appeared in pretraining.
- Published training maximum: 4,661 buses.
- Four accepted cases: 6,470–20,758 buses, or 1.39–4.45 times training maximum.
- 24k case failed conversion gate; 78k case skipped round-trip re-solve. Keep both exploratory.

Acceptance gate: source and exported-PyG AC-OPF solves must succeed; objective relative difference and every state/flow maximum difference must be at most `1e-3`.

Sources: [`PGLib report`](../artifacts_ood/summary.md), [`candidate inventory`](pglib_ood_candidates.md).

## Main comparison

| Metric | Almost-ID mean | Accepted OOD mean | Accepted OOD median |
|---|---:|---:|---:|
| Bus range | 651–1,271 | 6,470–20,758 | — |
| Cost MAPE | 10.778% | 64.253% | 12.729% |
| Voltage MAE | 0.00608 p.u. | 0.04690 p.u. | 0.04949 p.u. |
| Angle MAE | 2.018° | 27.873° | 27.715° |
| Generator P MAE | 0.08624 p.u. | 0.99783 p.u. | 0.17825 p.u. |
| Generator Q MAE | 0.14666 p.u. | 0.45342 p.u. | 0.30661 p.u. |
| Branch P MAE | 0.05737 p.u. | 0.72418 p.u. | 0.61225 p.u. |
| Branch Q MAE | 0.25896 p.u. | 1.22152 p.u. | 1.32162 p.u. |
| Active KCL MAE | 0.05579 p.u. | 1.01724 p.u. | 0.98499 p.u. |
| Reactive KCL MAE | 0.74716 p.u. | 3.03477 p.u. | 3.19400 p.u. |
| Overloaded-branch fraction | 0.621% | 4.202% | 1.196% |
| Maximum loading, per-case average | 2.33 times | 47.86 times | 44.04 times |

OOD mean versus almost-ID: voltage error 7.7 times higher, angle error 13.8 times higher, active-KCL error 18.2 times higher. `case8387_pegase` drives cost mean; retain median.

Source: [`combined summary`](../COMBINED_BENCHMARK_SUMMARY.md).

## PGLib case results

| Case | Status | Buses | Scale | Cost MAPE | V MAE | Angle MAE | KCL-P MAE | Overload | Feas. probability | GridSFM request | Fresh AC solve | Ipopt iter. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `case6470_rte` | Accepted | 6,470 | 1.39 times | 2.604% | 0.02230 | 1.328° | 0.66549 | 2.02% | 0.935 | 14.91 s | 50.94 s | 89 |
| `case8387_pegase` | Accepted | 8,387 | 1.80 times | 228.950% | 0.05137 | 54.735° | 1.30450 | 14.22% | 0.000106 | 22.57 s | 44.69 s | 75 |
| `case13659_pegase` | Accepted | 13,659 | 2.93 times | 6.624% | 0.04761 | 41.811° | 2.04792 | 0.37% | 1.13e-17 | 25.47 s | 60.60 s | 64 |
| `case20758_epigrids` | Accepted | 20,758 | 4.45 times | 18.834% | 0.06632 | 13.620° | 0.05105 | 0.19% | 0.00784 | 45.55 s | 100.19 s | 48 |
| `case24464_goc` | Gate failed | 24,464 | 5.25 times | 34.905% | 0.10262 | 60.158° | 0.06555 | 0.23% | 1.41e-5 | 46.13 s | 1,240.62 s | 440 |
| `case78484_epigrids` | Gate skipped | 78,484 | 16.84 times | 15.386% | 0.07506 | 93.442° | 0.06614 | 0.43% | 0.0795 | 159.35 s | 1,231.05 s | 104 |

Last two rows exploratory. Exclude from accuracy aggregates.

Sources: [`OOD artifacts`](../artifacts_ood/), [`fresh AC runs`](../artifacts_ood/acopf_fresh_container/).

## Feasibility signal

- Almost-ID probabilities: 0.965–0.99998; mean 0.99596.
- Accepted OOD probabilities: 0.935, 0.000106, 1.13e-17, 0.00784.
- Feasibility head predicts whether input AC-OPF instance is solvable. It does **not** certify GridSFM output.
- All reference cases solved. Low OOD values are classifier false negatives here.
- Raw predictions still violate KCL and thermal limits. Report those separately.
- Interesting OOD signal, but not calibrated OOD detector.

Sources: [`run_inference.py`](../benchmark/run_inference.py), [`model evaluation`](../model/gridsfm/eval.py), [`prediction artifacts`](../artifacts_ood/).

## Timing and iterations

- Almost-ID fresh AC solve: mean 8.37 s; 31–64 iterations, mean 38.24.
- Accepted OOD fresh AC solve: mean 64.10 s; 48–89 iterations, mean 69.
- Accepted OOD GridSFM resident request: mean 27.13 s.
- AC solve and GridSFM request timers have different boundaries. Use as component latency, not formal end-to-end speedup.
- Cold GridSFM process timing missing.

Source: [`timing definitions`](../benchmark/TIMING.md).

## Warm-start result

| Arm | AC solver median | Total median | Result |
|---|---:|---:|---|
| Cold | 1.827 s | 1.827 s | 25/25 converged |
| GridSFM seed | 1.555 s | 2.936 s | Solver faster on 21/25; total faster on 0/25 |
| DC seed | 1.577 s | 1.825 s | Total faster on 13/25 |

GridSFM seed helped solver phase but inference overhead erased gain. Warm-start iteration columns empty; no iteration-reduction claim yet.

Source: [`warm-start summary`](../artifacts_seed42_5x5/warmstart_5x5/summary.md).

## Blog thesis

GridSFM gives fast approximate AC-OPF predictions and scales well beyond training graph sizes. Accuracy and physical feasibility weaken under topology-and-scale shift. Best role today: proposal or warm start, with AC-OPF remaining final authority.

## Caveats

- Say almost-ID/training-scale, not certified held-out ID.
- Say topology-and-scale OOD, not unseen data family.
- Raw GridSFM output is not feasible dispatch.
- Do not mix accepted 4-case OOD aggregate with 24k/78k exploratory probes.
- Graph size is proven out of range. Electrical-feature ranges have not been audited.
- Timing boundaries differ. No formal end-to-end speedup.
- No warm-start iteration comparison yet.

## Figures

1. Almost-ID versus accepted-OOD errors.
2. Runtime versus bus count; accepted and exploratory markers.
3. Feasibility probability plus KCL/overload metrics.
4. Cold versus GridSFM-seeded solver time and total time.

Existing runtime plot: [`bus_count_vs_runtime.png`](../benchmark/plots/bus_count_vs_runtime.png).

## Missing before final blog

- Train-versus-PGLib electrical-feature distribution audit.
- Matched cold end-to-end GridSFM timing.
- Repeated OOD timings with dispersion.
- Warm-start iteration counts.
- Feasibility-head calibration.
