# PGLib OOD: Phase 1–3 Gate Report

Purpose: prove that the raw PGLib MATPOWER input is consumable by the
existing PowerModels → GridSFM PyG exporter without changing the source or
using an OPF relaxation. This is a conversion proof only; no GridSFM
inference or perturbation was performed.

## Phase 1 — raw input audit

All five candidate `.m` files parse with `PowerModels.parse_file(...,
validate=true)`. All use a 100-MVA base and have cost curves that the current
exporter can represent exactly (polynomial model 2 with at most three retained
coefficients). The parser did emit its normal normalization messages (for
example, branch-orientation and generator-setpoint warnings, and removal of
zero-leading cost terms); the unabridged parser output is retained for audit.

| Case | SHA-256 | Buses | Gens | Loads | Branches | AC lines | Transformers | Exact cost export |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `case8387_pegase` | `b49ec5bb…a886ee06` | 8,387 | 1,865 | 4,669 | 14,561 | 12,511 | 2,050 | yes |
| `case13659_pegase` | `778ccf66…df573831` | 13,659 | 4,092 | 5,544 | 20,467 | 13,792 | 6,675 | yes |
| `case20758_epigrids` | `07d33b87…78634c70f` | 20,758 | 2,250 | 15,546 | 33,368 | 24,852 | 8,491 | yes |
| `case24464_goc` | `a508250f…543042d0` | 24,464 | 1,591 | 15,687 | 37,816 | 27,472 | 10,344 | yes |
| `case78484_epigrids` | `b9d8f673…5eb3a7f2` | 78,484 | 6,873 | 56,504 | 126,146 | 88,369 | 37,646 | yes |

Machine-readable audit: [pglib_ood_phase1_audit.json](pglib_ood_phase1_audit.json).  
Full parser output: [pglib_ood_phase1_parser.log](pglib_ood_phase1_parser.log).

## Phase 2–3 — `case8387_pegase` strict round trip

Procedure:

```text
unmodified source .m
  → strict cold AC-OPF / export GridSFM PyG
  → reconstruct from PyG plus the unmodified .m base
  → strict AC-OPF re-solve
  → compare objective, V, theta, Pg, Qg, and both-end branch flows
```

Declared pass conditions: both solves are successful, relative objective
difference ≤ `1e-3` (0.1%), and every state/flow maximum absolute difference
≤ `1e-3` pu (or radians for theta).

| Check | Source export | Re-solve from exported PyG | Result |
|---|---:|---:|---|
| Termination | `LOCALLY_SOLVED` | `LOCALLY_SOLVED` | pass |
| Objective | 2,771,392.2860647226 | 2,771,392.2860646993 | relative difference `8.40e-15` |
| Solve wall time | 50.315 s | 42.444 s | recorded, not a speed comparison |
| Worst V/theta/Pg/Qg/branch-flow difference | — | `7.71e-11` | pass |

The worst difference was generator reactive power; it is approximately eight
orders of magnitude below the declared `1e-3` gate. This validates the source
→ PyG representation for this first OOD case. It does not yet validate
GridSFM inference at this size.

Artifacts:

- [Exported PyG](../artifacts_ood/phase2_3/case8387_pegase.pyg.json)
- [Round-trip comparison JSON](../artifacts_ood/phase2_3/case8387_pegase_roundtrip.json)
- [Source solve log](../artifacts_ood/phase2_3/source_export.log)
- [PyG re-solve log](../artifacts_ood/phase2_3/pyg_resolve.log)
