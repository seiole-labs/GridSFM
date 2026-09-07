# PGLib topology-and-scale OOD candidate inventory

## Eligibility rule

GridSFM-Open's paper states that it was trained on grids of **up to 4,661 buses**. Therefore every full, unmodified PGLib topology below exceeds that stated training-topology cap and is a topology-and-scale OOD candidate for the released Open checkpoint.

This does **not** mean PGLib or OPFData is an unseen source family: both were training sources. It means the complete graph was outside the stated training size range.

The inventory was read from the MATPOWER cases in `/home/abhey/seiole-labs/pglib-opf-data` on 2026-08-30. `Branches` is the number of MATPOWER branch rows; it includes lines and transformer branches. `x cap` is buses / 4,661.

| Candidate | Buses | Gens | Branches | Branches / bus | x cap | Practical first-run tier |
|---|---:|---:|---:|---:|---:|---|
| `case4837_goc` | 4,837 | 332 | 7,765 | 1.61 | 1.04x | A — closest just-over-cap smoke test |
| `case4917_goc` | 4,917 | 1,349 | 6,726 | 1.37 | 1.05x | A — closest just-over-cap smoke test |
| `case5658_epigrids` | 5,658 | 474 | 9,078 | 1.60 | 1.21x | A — moderate scale step |
| `case6468_rte` | 6,468 | 1,295 | 9,000 | 1.39 | 1.39x | A — matched-scale companion to case6470 |
| `case6470_rte` | 6,470 | 1,330 | 9,005 | 1.39 | 1.39x | **A — paper's explicit OOD case** |
| `case6495_rte` | 6,495 | 1,372 | 9,019 | 1.39 | 1.39x | A — matched-scale companion |
| `case6515_rte` | 6,515 | 1,388 | 9,037 | 1.39 | 1.40x | A — matched-scale companion |
| `case7336_epigrids` | 7,336 | 686 | 11,521 | 1.57 | 1.57x | B — larger stress test |
| `case8387_pegase` | 8,387 | 1,865 | 14,561 | 1.74 | 1.80x | B — larger stress test |
| `case9241_pegase` | 9,241 | 1,445 | 16,049 | 1.74 | 1.98x | B — larger stress test |
| `case9591_goc` | 9,591 | 365 | 15,915 | 1.66 | 2.06x | B — larger stress test |
| `case10000_goc` | 10,000 | 2,089 | 13,193 | 1.32 | 2.15x | B — larger stress test |
| `case10192_epigrids` | 10,192 | 722 | 17,043 | 1.67 | 2.19x | B — larger stress test |
| `case10480_goc` | 10,480 | 777 | 18,559 | 1.77 | 2.25x | B — larger stress test |
| `case13659_pegase` | 13,659 | 4,092 | 20,467 | 1.50 | 2.93x | C — memory/runtime qualification needed |
| `case19402_goc` | 19,402 | 971 | 34,704 | 1.79 | 4.16x | C — memory/runtime qualification needed |
| `case20758_epigrids` | 20,758 | 2,250 | 33,368 | 1.61 | 4.45x | C — memory/runtime qualification needed |
| `case24464_goc` | 24,464 | 1,591 | 37,816 | 1.55 | 5.25x | C — memory/runtime qualification needed |
| `case30000_goc` | 30,000 | 3,526 | 35,393 | 1.18 | 6.44x | C — memory/runtime qualification needed |
| `case78484_epigrids` | 78,484 | 6,873 | 126,146 | 1.61 | 16.84x | D — beyond the credible Open-model demo envelope |

## What is certified versus what still needs preparation

| Claim | Status |
|---|---|
| Full topology exceeds GridSFM-Open's reported training cap | Certified by the paper's stated 4,661-bus maximum |
| Full PGLib base-case file is locally available | Confirmed for every row above |
| This exact case was absent from pretraining by explicit case-ID manifest | Only `case6470_rte` has an explicit paper OOD evaluation; not published for the other rows |
| OPFData train/val/test scenarios with ground-truth labels exist locally | Not established by this PGLib repository; it contains base MATPOWER cases, not OPFData scenario splits |
| A candidate can run within available GPU memory | Must be measured; tiers are risk triage, not a guarantee |

## Recommended evaluation ladder

1. Reproduce the paper-backed zero-shot OOD reference: `case6470_rte` using OPFData's `fulltop/test` and `n1/test` scenario splits, if those labels are downloaded separately.
2. Run locally generated AC-OPF scenarios on the Tier-A cases (`4837`, `4917`, `5658`, `6468`, `6495`, `6515`) as an **additional topology-and-scale stress suite**. Label them as self-generated PGLib OOD cases, not an OPFData held-out test split.
3. Move to Tier B only after profiling memory and wall time on the target GPU.

For all claims, preserve the distinction between *unseen source family* (not established) and *full topology outside the published GridSFM-Open training-size cap* (established).
