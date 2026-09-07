# GridSFM US data, training coverage, and held-out evaluation

## Bottom line

- The released US dataset is **not a dump of a utility's exact real grid**, and it is not a distribution-grid dataset. It is a collection of geographically grounded, electrically coherent **transmission-network models reconstructed from public data**. OpenStreetMap supplies infrastructure geography; EIA and other public sources supply generation and demand information. Electrical parameters and some topology are estimated or incomplete, demand allocation is heuristic, and reactive compensation includes synthetic shunts. It is appropriate to call the networks *realistically grounded synthetic/reconstructed transmission models*, not ground-truth operational networks.
- GridSFM-Open was trained using the `msr_*` US transmission-topology corpus, together with PGLib-OPF and OPFData. Training expanded base topologies through load, outage, line-rating, voltage-limit, cost-order, and infeasibility perturbations. The paper reports roughly 200 grids and more than 500,000 scenarios.
- The paper's main 54-grid test corpus is **scenario-held-out, not topology-held-out**. It explicitly describes the aggregate as test splits of grids the checkpoint was trained on. Thus an `msr_*` result from that corpus tests unseen operating scenarios on a topology that was present during training; it does not demonstrate generalization to a new grid.
- The authors' documented topology-held-out showcase is **OPFData `pglib_opf_case6470_rte`**. GridSFM-Open was trained only on grids up to 4,661 buses, so the 6,470-bus case is described as an unseen OOD grid. Both its `fulltop` test split and `n1` split are zero-shot probes. In the paper's later fine-tuning experiment, training uses 1,000 `fulltop` training graphs while `n1` remains completely held out.
- The 30 `model/samples/msr_*.pyg.json` files are unperturbed 16h peak snapshots derived from the US release. Neither the paper nor the sample README assigns those particular files a train/validation/test membership. Therefore they are convenient demos with AC-OPF ground truth, but they should **not** be presented as a certified never-seen test cut. The local US release's dataset card recommends users create geographic or operating-condition splits; it does not ship an official split manifest for the released base snapshots.

## What “real” means here

The official dataset card calls the 54 instances (48 contiguous states and six multi-state regions) geographically grounded transmission models derived entirely from open data. It also warns that exact topology, utility-grade measured electrical parameters, protection settings, dynamics, and detailed distribution-level networks are absent. In particular, estimated impedances/thermal limits, incomplete parallel circuits, heuristic demand placement, and synthetic reactive compensation separate it from an as-operated utility model.

So the accurate phrasing is:

> GridSFM US contains reconstructed US transmission models grounded in real public geography, generator metadata, and demand data, with inferred and synthetic engineering parameters where public data is incomplete.

## Which evaluation cut answers which question?

| Cut | Topology seen during pre-training? | Scenario held out? | Suitable claim |
|---|---:|---:|---|
| Paper's 54-grid GridSFM-Open test corpus | Yes | Yes | In-distribution performance on held-out operating scenarios |
| Local `model/samples/msr_*` base snapshots | The `msr_*` topology family was used in training; exact file membership is undocumented | Undocumented for these exact files | Inference/viewer demo, not a provably unseen benchmark |
| OPFData `case6470_rte/fulltop` test | No during pre-training | Yes | Zero-shot generalization to an unseen, larger topology |
| OPFData `case6470_rte/n1` | No during pre-training; also held out during the paper's `fulltop` fine-tune | Yes | Strongest documented unseen-topology/contingency showcase |

## Practical recommendation

For an honest local visual demo, use California/Texas/New England from `model/samples` and label it “pretrained inference against AC-OPF ground truth on US-derived topologies.” For a scientifically cleaner unseen-grid test, use the repository's `model/examples/opfdata.py` with `case6470_rte` and `--split test`, and explicitly label it “zero-shot OOD.” Do not claim the exact local `msr_*` sample was absent from training unless Microsoft publishes a scenario-ID split manifest that proves that.

## Primary sources

1. Microsoft Research, *GridSFM: A Foundation Model for AC Optimal Power Flow*, May 2026. Training sources and perturbations: Section 5, pp. 4–5; held-out 54-grid results: Sections 6.1–6.3, pp. 5–9; unseen `case6470_rte`: Section 7, p. 13; fine-tuning and fully held-out `n1`: Section 8, pp. 13–14. [Official white paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/05/GridFM_white_paper.pdf)
2. Microsoft, *GridSFM US Power Grid Dataset* card. See “Dataset Contents,” “Data Creation & Processing,” “Limitations,” and “Best Practices.” [Official Hugging Face dataset card](https://huggingface.co/datasets/microsoft/GridSFM_US_power_grid)
3. Microsoft GridSFM repository, sample provenance and the warning that shipped samples are unperturbed base cases. [Official sample README](https://github.com/microsoft/GridSFM/tree/main/model/samples)
4. Microsoft GridSFM repository, OPFData evaluator supporting `train`, `val`, and `test` splits. [Official evaluator source](https://github.com/microsoft/GridSFM/blob/main/model/examples/opfdata.py)

