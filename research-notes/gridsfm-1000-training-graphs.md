# What GridSFM means by “1,000 training graphs”

## Short answer

The paper's **1,000 training graphs are 1,000 operating scenarios encoded as
graph samples for one fixed grid topology**, OPFData
`pglib_opf_case6470_rte/fulltop`. They are not 1,000 distinct power grids.

The white paper makes this equivalence explicit: Section 8 first says “1,000
training graphs,” later calls them approximately 1,000 “scenarios,” and Table
9 describes the same run as a “1,000-scenario fine-tune.” The checkpoint is
fine-tuned for 10 epochs, while the N-1 split is entirely held out.

Source: [GridSFM white paper, Section 8 and Table 9, pp. 13–14](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/05/GridFM_white_paper.pdf#page=13).

## What varies between graph samples

For OPFData's `FullTop` variant, the network connectivity is fixed. Each
example is a self-contained AC-OPF problem and solution with different load
conditions: every active and reactive load is independently multiplied by a
uniformly sampled factor in `[0.8, 1.2]`. Thus, a useful abstraction is

`G_i = (V, E, x_i, y_i)` for `i = 1, ..., 1000`,

where the bus/component sets and transmission edges `(V, E)` remain the same,
the operating features `x_i` vary, and `y_i` is the corresponding solved
AC-OPF target.

Source: [OPFData paper, Section 2, pp. 2–3](https://arxiv.org/pdf/2406.07234#page=2).

## Counts and held-out data

| Quantity | Meaning |
|---|---|
| Distinct fine-tuning topologies | 1: intact 6,470-bus `case6470_rte` |
| Base training samples | 1,000 `fulltop` operating scenarios |
| Epochs | 10 passes over the selected training set |
| Base graph presentations | 10,000, before considering batching |
| N-1 fine-tuning samples | 0; N-1 is held out |
| Validation and test samples | Separate; not part of the 1,000 |

The paper's few-shot ablation treats `n_train` as the scenario count and
sweeps `{10, 100, 500, 1,000}`. It reports that different output channels need
different budgets: cost and active dispatch improve with about 10 scenarios,
whereas voltage calibration continues improving toward 1,000.

Source: [GridSFM white paper, Section 9 and Tables 11–12, p. 15](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/05/GridFM_white_paper.pdf#page=15).

## Practical implication

A PGLib `.m` file supplies one base topology, not the 1,000 fine-tuning
examples. Reproducing the experiment requires 1,000 scenario records for that
topology with AC-OPF solution labels, either downloaded from OPFData or
generated and solved locally.

The released adapter expresses this directly with `variant="fulltop"`,
`split="train"`, and `n_graphs=1000`; its documented split contains 13,500
training graphs before applying that cap.

Source: [Microsoft GridSFM OPFData adapter](https://github.com/microsoft/GridSFM/blob/1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6/model/gridsfm/opfdata_train.py#L33-L102).

## Released-code nuance

The later released notebook uses a batch size of 8 and demonstrates training
budgets `{16, 104, 1,000}` rather than the paper's `{10, 100, 500, 1,000}` so
the smaller runs form whole batches. For the 1,000-sample run, this is 125
batches per epoch and 1,250 scheduled batch updates over 10 epochs.

It also wraps the 1,000 base samples in `SyntheticMixedDataset`; roughly 30%
are converted to synthetic infeasible variants, with perturbations changing by
epoch. The dataset cardinality remains 1,000 even though a sample's realized
numerical features can vary between epochs.

Sources: [released fine-tuning notebook](https://github.com/microsoft/GridSFM/blob/1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6/model/examples/finetune_opfdata_case6470.ipynb), [synthetic dataset implementation](https://github.com/microsoft/GridSFM/blob/1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6/model/gridsfm/synthetic.py#L404-L433).
