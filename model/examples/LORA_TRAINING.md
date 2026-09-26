# LoRA training for GridSFM case6470

This guide trains the FFN-only LoRA adapter for
`pglib_opf_case6470_rte` with:

```yaml
adapter:
  target: gridblock_ffn
  rank: 4
  alpha: 4.0
  dropout: 0.0
```

Because `alpha / rank = 1`, the LoRA update has a scaling factor of 1. The
released GridSFM v1.1 backbone stays frozen; only the adapter parameters are
optimized. This training script is intentionally CPU-only.

## 1. Create the environment

Run these commands from the repository root:

```bash
cd model
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
cd ..
```

Python 3.10 or newer is required.

## 2. Download the v1.1 backbone

The LoRA training path is validated only with GridSFM v1.1:

```bash
model/.venv/bin/hf download microsoft/GridSFM_Open \
  gridsfm_open_v1.1.pt \
  --local-dir model/checkpoints
```

After the download, this file must exist:

```text
model/checkpoints/gridsfm_open_v1.1.pt
```

## 3. Understand the dataset download

The trainer uses the public OPFData `pglib_opf_case6470_rte` dataset. On the
first run, PyTorch Geometric downloads and processes group 0 automatically
under the cache path configured in your YAML.

Important: group 0 contains 15,000 scenarios and its compressed archive is
about 13.55 GB. Setting the recipe to 10, 100, or 1,000 graphs limits training
only after the entire shard has been downloaded and processed. Allow enough
disk space for both the archive and processed cache.

## 4. Create a rank-4, alpha-4 configuration

The checked-in YAML recipes remain unchanged. Make a local copy of the full
recipe:

```bash
cp model/examples/lora_case6470.yaml /tmp/lora_case6470_r4_a4.yaml
```

Open `/tmp/lora_case6470_r4_a4.yaml` and change only these values:

```yaml
run:
  name: case6470_gridblock_ffn_lora_r4_a4

adapter:
  checkpoint: null
  target: gridblock_ffn
  rank: 4
  alpha: 4.0
  dropout: 0.0
```

The remaining settings are inherited from the repository's full recipe.

For a smoke run, copy the smoke recipe and make the same changes:

```bash
cp model/examples/lora_case6470_smoke.yaml \
  /tmp/lora_case6470_r4_a4_smoke.yaml
```

Use a distinct `run.name`, such as
`case6470_gridblock_ffn_lora_r4_a4_smoke10`. To avoid downloading the large
shard twice, you may also set the copied smoke config's `data.root` to the same
value used by your copied full config.

## 5. Run a smoke training job

The smoke recipe trains on 10 graphs and validates on 2 graphs. It skips the
final test-set comparison, but the initial dataset download is still the full
group-0 download described above.

```bash
PYTHONPATH=model model/.venv/bin/python \
  model/examples/train_lora_case6470.py \
  --config /tmp/lora_case6470_r4_a4_smoke.yaml
```

Use this run to verify that the environment, checkpoint, dataset processing,
forward pass, validation, checkpoint writing, and plots all work.

## 6. Run the full recipe

The full recipe trains on 1,000 `fulltop/train` graphs for 10 epochs, validates
on 750 `fulltop/val` graphs, and evaluates the frozen base model and best LoRA
adapter on both the 750-graph `fulltop/test` and `n1/test` splits.

```bash
PYTHONPATH=model model/.venv/bin/python \
  model/examples/train_lora_case6470.py \
  --config /tmp/lora_case6470_r4_a4.yaml
```

Run the command from the repository root. Relative paths in the YAML are
resolved from the repository root, regardless of the current shell directory,
but `PYTHONPATH=model` in the command assumes the root directory.

## 7. Inspect the outputs

Each invocation creates a timestamped directory:

```text
artifacts/lora_runs/<timestamp>_case6470_gridblock_ffn_lora_r4_a4/
```

The smoke recipe writes to `artifacts/lora_smoke_runs/` instead. A completed
run contains:

- `config.yaml`: exact configuration snapshot used by the run
- `metrics.jsonl`: append-only training, validation, and evaluation metrics
- `learning_rate.svg`: cosine learning-rate schedule
- `train_validation_loss.svg`: epoch-level loss curves
- `best_adapter.pt`: adapter with the lowest validation loss
- `final_adapter.pt`: adapter from the final epoch

Progress is also printed to the terminal for every optimizer step. If a run
fails, its `metrics.jsonl` ends with a `run_error` record containing the error
type and message.

## Configuration notes

The requested rank and alpha must be set explicitly in your copied YAML:

```yaml
rank: 4
alpha: 4.0
```

Edit `training.epochs`, `data.train.graphs`, or `loader.batch_size` in a copied
YAML file to create another experiment. Give the copy a distinct `run.name` so
its output directory is easy to identify.

To initialize from an existing adapter, set `adapter.checkpoint` to its `.pt`
path. This loads the adapter weights, but starts a new optimizer and cosine
schedule; it does not restore optimizer state from an interrupted run.
