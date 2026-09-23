# GridSFM FFN LoRA specification

## Goal

Add a named `gridblock_ffn_r2_a2` LoRA adapter to both linear layers of every
`GridBlock` FFN. It must work with any grid supported by the same GridSFM
architecture.

## Implementation

Add `gridsfm/lora.py` with:

- `LoRAConfig(target="gridblock_ffn", rank=2, alpha=2.0)`
- `LoRAAdapter`: computes `(alpha / rank) * B(A(x))`; initialize `A` normally
  and `B` to zero.
- `LoRALinear`: preserves the original linear weight and bias, then adds the
  active adapter output.
- `GridSFMLoRAModel`: wraps a loaded backbone and manages named adapters.

Target these layers structurally:

```text
blocks.{block}.ffn.{node_type}.0
blocks.{block}.ffn.{node_type}.2
```

The default model has 112 targets. Rank 2 adds 143,360 trainable parameters.
All original model parameters remain frozen.

Minimal interface:

```python
base = load_model(checkpoint, device=device)
model = GridSFMLoRAModel(base)
model.add_adapter(LoRAConfig(target="gridblock_ffn", rank=2, alpha=2))
# Returns "gridblock_ffn_r2_a2"
model.enable_adapter("gridblock_ffn_r2_a2")
model.disable_adapter()
model.save_adapter("gridblock_ffn_r2_a2", path)
model.load_adapter(path)
```

## Pre-training equivalence gate

Before training:

1. Load the base model in `eval()` mode and run one cloned batch.
2. Capture with forward hooks:
   - outputs of both FFN linear layers;
   - output dictionaries from every `GridBlock`;
   - final bus, generator, flow, and feasibility predictions.
3. Add and enable `gridblock_ffn_r2_a2`.
4. Run an identical clone of the batch and capture the same tensors.
5. Compare every tensor with:

```python
torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
```

This must pass on CPU float32. It proves the zero-initialized adapter does not
change the first forward pass internally or at the final outputs.

## Training changes

Update `finetune_opfdata()` to optimize and clip only trainable parameters:

```python
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
```

Initial training recipe:

```text
adapter:        gridblock_ffn_r2_a2
learning rate:  1e-3
weight decay:   0
loss:           existing compute_loss()
base model:     frozen
```

First overfit one fixed batch for 25-50 steps. Require:

- finite and decreasing loss;
- at least one LoRA `B` weight changes;
- no original model weight changes;
- disabling the adapter restores the original predictions;
- re-enabling it restores the trained predictions.

Then run the normal 10-epoch OPFData fine-tuning path. Compare against the
zero-shot model and existing full fine-tuning. Save the best validation adapter,
not simply the last epoch.

## Required tests

1. Exactly 112 FFN linear layers receive the adapter.
2. Exact inner-layer and final-output equivalence before training.
3. Exactly 143,360 parameters are trainable for `gridblock_ffn_r2_a2`.
4. Gradients reach LoRA weights and never reach frozen weights.
5. Enable/disable and named-adapter switching work.
6. Saving and loading the adapter on a fresh base model reproduces predictions.
7. One fixed-batch overfit and one normal fine-tuning epoch complete.

After this passes, run the rank/alpha sweep while keeping the dataset, seed,
learning rate, and epoch count fixed. Record both `alpha` and `alpha / rank`.
