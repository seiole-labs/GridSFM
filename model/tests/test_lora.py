"""FFN-only LoRA coverage for GridSFM."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

import gridsfm.cycle_basis as cycle_basis
from gridsfm import (
    GridSFMLoRAModel,
    GridTransformerBackbone,
    LoRAConfig,
    batch_data_list,
    load_model,
    load_pyg_json,
    prepare_for_inference,
)


ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "gridsfm_open_v1.1.pt"
SAMPLE = ROOT / "samples" / "case500_goc.pyg.json"


def test_rank2_alpha2_adapter_targets_only_gridblock_ffns():
    base = GridTransformerBackbone(
        hidden_dim=8,
        num_blocks=2,
        num_heads=2,
        ffn_mult=4,
    )

    model = GridSFMLoRAModel(base)
    name = model.add_adapter(
        LoRAConfig(target="gridblock_ffn", rank=2, alpha=2.0)
    )
    summary = model.trainable_parameter_summary()

    assert name == "gridblock_ffn_r2_a2"
    assert model.active_adapter == name
    assert summary["target_modules"] == 2 * 7 * 2
    assert summary["trainable_params"] == 2_240
    assert all(
        (".adapters.gridblock_ffn_r2_a2." in parameter_name)
        == parameter.requires_grad
        for parameter_name, parameter in model.named_parameters()
    )
    assert all(".ffn." in target for target in model.target_modules)
    assert all(target.endswith((".0", ".2")) for target in model.target_modules)


def test_named_adapter_can_be_disabled_and_reenabled():
    torch.manual_seed(7)
    base = GridTransformerBackbone(
        hidden_dim=8, num_blocks=1, num_heads=2, ffn_mult=4,
    )
    model = GridSFMLoRAModel(base)
    name = model.add_adapter(
        LoRAConfig(target="gridblock_ffn", rank=2, alpha=2.0)
    )
    layer = model.base_model.blocks[0].ffn["bus"][0]
    x = torch.randn(3, 8)

    model.disable_adapter()
    base_output = layer(x).detach().clone()

    with torch.no_grad():
        layer.adapters[name].lora_B.weight.fill_(0.1)
    model.enable_adapter(name)
    adapted_output = layer(x).detach().clone()
    assert not torch.equal(adapted_output, base_output)

    model.disable_adapter()
    torch.testing.assert_close(layer(x), base_output, rtol=0, atol=0)
    model.enable_adapter()
    torch.testing.assert_close(layer(x), adapted_output, rtol=0, atol=0)


def test_backward_updates_adapter_without_touching_base_weight():
    torch.manual_seed(11)
    base = GridTransformerBackbone(
        hidden_dim=8, num_blocks=1, num_heads=2, ffn_mult=4,
    )
    model = GridSFMLoRAModel(base)
    name = model.add_adapter(
        LoRAConfig(target="gridblock_ffn", rank=2, alpha=2.0)
    )
    layer = model.base_model.blocks[0].ffn["bus"][0]
    base_weight = layer.weight.detach().clone()
    adapter_B = layer.adapters[name].lora_B.weight
    adapter_B_before = adapter_B.detach().clone()
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=1e-2)

    loss = layer(torch.randn(4, 8)).square().mean()
    loss.backward()
    optimizer.step()

    assert layer.weight.grad is None
    torch.testing.assert_close(layer.weight, base_weight, rtol=0, atol=0)
    assert adapter_B.grad is not None
    assert torch.isfinite(adapter_B.grad).all()
    assert torch.count_nonzero(adapter_B.grad) > 0
    assert not torch.equal(adapter_B, adapter_B_before)


def test_adapter_checkpoint_round_trip_reproduces_output(tmp_path):
    torch.manual_seed(13)
    base = GridTransformerBackbone(
        hidden_dim=8, num_blocks=1, num_heads=2, ffn_mult=4,
    )
    fresh_base = GridTransformerBackbone(
        hidden_dim=8, num_blocks=1, num_heads=2, ffn_mult=4,
    )
    fresh_base.load_state_dict(base.state_dict())
    model = GridSFMLoRAModel(base)
    name = model.add_adapter(
        LoRAConfig(target="gridblock_ffn", rank=2, alpha=2.0)
    )
    layer = model.base_model.blocks[0].ffn["bus"][0]
    with torch.no_grad():
        layer.adapters[name].lora_B.weight.normal_()
    x = torch.randn(3, 8)
    expected = layer(x).detach().clone()

    path = tmp_path / "gridblock_ffn_r2_a2.pt"
    model.save_adapter(name, path)

    restored = GridSFMLoRAModel(fresh_base)
    restored_name = restored.load_adapter(path)
    actual = restored.base_model.blocks[0].ffn["bus"][0](x)

    assert restored_name == name
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _capture_forward(model, batch):
    activations = {}
    handles = []

    def capture_tensor(name):
        def hook(_module, _inputs, output):
            activations[name] = output.detach().clone()
        return hook

    def capture_block(name):
        def hook(_module, _inputs, output):
            activations[name] = {
                node_type: tensor.detach().clone()
                for node_type, tensor in output.items()
            }
        return hook

    backbone = getattr(model, "base_model", model)
    for block_index, block in enumerate(backbone.blocks):
        handles.append(
            block.register_forward_hook(capture_block(f"block.{block_index}"))
        )
        for node_type, ffn in block.ffn.items():
            for projection_index in (0, 2):
                name = f"block.{block_index}.ffn.{node_type}.{projection_index}"
                handles.append(
                    ffn[projection_index].register_forward_hook(capture_tensor(name))
                )

    try:
        with torch.inference_mode():
            output = model(batch)
    finally:
        for handle in handles:
            handle.remove()

    predictions = {
        "bus": output["bus"].pred.detach().clone(),
        "generator": output["generator"].pred.detach().clone(),
        "feas_logit": output.feas_logit.detach().clone(),
    }
    for edge_type in output.edge_types:
        if hasattr(output[edge_type], "edge_flow_pred"):
            predictions[f"flow.{edge_type[1]}"] = (
                output[edge_type].edge_flow_pred.detach().clone()
            )
    return activations, predictions


def _assert_nested_exact(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], dict):
            _assert_nested_exact(actual[key], expected[key])
        else:
            torch.testing.assert_close(
                actual[key], expected[key], rtol=0, atol=0, equal_nan=True,
            )


def test_fresh_adapter_preserves_internal_and_final_forward_exactly(
    tmp_path, monkeypatch,
):
    if not CKPT.exists():
        pytest.skip(f"checkpoint not found at {CKPT}")

    monkeypatch.setattr(
        cycle_basis,
        "DEFAULT_CYCLE_CACHE",
        cycle_basis.CycleBasisCache(cache_dir=tmp_path),
    )
    base = load_model(CKPT, device="cpu")
    prepared = prepare_for_inference(load_pyg_json(SAMPLE))
    batch_before = batch_data_list([prepared.clone()])
    batch_after = batch_data_list([prepared.clone()])

    base_activations, base_predictions = _capture_forward(base, batch_before)

    model = GridSFMLoRAModel(base)
    model.add_adapter(
        LoRAConfig(target="gridblock_ffn", rank=2, alpha=2.0)
    )
    model.eval()
    lora_activations, lora_predictions = _capture_forward(model, batch_after)

    assert len(model.target_modules) == 112
    assert model.trainable_parameter_summary()["trainable_params"] == 143_360
    _assert_nested_exact(lora_activations, base_activations)
    _assert_nested_exact(lora_predictions, base_predictions)
