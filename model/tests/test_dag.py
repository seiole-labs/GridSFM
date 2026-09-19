from __future__ import annotations

import io

import torch
from torch import nn
from torch_geometric.data import HeteroData

from gridsfm import (
    GridTransformerBackbone,
    format_gridsfm_architecture,
    format_module_dag,
    print_module_dag,
    trace_module_dag,
)


def test_trace_module_dag_records_shapes_params_and_dependencies():
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )
    x = torch.zeros(5, 3)

    with torch.inference_mode():
        trace = trace_module_dag(model, x)

    assert [call.name for call in trace.calls] == ["0", "1", "2"]
    assert trace.calls[0].input_summary == "(f32[5,3]@cpu)"
    assert trace.calls[0].output_summary == "f32[5,4]@cpu"
    assert trace.calls[1].input_from == (1,)
    assert trace.calls[2].input_from == (2,)
    assert trace.calls[0].direct_parameters == 16
    assert trace.total_parameters == 26

    report = format_module_dag(trace)
    assert "[0001] 0 <Linear>" in report
    assert "from=0001" in report
    assert "MODEL OUTPUT f32[5,2]@cpu" in report


def test_print_module_dag_writes_to_stream_and_returns_trace():
    stream = io.StringIO()
    model = nn.Sequential(nn.Linear(2, 1))

    with torch.inference_mode():
        trace = print_module_dag(model, torch.ones(3, 2), file=stream)

    assert len(trace.calls) == 1
    assert "f32[3,2]@cpu" in stream.getvalue()
    assert "f32[3,1]@cpu" in stream.getvalue()


def test_trace_captures_input_before_in_place_model_mutation():
    class MutatingModel(nn.Module):
        def forward(self, value):
            value["prediction"] = torch.ones(1)
            return value

    value = {"features": torch.zeros(2, 3)}
    with torch.inference_mode():
        trace = trace_module_dag(MutatingModel(), value)

    assert "prediction" not in trace.input_summary
    assert "prediction" in trace.output_summary


def test_collapsed_gridsfm_architecture_keeps_stage_shapes_and_multiplicity():
    model = GridTransformerBackbone(hidden_dim=16, num_blocks=2, num_heads=4)
    data = HeteroData()
    counts = {
        "bus": (5, 16),
        "generator": (2, 11),
        "load": (3, 2),
        "shunt": (1, 2),
        "branch_ac": (4, 13),
        "branch_tr": (1, 15),
        "cycle": (1, 4),
    }
    for node_type, (count, width) in counts.items():
        data[node_type].x = torch.zeros(count, width)
    data["bus"].pred = torch.zeros(5, 2)
    data["generator"].pred = torch.zeros(2, 2)
    data.feas_logit = torch.zeros(1)
    for relation in model.EDGE_TYPES:
        data[relation].edge_index = torch.zeros(2, 1, dtype=torch.long)
    for relation, count in [
        (("bus", "ac_line", "bus"), 4),
        (("bus", "transformer", "bus"), 1),
    ]:
        data[relation].edge_index = torch.zeros(2, count, dtype=torch.long)
        data[relation].edge_flow_pred = torch.zeros(count, 4)

    report = format_gridsfm_architecture(model, data)

    assert "GridBlock x2" in report
    assert "linear self-attention (4 heads)" in report
    assert "16 -> 64 -> 16" in report
    assert "bus.pred        f32[5,2]@cpu" in report
    assert "generator.pred  f32[2,2]@cpu" in report
