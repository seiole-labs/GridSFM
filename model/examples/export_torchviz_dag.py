"""Export the real GridSFM PyTorch autograd DAG with torchviz.

Unlike ``print_model_dag.py``, this file does not draw an architectural
summary. It runs a real forward pass with gradients enabled and asks torchviz
to walk the resulting ``grad_fn`` graph. The result therefore contains raw
PyTorch operations, parameter leaves, tensor shapes, residual connections,
scatter operations, and every path to every model output.

From ``model/``::

    pip install torchviz
    python examples/export_torchviz_dag.py

The default output is ``docs/gridsfm_torchviz_raw.dot``. If the Graphviz
``dot`` executable is installed, pass ``--svg`` to render a zoomable SVG too.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

from gridsfm import load_model, load_pyg_json, prepare_for_inference
import gridsfm.model as model_implementation


def _enable_named_input_gradients(data) -> dict[str, torch.Tensor]:
    """Expose differentiable model inputs as named leaves in torchviz."""
    named: dict[str, torch.Tensor] = {}
    for node_type in data.node_types:
        x = getattr(data[node_type], "x", None)
        if torch.is_tensor(x) and x.is_floating_point():
            x.requires_grad_(True)
            named[f"INPUT.nodes.{node_type}.x"] = x

    for edge_type in data.edge_types:
        edge_attr = getattr(data[edge_type], "edge_attr", None)
        if torch.is_tensor(edge_attr) and edge_attr.is_floating_point():
            edge_attr.requires_grad_(True)
            relation = "__".join(edge_type)
            named[f"INPUT.edges.{relation}.edge_attr"] = edge_attr

    for attribute in ("g_ctx",):
        value = getattr(data, attribute, None)
        if torch.is_tensor(value) and value.is_floating_point():
            value.requires_grad_(True)
            named[f"INPUT.graph.{attribute}"] = value
    return named


def _forward_with_traceable_inputs(model, data):
    """Run forward while preserving the DC prior's intentional detach.

    ``compute_dc_prior`` crosses from Torch into NumPy/SciPy and is therefore
    non-differentiable. Normal training inputs do not require gradients, but
    this visualizer enables them so they show up as torchviz leaf nodes. We
    temporarily present detached data tensors only to the DC-prior function;
    every differentiable Torch path still sees the original leaves.
    """
    original_compute_dc_prior = model_implementation.compute_dc_prior

    def compute_dc_prior_with_detached_data(prior_data, gen_p, cache):
        saved: list[tuple[object, str, torch.Tensor]] = []
        try:
            for node_type in prior_data.node_types:
                store = prior_data[node_type]
                value = getattr(store, "x", None)
                if torch.is_tensor(value) and value.requires_grad:
                    saved.append((store, "x", value))
                    store.x = value.detach()
            for edge_type in prior_data.edge_types:
                store = prior_data[edge_type]
                value = getattr(store, "edge_attr", None)
                if torch.is_tensor(value) and value.requires_grad:
                    saved.append((store, "edge_attr", value))
                    store.edge_attr = value.detach()
            return original_compute_dc_prior(prior_data, gen_p, cache)
        finally:
            for store, attribute, value in saved:
                setattr(store, attribute, value)

    model_implementation.compute_dc_prior = compute_dc_prior_with_detached_data
    try:
        return model(data)
    finally:
        model_implementation.compute_dc_prior = original_compute_dc_prior


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/gridsfm_open_v1.1.pt"),
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("samples/case500_goc.pyg.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/gridsfm_torchviz_raw.dot"),
        help="Destination DOT file",
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Also render SVG (requires the Graphviz 'dot' executable)",
    )
    parser.add_argument(
        "--show-attrs",
        action="store_true",
        help="Include backward-operation attributes; makes the graph much larger",
    )
    parser.add_argument(
        "--show-saved",
        action="store_true",
        help="Include saved backward tensors; makes the graph much larger",
    )
    args = parser.parse_args()

    try:
        from torchviz import make_dot
    except ImportError as error:
        raise SystemExit(
            "torchviz is required: python -m pip install torchviz"
        ) from error

    model = load_model(args.checkpoint, device="cpu")
    data = prepare_for_inference(load_pyg_json(args.scenario))
    named_inputs = _enable_named_input_gradients(data)

    # Do not use torch.no_grad(): torchviz traverses these grad_fn links.
    output = _forward_with_traceable_inputs(model, data)
    output_tensors: dict[str, torch.Tensor] = {
        "OUTPUT.bus.pred(theta,V)": output["bus"].pred,
        "OUTPUT.generator.pred(Pg,Qg)": output["generator"].pred,
        "OUTPUT.feas_logit": output.feas_logit,
    }
    for edge_type, name in (
        (("bus", "ac_line", "bus"), "OUTPUT.ac_line.flows(Pij,Qij,Pji,Qji)"),
        (("bus", "transformer", "bus"), "OUTPUT.transformer.flows(Pij,Qij,Pji,Qji)"),
    ):
        if edge_type in output.edge_types and hasattr(output[edge_type], "edge_flow_pred"):
            output_tensors[name] = output[edge_type].edge_flow_pred

    named_tensors = dict(model.named_parameters())
    named_tensors.update(named_inputs)
    named_tensors.update(output_tensors)

    # torchviz recursively walks a deep autograd graph; the default Python
    # recursion limit is too low for eight heterogeneous transformer blocks.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))
    dot = make_dot(
        tuple(output_tensors.values()),
        params=named_tensors,
        show_attrs=args.show_attrs,
        show_saved=args.show_saved,
    )
    # torchviz's generic resize heuristic produces an enormous requested page
    # size for this 7k-node graph. Let Graphviz determine its natural SVG size.
    dot.graph_attr.pop("size", None)
    dot.graph_attr.update(
        rankdir="LR",
        bgcolor="#ffffff",
        label=(
            f"Raw torchviz autograd DAG: {args.scenario.name} | "
            f"outputs={len(output_tensors)}"
        ),
        labelloc="t",
        fontsize="18",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dot.source, encoding="utf-8")
    print(f"Raw torchviz DOT: {args.output.resolve()}")
    print(f"DOT lines: {len(dot.source.splitlines()):,}")

    if args.svg:
        if shutil.which("dot") is None:
            raise SystemExit(
                "DOT was written, but SVG rendering needs Graphviz. Install "
                "it (for Ubuntu: sudo apt-get install graphviz), then rerun --svg."
            )
        rendered = Path(dot.render(
            filename=str(args.output.with_suffix("")),
            format="svg",
            cleanup=False,
        ))
        print(f"Zoomable SVG: {rendered.resolve()}")


if __name__ == "__main__":
    main()
