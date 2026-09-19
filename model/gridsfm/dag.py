"""Runtime module-DAG tracing with tensor input/output shapes.

PyTorch FX cannot symbolically trace GridSFM's dynamic ``HeteroData`` control
flow. This module instead attaches temporary forward hooks and records the
modules that actually execute for a representative input graph.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional, TextIO

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LayerCall:
    """One executed ``nn.Module`` invocation."""

    index: int
    name: str
    module_type: str
    depth: int
    input_summary: str
    output_summary: str
    input_from: tuple[int, ...]
    direct_parameters: int
    direct_trainable_parameters: int
    subtree_parameters: int
    is_leaf: bool


@dataclass
class ModuleDagTrace:
    """Result of tracing one model forward pass."""

    calls: list[LayerCall]
    input_summary: str
    output_summary: str
    total_parameters: int
    trainable_parameters: int
    output: Any


_DTYPE_NAMES = {
    torch.float16: "f16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.bfloat16: "bf16",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.uint8: "u8",
    torch.bool: "bool",
}


def _tensor_summary(value: Tensor) -> str:
    shape = ",".join(str(int(dim)) for dim in value.shape)
    dtype = _DTYPE_NAMES.get(value.dtype, str(value.dtype).removeprefix("torch."))
    return f"{dtype}[{shape}]@{value.device.type}"


def _summary(value: Any, *, max_items: int = 32) -> str:
    if isinstance(value, Tensor):
        return _tensor_summary(value)

    # HeteroData exposes a compact nested tensor mapping through to_dict().
    # Duck typing keeps this utility usable without importing PyG itself.
    if (hasattr(value, "node_types") and hasattr(value, "edge_types")
            and callable(getattr(value, "to_dict", None))):
        return f"{type(value).__name__}" + _summary(
            value.to_dict(), max_items=max_items
        )

    if isinstance(value, Mapping):
        items = list(value.items())
        shown = items[:max_items]
        body = ", ".join(
            f"{key!s}: {_summary(item, max_items=max_items)}"
            for key, item in shown
        )
        if len(items) > max_items:
            body += f", ... +{len(items) - max_items}"
        return "{" + body + "}"

    if isinstance(value, tuple):
        return "(" + ", ".join(
            _summary(item, max_items=max_items) for item in value
        ) + ")"

    if isinstance(value, list):
        shown = ", ".join(
            _summary(item, max_items=max_items) for item in value[:max_items]
        )
        if len(value) > max_items:
            shown += f", ... +{len(value) - max_items}"
        return "[" + shown + "]"

    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    return f"<{type(value).__name__}>"


def _walk_tensors(value: Any, seen: Optional[set[int]] = None) -> Iterator[Tensor]:
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    if isinstance(value, Tensor):
        yield value
        return
    if (hasattr(value, "node_types") and hasattr(value, "edge_types")
            and callable(getattr(value, "to_dict", None))):
        yield from _walk_tensors(value.to_dict(), seen)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_tensors(item, seen)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_tensors(item, seen)


def trace_module_dag(
    model: nn.Module,
    *model_args: Any,
    **model_kwargs: Any,
) -> ModuleDagTrace:
    """Execute ``model`` once and capture every invoked module's I/O shapes.

    ``input_from`` contains earlier call indices when an input tensor is the
    exact output object of another module. Empty dependencies mean the value
    came from the model input or from a functional operation (for example
    ``torch.cat`` or a residual add) that has no ``nn.Module`` hook.

    Hooks are always removed, including when the forward pass raises. The
    caller controls grad/eval state; use ``torch.inference_mode()`` for a pure
    inference trace.
    """
    calls: list[LayerCall] = []
    producer_by_tensor_id: dict[
        int, tuple[weakref.ReferenceType[Tensor], int]
    ] = {}
    handles: list[Any] = []

    def make_hook(name: str):
        depth = name.count(".") + 1

        def hook(
            module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            input_value: Any = args
            if kwargs:
                input_value = {"args": args, "kwargs": kwargs}
            parents = set()
            for tensor in _walk_tensors(input_value):
                producer = producer_by_tensor_id.get(id(tensor))
                # Checking the weak reference prevents a dead tensor's reused
                # Python id from creating a false DAG edge.
                if producer is not None and producer[0]() is tensor:
                    parents.add(producer[1])
            direct = list(module.parameters(recurse=False))
            subtree = list(module.parameters(recurse=True))
            call_index = len(calls) + 1
            calls.append(LayerCall(
                index=call_index,
                name=name,
                module_type=type(module).__name__,
                depth=depth,
                input_summary=_summary(input_value),
                output_summary=_summary(output),
                input_from=tuple(sorted(parents)),
                direct_parameters=sum(p.numel() for p in direct),
                direct_trainable_parameters=sum(
                    p.numel() for p in direct if p.requires_grad
                ),
                subtree_parameters=sum(p.numel() for p in subtree),
                is_leaf=not any(module.children()),
            ))
            for tensor in _walk_tensors(output):
                producer_by_tensor_id[id(tensor)] = (weakref.ref(tensor), call_index)

        return hook

    for name, module in model.named_modules():
        if not name:  # Root I/O is recorded separately in ModuleDagTrace.
            continue
        handles.append(module.register_forward_hook(make_hook(name), with_kwargs=True))

    input_value: Any = model_args
    if model_kwargs:
        input_value = {"args": model_args, "kwargs": model_kwargs}
    # Capture this before forward: models such as GridSFM attach predictions
    # to and return the same mutable HeteroData object they receive.
    input_summary = _summary(input_value)
    try:
        output = model(*model_args, **model_kwargs)
    finally:
        for handle in handles:
            handle.remove()

    parameters = list(model.parameters())
    return ModuleDagTrace(
        calls=calls,
        input_summary=input_summary,
        output_summary=_summary(output),
        total_parameters=sum(p.numel() for p in parameters),
        trainable_parameters=sum(p.numel() for p in parameters if p.requires_grad),
        output=output,
    )


def format_module_dag(trace: ModuleDagTrace, *, leaf_only: bool = False) -> str:
    """Render a runtime trace as a readable, line-oriented DAG report."""
    visible = [call for call in trace.calls if call.is_leaf or not leaf_only]
    lines = [
        "PyTorch runtime module DAG",
        "==========================",
        f"parameters: {trace.total_parameters:,} total; "
        f"{trace.trainable_parameters:,} trainable",
        f"executed modules: {len(trace.calls):,} total; "
        f"{sum(call.is_leaf for call in trace.calls):,} leaf calls",
        f"display: {'leaf modules only' if leaf_only else 'all invoked modules'}",
        "",
        f"MODEL INPUT  {trace.input_summary}",
        f"MODEL OUTPUT {trace.output_summary}",
        "",
        "Dependencies in `from=` are exact module-output tensor identities.",
        "`input/functional` denotes model inputs or unhooked torch operations",
        "such as concatenation, indexing, scatter, residual addition, and physics code.",
        "",
    ]
    for call in visible:
        parents = ",".join(f"{idx:04d}" for idx in call.input_from)
        if not parents:
            parents = "input/functional"
        params = (
            f"params={call.direct_parameters:,}"
            if call.is_leaf
            else f"subtree_params={call.subtree_parameters:,}"
        )
        lines.extend([
            f"[{call.index:04d}] {'  ' * (call.depth - 1)}{call.name} "
            f"<{call.module_type}> from={parents} {params}",
            f"         in:  {call.input_summary}",
            f"         out: {call.output_summary}",
        ])
    return "\n".join(lines) + "\n"


def format_gridsfm_architecture(model: nn.Module, data: Any) -> str:
    """Render GridSFM as a compact architecture DAG with concrete shapes.

    Repeated type-specific layers and transformer blocks are shown once with
    multiplicities (for example ``per node type`` and ``GridBlock x8``). The
    supplied ``data`` should be the result of a representative forward pass so
    final prediction and flow shapes are available.
    """
    hidden_dim = int(model.hidden_dim)
    blocks = list(model.blocks)
    n_blocks = len(blocks)
    hodge_k = int(model.hodge_pe.K)
    hodge_steps = int(model.hodge_pe.T)
    hodge_forms = set(model.hodge_pe.active_forms)
    n_graphs = int(getattr(data, "num_graphs", 1) or 1)

    first_block = blocks[0] if blocks else None
    if first_block is not None:
        first_attn = next(iter(first_block.attn.values()))
        num_heads = int(first_attn.num_heads)
        ffn_hidden = int(next(iter(first_block.ffn.values()))[0].out_features)
    else:
        num_heads = 0
        ffn_hidden = 0

    node_rows = []
    for node_type in model.NODE_TYPES:
        if node_type not in data.node_types:
            continue
        x = data[node_type].x
        count, input_width = int(x.size(0)), int(x.size(1))
        hodge_width = hodge_k if node_type in hodge_forms else 0
        encoder_width = input_width + hodge_width
        hodge_text = f"+{hodge_width}" if hodge_width else "-"
        node_rows.append(
            (node_type, count, input_width, hodge_text, encoder_width, hidden_dim)
        )

    relation_count = sum(
        1 for edge_type in model.EDGE_TYPES
        if edge_type in data.edge_types
        and int(data[edge_type].edge_index.size(1)) > 0
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )

    lines = [
        "GridSFM architecture DAG (replicated layers collapsed)",
        "======================================================",
        f"parameters: {total_parameters:,} total; "
        f"{trainable_parameters:,} trainable",
        f"representative batch: {n_graphs} graph(s)",
        "",
        "INPUT: heterogeneous power-grid graph",
        "-------------------------------------",
        "type          count   input   Hodge PE   encoder in   trunk out",
    ]
    for node_type, count, input_width, hodge_text, encoder_width, trunk_width in node_rows:
        lines.append(
            f"{node_type:<12} {count:>7,} {input_width:>7} "
            f"{hodge_text:>10} {encoder_width:>12} {trunk_width:>11}"
        )

    norm_name = "LayerNorm + " if bool(model.input_norm_enabled) else ""
    lines.extend([
        "",
        "                         heterogeneous input",
        "                                  |",
        "                                  v",
        "  Hodge positional encoder  [per bus/branch/cycle type]",
        f"    [N_type, F_type] -> {hodge_steps} diffusion steps -> "
        f"[N_type, {hodge_k}]",
        "                                  | concatenate PE",
        "                                  v",
        f"  Type encoders x{len(node_rows)}  [{norm_name}Linear]",
        f"    [N_type, F_type(+{hodge_k} where active)] -> "
        f"[N_type, {hidden_dim}]",
        "                                  |",
        "                                  v",
        f"+---------------- GridBlock x{n_blocks} ----------------+",
        f"| input/output: [N_type, {hidden_dim}]                 |",
        "|                                                     |",
        f"|  1. LayerNorm -> linear self-attention ({num_heads} heads) |",
        f"|     Q/K/V/O: {hidden_dim} -> {hidden_dim}             |",
        f"|     + residual: [N_type,{hidden_dim}] -> "
        f"[N_type,{hidden_dim}]       |",
        "|                                                     |",
        "|  2. LayerNorm -> heterogeneous message passing      |",
        f"|     {relation_count} active typed relations; signed incidence/GraphSAGE |",
        f"|     + residual: [N_type,{hidden_dim}] -> "
        f"[N_type,{hidden_dim}]       |",
        "|                                                     |",
        "|  3. LayerNorm -> FFN -> residual                    |",
        f"|     {hidden_dim} -> {ffn_hidden} -> {hidden_dim}; "
        f"[N_type,{hidden_dim}] -> [N_type,{hidden_dim}] |",
        "+-----------------------------------------------------+",
        "                                  |",
        "                                  v",
    ])

    n_bus = int(data["bus"].x.size(0))
    n_gen = int(data["generator"].x.size(0))
    lines.extend([
        "  Typed fusion + graph pooling",
        f"    type embeddings -> bus [{n_bus},{hidden_dim}]",
        f"                    -> generator [{n_gen},{hidden_dim}]",
        f"                    -> global [{n_graphs},{hidden_dim}]",
        "                                  |",
        "                    +-------------+-------------+",
        "                    |             |             |",
        "  Prediction heads",
        "  ----------------",
    ])

    head_specs = [
        ("theta", model.head_theta, n_bus),
        ("V", model.head_V, n_bus),
        ("Pg", model.head_Pg, n_gen),
        ("Qg", model.head_Qg, n_gen),
        ("feas", model.head_feas, n_graphs),
    ]
    for name, head, count in head_specs:
        input_width = int(head[0].in_features)
        mid_width = int(head[0].out_features)
        output_width = int(head[-1].out_features)
        lines.append(
            f"  {name:<5}: [{count},{input_width}] -> Linear/GELU "
            f"[{count},{mid_width}] -> Linear [{count},{output_width}]"
        )

    bus_pred = getattr(data["bus"], "pred", None)
    gen_pred = getattr(data["generator"], "pred", None)
    feas_logit = getattr(data, "feas_logit", None)
    lines.extend([
        "",
        "  Analytical post-processing (no learned layers)",
        "  ------------------------------------------------",
        "  theta head + DC prior -> slack/mean anchor",
        "  V/Pg/Qg -> physical-limit clipping",
        "  V + theta -> AC pi-model branch flows",
        "",
        "OUTPUT",
        "------",
    ])
    if isinstance(bus_pred, Tensor):
        lines.append(f"bus.pred        {_tensor_summary(bus_pred)}  columns=(theta, V)")
    if isinstance(gen_pred, Tensor):
        lines.append(f"generator.pred  {_tensor_summary(gen_pred)}  columns=(Pg, Qg)")
    for edge_type in (("bus", "ac_line", "bus"),
                      ("bus", "transformer", "bus")):
        if edge_type not in data.edge_types:
            continue
        flows = getattr(data[edge_type], "edge_flow_pred", None)
        if isinstance(flows, Tensor):
            lines.append(
                f"{edge_type[1] + '.flows':<15} {_tensor_summary(flows)}  "
                "columns=(Pij, Qij, Pji, Qji)"
            )
    if isinstance(feas_logit, Tensor):
        lines.append(f"feas_logit      {_tensor_summary(feas_logit)}")

    lines.extend([
        "",
        "N_type is the number of components of that type. Shapes above are",
        "from the supplied sample; xN denotes shared architecture replicated N times.",
    ])
    return "\n".join(lines) + "\n"


def print_module_dag(
    model: nn.Module,
    *model_args: Any,
    file: Optional[TextIO] = None,
    leaf_only: bool = False,
    **model_kwargs: Any,
) -> ModuleDagTrace:
    """Trace one forward pass, print its report, and return the raw trace."""
    trace = trace_module_dag(model, *model_args, **model_kwargs)
    print(format_module_dag(trace, leaf_only=leaf_only), end="", file=file)
    return trace
