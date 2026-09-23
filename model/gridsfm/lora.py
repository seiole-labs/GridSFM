"""Named, FFN-only LoRA adapters for GridSFM."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), "g").replace(".", "p")


@dataclass(frozen=True)
class LoRAConfig:
    """Configuration shared by every targeted linear layer in an adapter."""

    rank: int = 2
    alpha: float = 2.0
    target: str = "gridblock_ffn"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.rank, int)
            or isinstance(self.rank, bool)
            or self.rank <= 0
        ):
            raise ValueError(f"rank must be a positive integer, got {self.rank!r}")
        if not math.isfinite(float(self.alpha)) or self.alpha <= 0:
            raise ValueError(f"alpha must be finite and positive, got {self.alpha!r}")
        if self.target != "gridblock_ffn":
            raise ValueError(
                "only target='gridblock_ffn' is supported, "
                f"got {self.target!r}"
            )
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout!r}")

    @property
    def name(self) -> str:
        return f"{self.target}_r{self.rank}_a{_format_number(self.alpha)}"


class LoRAAdapter(nn.Module):
    """One zero-initialized low-rank update for a linear projection."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.rank = config.rank
        self.alpha = float(config.alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        self.lora_A = nn.Linear(
            in_features, self.rank, bias=False, device=device, dtype=dtype,
        )
        self.lora_B = nn.Linear(
            self.rank, out_features, bias=False, device=device, dtype=dtype,
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class LoRALinear(nn.Linear):
    """A linear layer that can apply one named LoRA update at a time."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.adapters = nn.ModuleDict()
        self._active_adapter: Optional[str] = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "LoRALinear":
        wrapped = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        wrapped.weight = linear.weight
        wrapped.bias = linear.bias
        wrapped.train(linear.training)
        return wrapped

    def add_adapter(self, name: str, config: LoRAConfig) -> None:
        if name in self.adapters:
            raise ValueError(f"adapter {name!r} already exists")
        self.adapters[name] = LoRAAdapter(
            self.in_features,
            self.out_features,
            config,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

    def set_adapter(self, name: Optional[str]) -> None:
        if name is not None and name not in self.adapters:
            raise KeyError(f"unknown adapter {name!r}")
        self._active_adapter = name

    def forward(self, x: Tensor) -> Tensor:
        result = F.linear(x, self.weight, self.bias)
        if self._active_adapter is not None:
            result = result + self.adapters[self._active_adapter](x)
        return result


class GridSFMLoRAModel(nn.Module):
    """Wrap a GridSFM backbone with named adapters on GridBlock FFNs only."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        self._configs: dict[str, LoRAConfig] = {}
        self._active_adapter: Optional[str] = None
        self._last_adapter: Optional[str] = None
        self._target_modules = self._wrap_ffn_linears()
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

    def _wrap_ffn_linears(self) -> tuple[str, ...]:
        targets: list[str] = []
        blocks = getattr(self.base_model, "blocks", None)
        if blocks is None:
            raise TypeError("base model has no GridSFM 'blocks' collection")

        for block_index, block in enumerate(blocks):
            for node_type, ffn in block.ffn.items():
                for projection_index in (0, 2):
                    linear = ffn[projection_index]
                    if not isinstance(linear, nn.Linear):
                        raise TypeError(
                            f"blocks.{block_index}.ffn.{node_type}.{projection_index} "
                            f"must be nn.Linear, got {type(linear).__name__}"
                        )
                    if not isinstance(linear, LoRALinear):
                        ffn[projection_index] = LoRALinear.from_linear(linear)
                    targets.append(
                        f"blocks.{block_index}.ffn.{node_type}.{projection_index}"
                    )
        return tuple(targets)

    def _lora_layers(self) -> Iterator[LoRALinear]:
        for block in self.base_model.blocks:
            for ffn in block.ffn.values():
                yield ffn[0]
                yield ffn[2]

    def _named_lora_layers(self) -> Iterator[tuple[str, LoRALinear]]:
        yield from zip(self._target_modules, self._lora_layers())

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str) or not name or "." in name:
            raise ValueError("adapter name must be non-empty and cannot contain '.'")

    @property
    def target_modules(self) -> tuple[str, ...]:
        return self._target_modules

    @property
    def active_adapter(self) -> Optional[str]:
        return self._active_adapter

    def add_adapter(self, config: LoRAConfig, name: Optional[str] = None) -> str:
        name = config.name if name is None else name
        self._validate_name(name)
        if name in self._configs:
            raise ValueError(f"adapter {name!r} already exists")
        for layer in self._lora_layers():
            layer.add_adapter(name, config)
        self._configs[name] = config
        self.set_adapter(name)
        return name

    def set_adapter(self, name: str) -> None:
        if name not in self._configs:
            raise KeyError(f"unknown adapter {name!r}")
        for layer in self._lora_layers():
            layer.set_adapter(name)
            for adapter_name, adapter in layer.adapters.items():
                adapter.requires_grad_(adapter_name == name)
        self._active_adapter = name
        self._last_adapter = name

    def enable_adapter(self, name: Optional[str] = None) -> None:
        selected = name if name is not None else self._last_adapter
        if selected is None:
            raise RuntimeError("no adapter has been added or selected")
        self.set_adapter(selected)

    def disable_adapter(self) -> None:
        for layer in self._lora_layers():
            layer.set_adapter(None)
        self._active_adapter = None

    def adapter_parameters(self, name: Optional[str] = None) -> Iterator[nn.Parameter]:
        selected = name if name is not None else self._active_adapter
        if selected is None or selected not in self._configs:
            raise KeyError(f"unknown adapter {selected!r}")
        for layer in self._lora_layers():
            yield from layer.adapters[selected].parameters()

    def save_adapter(self, name: str, path: str | Path) -> None:
        if name not in self._configs:
            raise KeyError(f"unknown adapter {name!r}")
        state_dict: dict[str, Tensor] = {}
        for target, layer in self._named_lora_layers():
            adapter = layer.adapters[name]
            state_dict[f"{target}.lora_A.weight"] = (
                adapter.lora_A.weight.detach().cpu().clone()
            )
            state_dict[f"{target}.lora_B.weight"] = (
                adapter.lora_B.weight.detach().cpu().clone()
            )

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": "gridsfm_lora_v1",
            "adapter_name": name,
            "config": asdict(self._configs[name]),
            "base_model": getattr(self.base_model, "_checkpoint_metadata", None),
            "target_modules": list(self._target_modules),
            "state_dict": state_dict,
        }, destination)

    def load_adapter(self, path: str | Path, name: Optional[str] = None) -> str:
        blob = torch.load(path, weights_only=True, map_location="cpu")
        if not isinstance(blob, dict) or blob.get("format") != "gridsfm_lora_v1":
            raise ValueError(f"{path}: not a GridSFM LoRA adapter checkpoint")
        try:
            config = LoRAConfig(**blob["config"])
            stored_name = blob["adapter_name"]
            stored_targets = blob["target_modules"]
            state_dict = blob["state_dict"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path}: malformed adapter checkpoint") from exc

        runtime_name = stored_name if name is None else name
        self._validate_name(runtime_name)
        if runtime_name in self._configs:
            raise ValueError(f"adapter {runtime_name!r} already exists")
        if list(self._target_modules) != stored_targets:
            raise RuntimeError(f"{path}: FFN target modules do not match the base model")

        expected: dict[str, tuple[int, ...]] = {}
        for target, layer in self._named_lora_layers():
            expected[f"{target}.lora_A.weight"] = (config.rank, layer.in_features)
            expected[f"{target}.lora_B.weight"] = (layer.out_features, config.rank)
        if set(state_dict) != set(expected):
            missing = sorted(set(expected) - set(state_dict))
            unexpected = sorted(set(state_dict) - set(expected))
            raise RuntimeError(
                f"{path}: adapter tensor keys mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for key, shape in expected.items():
            value = state_dict[key]
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                actual = (
                    tuple(value.shape)
                    if torch.is_tensor(value)
                    else type(value).__name__
                )
                raise RuntimeError(
                    f"{path}: {key} has shape/type {actual}, expected {shape}"
                )

        saved_base = blob.get("base_model")
        current_base = getattr(self.base_model, "_checkpoint_metadata", None)
        if (
            saved_base
            and current_base
            and saved_base.get("hash") != current_base.get("hash")
        ):
            raise RuntimeError(f"{path}: adapter base checkpoint hash does not match")

        self.add_adapter(config, name=runtime_name)
        with torch.no_grad():
            for target, layer in self._named_lora_layers():
                adapter = layer.adapters[runtime_name]
                for projection_name, parameter in (
                    ("lora_A", adapter.lora_A.weight),
                    ("lora_B", adapter.lora_B.weight),
                ):
                    key = f"{target}.{projection_name}.weight"
                    parameter.copy_(state_dict[key].to(
                        device=parameter.device, dtype=parameter.dtype,
                    ))
        return runtime_name

    def trainable_parameter_summary(self) -> dict[str, object]:
        return {
            "active_adapter": self._active_adapter,
            "target_modules": len(self._target_modules),
            "trainable_params": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "total_params": sum(parameter.numel() for parameter in self.parameters()),
        }

    def forward(self, data):
        return self.base_model(data)


__all__ = ["GridSFMLoRAModel", "LoRAAdapter", "LoRAConfig", "LoRALinear"]
