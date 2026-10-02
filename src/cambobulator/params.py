"""Tunable parameter metadata.

Filters describe their knobs with :class:`Param` objects. The UI reads this
metadata to build sliders, checkboxes and drop-downs without knowing anything
about a particular filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PARAM_KINDS = ("float", "int", "bool", "choice")


@dataclass(frozen=True)
class Param:
    name: str
    kind: str
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    label: str = ""
    help: str = ""
    choices: tuple[str, ...] = field(default_factory=tuple)
    group: str = ""  # the UI puts a heading above the first param of each group

    def __post_init__(self) -> None:
        if self.kind not in PARAM_KINDS:
            raise ValueError(f"Param {self.name!r}: unknown kind {self.kind!r}")
        if self.kind == "choice" and self.default not in self.choices:
            raise ValueError(f"Param {self.name!r}: default not in choices")

    def coerce(self, value: Any) -> Any:
        """Convert ``value`` to this param's type and clamp it to [min, max]."""
        if self.kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if self.kind == "choice":
            value = str(value)
            if value not in self.choices:
                raise ValueError(f"{self.name}: {value!r} is not one of {self.choices}")
            return value
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{self.name}: expected a number, got {value!r}") from None
        if number != number:  # NaN
            raise ValueError(f"{self.name}: NaN is not allowed")
        if self.min is not None:
            number = max(self.min, number)
        if self.max is not None:
            number = min(self.max, number)
        return int(round(number)) if self.kind == "int" else number

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "label": self.label or self.name,
            "help": self.help,
            "choices": list(self.choices),
            "group": self.group,
        }


@dataclass(frozen=True)
class Action:
    """A one-shot button a filter exposes, e.g. "capture background"."""

    name: str
    label: str = ""
    help: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "label": self.label or self.name, "help": self.help}
