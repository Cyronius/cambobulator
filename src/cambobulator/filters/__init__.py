"""Filter registry.

The app runs a single effect (``fade.FadeIntoBackground``); the registry maps
the name stored in the config file to its class.
"""

from __future__ import annotations

from typing import Any

from cambobulator.filters.base import Filter, FrameContext

_REGISTRY: dict[str, type[Filter]] = {}


def register_filter(cls: type[Filter]) -> type[Filter]:
    if not cls.NAME:
        raise ValueError(f"{cls.__name__} must set NAME")
    existing = _REGISTRY.get(cls.NAME)
    if existing is not None and existing is not cls:
        raise ValueError(f"A filter named {cls.NAME!r} is already registered ({existing.__name__})")
    _REGISTRY[cls.NAME] = cls
    return cls


def unregister_filter(name: str) -> None:
    _REGISTRY.pop(name, None)


def get_filter_class(name: str) -> type[Filter]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown filter {name!r}. Known filters: {', '.join(sorted(_REGISTRY))}") from None


def available_filters() -> list[dict[str, Any]]:
    return [
        {"type": cls.NAME, "label": cls.LABEL or cls.NAME, "description": cls.DESCRIPTION}
        for cls in sorted(_REGISTRY.values(), key=lambda c: c.LABEL or c.NAME)
    ]


def create_filter(name: str, segmenter=None, **params: Any) -> Filter:
    return get_filter_class(name)(segmenter=segmenter, **params)


# Import built-ins so they register themselves.
from cambobulator.filters import fade  # noqa: E402,F401

__all__ = [
    "Filter",
    "FrameContext",
    "available_filters",
    "create_filter",
    "get_filter_class",
    "register_filter",
    "unregister_filter",
]
