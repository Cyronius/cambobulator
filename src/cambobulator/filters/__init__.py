"""Filter registry.

To add a filter, subclass :class:`Filter` and decorate it with
:func:`register_filter`. Built-in filters live in this package. Third-party
packages can expose filters through the ``cambobulator.filters`` entry-point
group; :func:`load_plugins` imports them.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any

from cambobulator.filters.base import Filter, FrameContext

log = logging.getLogger(__name__)

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


_plugins_loaded = False


def load_plugins() -> None:
    """Import filters advertised by other installed packages."""
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    for ep in entry_points(group="cambobulator.filters"):
        try:
            ep.load()
        except Exception:  # a broken plugin must not take the camera down
            log.exception("Failed to load filter plugin %s", ep.name)


# Import built-ins so they register themselves.
from cambobulator.filters import basic, fade  # noqa: E402,F401

__all__ = [
    "Filter",
    "FrameContext",
    "available_filters",
    "create_filter",
    "get_filter_class",
    "load_plugins",
    "register_filter",
    "unregister_filter",
]
