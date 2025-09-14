from __future__ import annotations

from importlib.metadata import entry_points
from typing import Dict, Type

from .interfaces import SlicingPlugin, ContinualPlugin


def _load_from_entry_points(group: str) -> Dict[str, type]:
    try:
        eps = entry_points(group=group)
        return {ep.name: ep.load() for ep in eps}  # type: ignore[no-any-return]
    except Exception:
        return {}


def load_slicing_plugins() -> Dict[str, Type[SlicingPlugin]]:
    """Return available slicing plugins.

    First tries to load via package entry points. If none found (e.g., package not
    installed), falls back to built-in plugins in the `plugins` package.
    """
    plugins: Dict[str, Type[SlicingPlugin]] = {}
    # Attempt to load via entry points
    for name, cls in _load_from_entry_points("dynai.slicing").items():
        plugins[name] = cls
    # Fallback to built-in
    if not plugins:
        try:
            from plugins.slicing_windowed.plugin import WindowedSlicing  # type: ignore

            plugins["windowed"] = WindowedSlicing
        except Exception:
            pass
    return plugins


def load_continual_plugins() -> Dict[str, Type[ContinualPlugin]]:
    """Return available continual learning plugins.

    First tries to load via package entry points. If none found (e.g., package not
    installed), falls back to built-in plugins in the `plugins` package.
    """
    plugins: Dict[str, Type[ContinualPlugin]] = {}
    for name, cls in _load_from_entry_points("dynai.continual").items():
        plugins[name] = cls
    if not plugins:
        try:
            from plugins.continual_kd_head.plugin import KDHeadContinual  # type: ignore

            plugins["kd_head"] = KDHeadContinual
        except Exception:
            pass
    return plugins