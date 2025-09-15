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

        # Load new quantum-inspired plugins
        try:
            from plugins.quantum_holographic.plugin import QuantumHolographicPlugin  # type: ignore
            plugins["quantum_holographic"] = QuantumHolographicPlugin
        except Exception:
            pass

        try:
            from plugins.temporal_multiplexing.plugin import TemporalMultiplexingPlugin  # type: ignore
            plugins["temporal_multiplexing"] = TemporalMultiplexingPlugin
        except Exception:
            pass

        try:
            from plugins.adaptive_routing.plugin import AdaptiveRoutingPlugin  # type: ignore
            plugins["adaptive_routing"] = AdaptiveRoutingPlugin
        except Exception:
            pass

        try:
            from plugins.predictive_cache.plugin import PredictiveCachePlugin  # type: ignore
            plugins["predictive_cache"] = PredictiveCachePlugin
        except Exception:
            pass

        try:
            from plugins.neural_architecture_search.plugin import NeuralArchitectureSearchPlugin  # type: ignore
            plugins["neural_architecture_search"] = NeuralArchitectureSearchPlugin
        except Exception:
            pass

        try:
            from plugins.quantum_superposition.plugin import QuantumSuperpositionPlugin  # type: ignore
            plugins["quantum_superposition"] = QuantumSuperpositionPlugin
        except Exception:
            pass

        # Also load existing extreme plugins
        try:
            from plugins.extreme_405b.plugin import Extreme405BSlicing  # type: ignore
            plugins["extreme_405b"] = Extreme405BSlicing
        except Exception:
            pass

        try:
            from plugins.ultra_slicing.plugin import UltraSlicing  # type: ignore
            plugins["ultra_slicing"] = UltraSlicing
        except Exception:
            pass

        # Load R-Tuning and Deep Thinking plugins
        try:
            from plugins.r_tuning.plugin import RTuningPlugin  # type: ignore
            plugins["r_tuning"] = RTuningPlugin
        except Exception:
            pass

        try:
            from plugins.deep_thinking.plugin import DeepThinkingPlugin  # type: ignore
            plugins["deep_thinking"] = DeepThinkingPlugin
        except Exception:
            pass

        # Load Memory Persistence plugin
        try:
            from plugins.memory_persistence.plugin import MemoryPersistencePlugin  # type: ignore
            plugins["memory_persistence"] = MemoryPersistencePlugin
        except Exception:
            pass

        # Load novel slicing techniques
        try:
            from plugins.novel_4d_slicing.plugin import Novel4DSlicingPlugin  # type: ignore
            plugins["novel_4d_slicing"] = Novel4DSlicingPlugin
        except Exception:
            pass

        try:
            from plugins.fractal_recursive_slicing.plugin import FractalRecursiveSlicingPlugin  # type: ignore
            plugins["fractal_recursive_slicing"] = FractalRecursiveSlicingPlugin
        except Exception:
            pass

        try:
            from plugins.neural_pathway_slicing.plugin import NeuralPathwaySlicingPlugin  # type: ignore
            plugins["neural_pathway_slicing"] = NeuralPathwaySlicingPlugin
        except Exception:
            pass

        try:
            from plugins.spectral_domain_slicing.plugin import SpectralDomainSlicingPlugin  # type: ignore
            plugins["spectral_domain_slicing"] = SpectralDomainSlicingPlugin
        except Exception:
            pass

        try:
            from plugins.topological_slicing.plugin import TopologicalSlicingPlugin  # type: ignore
            plugins["topological_slicing"] = TopologicalSlicingPlugin
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