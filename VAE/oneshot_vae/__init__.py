"""One-shot full-SNP masked VAE tooling."""

from .data import GenotypeMemmapMeta, OneShotMemmapDataset, compute_observed_fraction, load_meta

__all__ = [
    "GenotypeMemmapMeta",
    "OneShotMemmapDataset",
    "compute_observed_fraction",
    "load_meta",
]

try:  # pragma: no cover - import depends on torch availability
    from .model import OneShotMaskedVAE, OneShotMaskedVAEConfig

    __all__.extend(["OneShotMaskedVAE", "OneShotMaskedVAEConfig"])
except ImportError:
    pass
