from .background import is_tissue
from .stain_norm import MacenkoNormalizer

__all__ = ["TilePipeline", "tile_cohort", "is_tissue", "MacenkoNormalizer"]


def __getattr__(name):
    if name in {"TilePipeline", "tile_cohort"}:
        from .tiling import TilePipeline, tile_cohort

        return {"TilePipeline": TilePipeline, "tile_cohort": tile_cohort}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
