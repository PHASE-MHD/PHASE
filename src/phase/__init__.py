"""PHASE package."""

__version__ = "0.1.0"

__all__ = ["PHASEPipeline"]


def __getattr__(name):
    if name == "PHASEPipeline":
        from .hub import PHASEPipeline

        return PHASEPipeline
    raise AttributeError(name)
