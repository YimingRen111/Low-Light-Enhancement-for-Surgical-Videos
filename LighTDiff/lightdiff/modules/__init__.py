"""Compatibility shim: re-export modules that some code expects under `lightdiff.modules`.
Currently re-exports `TemporalSE` from `lightdiff.models.temporal_se`.
"""
from importlib import import_module

try:
    # import the TemporalSE implementation from models
    mod = import_module('lightdiff.models.temporal_se')
    TemporalSE = getattr(mod, 'TemporalSE')
except Exception:
    # fallback: define a placeholder to raise a clear error if used
    TemporalSE = None

__all__ = ['TemporalSE']
