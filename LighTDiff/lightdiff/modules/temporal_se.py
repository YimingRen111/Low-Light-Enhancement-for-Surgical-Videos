"""Compatibility shim for temporal_se under lightdiff.modules
This imports the implementation from lightdiff.models.temporal_se.
"""
from importlib import import_module

mod = import_module('lightdiff.models.temporal_se')
TemporalSE = getattr(mod, 'TemporalSE')

__all__ = ['TemporalSE']
