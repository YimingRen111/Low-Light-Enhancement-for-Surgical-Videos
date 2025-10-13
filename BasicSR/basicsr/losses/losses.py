"""Compatibility wrapper for losses expected by some models.
This file re-exports common loss helper functions implemented elsewhere
in the `basicsr.losses` package (e.g., `gan_loss.py`).
"""
from .gan_loss import g_path_regularize, gradient_penalty_loss, r1_penalty

__all__ = ['g_path_regularize', 'gradient_penalty_loss', 'r1_penalty']
