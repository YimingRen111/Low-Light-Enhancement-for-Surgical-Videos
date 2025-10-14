"""LPIPS metric integration for BasicSR.

This module provides a registry-registered function that computes the Learned
Perceptual Image Patch Similarity (LPIPS) between two images. It mirrors the
behaviour of other metrics in the project by accepting numpy arrays in HWC/CHW
format and supports optional cropping prior to evaluation.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
import torch

from basicsr.metrics.metric_util import reorder_image
from basicsr.utils.registry import METRIC_REGISTRY

# Cache LPIPS models so they are instantiated only once per device/net combo.
_LPIPS_MODEL_CACHE: Dict[Tuple[str, str], torch.nn.Module] = {}


def _get_lpips_model(net: str, device: torch.device) -> torch.nn.Module:
    """Create (or fetch) a cached LPIPS model on the requested device."""
    cache_key = (net, str(device))
    if cache_key not in _LPIPS_MODEL_CACHE:
        try:
            import lpips  # type: ignore
        except ImportError as exc:  # pragma: no cover - mirrors other metrics' behaviour
            raise ImportError('Please install lpips: pip install lpips') from exc

        model = lpips.LPIPS(net=net)
        model = model.to(device)
        model.eval()
        _LPIPS_MODEL_CACHE[cache_key] = model
    return _LPIPS_MODEL_CACHE[cache_key]


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert an image in [0, 255] HWC format to a LPIPS-ready tensor."""
    if img.ndim == 2:  # allow grayscale by expanding the channel dimension
        img = np.expand_dims(img, axis=2)
    assert img.ndim == 3, f'Expected HWC image, but received shape {img.shape}.'
    tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()  # CHW
    tensor = tensor.unsqueeze(0)  # BCHW
    tensor = tensor / 127.5 - 1.0  # normalize to [-1, 1]
    return tensor



def _normalize_device(device: Union[None, str, torch.device]) -> Union[None, torch.device]:
    """Return a ``torch.device`` (or ``None``) given a mixed input type."""

    if device is None:
        return None
    if isinstance(device, torch.device):
        return device
    return torch.device(device)

@METRIC_REGISTRY.register()
def calculate_lpips(
    img: np.ndarray,
    img2: np.ndarray,
    crop_border: int = 0,
    input_order: str = 'HWC',
    use_gpu: bool = True,
    net: str = 'alex',
    device: Union[None, str, torch.device] = None,
    model: Union[None, torch.nn.Module] = None,
    **_: Union[str, int, float, bool],
) -> float:
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels
            are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
        use_gpu (bool): Whether to use CUDA when available. Default: True.
        net (str): Which LPIPS network backbone to use ("alex", "vgg", etc.).
            Default: "alex".

    Returns:
        float: LPIPS distance between ``img`` and ``img2``.
    """

    if img.shape != img2.shape:
        raise ValueError(f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW".'
        )

    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Convert to tensors and push to the desired device.
    tensor_a = _to_tensor(img)
    tensor_b = _to_tensor(img2)

    resolved_device = _normalize_device(device)
    model_device: Union[None, torch.device] = None
    if model is not None:
        try:
            model_device = next(model.parameters()).device
        except StopIteration:  # pragma: no cover - LPIPS has parameters
            model_device = torch.device('cpu')

    if resolved_device is None:
        if model_device is not None:
            resolved_device = model_device
        elif use_gpu and torch.cuda.is_available():
            resolved_device = torch.device('cuda')
        else:
            resolved_device = torch.device('cpu')

    tensor_a = tensor_a.to(resolved_device)
    tensor_b = tensor_b.to(resolved_device)

    if model is None:
        model = _get_lpips_model(net=net, device=resolved_device)
    else:
        if model_device is not None and model_device != resolved_device:
            model = model.to(resolved_device)
        model_device = resolved_device

    model.eval()

    with torch.no_grad():
        distance = model(tensor_a, tensor_b)
    return float(distance.item())


@METRIC_REGISTRY.register()
def calculate_lpips_lol(
    img: np.ndarray,
    img2: np.ndarray,
    device: Union[None, str, torch.device] = None,
    model: Union[None, torch.nn.Module] = None,
    **kwargs,
) -> float:
    """LightDiff-compatible wrapper around :func:`calculate_lpips`.

    LightDiff stores a pre-constructed LPIPS network inside the model and passes
    it (together with the expected device) through ``calculate_metric``.  This
    wrapper reuses that instance when provided, avoiding duplicate GPU memory
    usage while still supporting the cached fallback used in BasicSR.
    """

    return calculate_lpips(
        img=img,
        img2=img2,
        device=device,
        model=model,
        **kwargs,
    )
