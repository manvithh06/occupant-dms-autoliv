"""
utils/image.py — Shared image-processing helpers.

Previously ``normalize_frame`` was copy-pasted in both ``main.py`` and
``demo_recorder.py``.  It now lives here as the single authoritative copy.
"""

from __future__ import annotations

import cv2
import numpy as np


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE-based lighting normalization to a BGR frame.

    Converts to LAB colour space, equalises the Luminance channel with
    Contrast-Limited Adaptive Histogram Equalisation (CLAHE), then
    converts back to BGR.  This improves face/eye detection reliability
    under dim or uneven cabin lighting.

    Args:
        frame: Input BGR image as a NumPy array (H×W×3, uint8).

    Returns:
        Normalized BGR image of the same shape and dtype.
    """
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    lab     = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    """
    Resize *frame* to *width* pixels, preserving aspect ratio.

    Args:
        frame: Input BGR image.
        width: Target width in pixels.

    Returns:
        Resized BGR image.
    """
    h, w = frame.shape[:2]
    scale  = width / w
    new_h  = int(h * scale)
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_LINEAR)


def overlay_alpha(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blend *overlay* onto *base* with the given *alpha* (0–1).

    Equivalent to ``cv2.addWeighted`` but expressed clearly.

    Args:
        base:    Background BGR image.
        overlay: Foreground BGR image (same shape as *base*).
        alpha:   Opacity of *overlay* (0 = invisible, 1 = opaque).

    Returns:
        Blended image (same shape as *base*).
    """
    return cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)
