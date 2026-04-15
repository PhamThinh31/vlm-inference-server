# app/utils/image_ops.py
"""
Image operation utilities.

Intentionally narrow: we only keep what the pipelines actually call.
Previously this module also shipped `calculate_image_hash`,
`batch_load_images`, `normalize_image`, and base64 helpers that no caller
in the repo imported. Dead code was removed to keep the surface honest.
"""

from typing import Tuple, Union

from PIL import Image


def resize_image(image: Image.Image, max_size: Union[int, Tuple[int, int]] = 1024) -> Image.Image:
    """Resize without mutating the input.

    Accepts either a single int (max dimension, symmetric) or a
    (max_w, max_h) tuple. Returns the original object when it already
    fits so we skip the copy on the hot path.
    """
    if isinstance(max_size, int):
        target = (max_size, max_size)
    else:
        target = max_size

    if image.width <= target[0] and image.height <= target[1]:
        return image

    img_copy = image.copy()
    img_copy.thumbnail(target, Image.Resampling.LANCZOS)
    return img_copy
