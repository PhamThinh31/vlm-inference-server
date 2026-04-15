"""Shared VLM call composition.

Every task module did the same five steps before this module existed:
resize images, build the chat-template prompt, call engine.generate,
log timing, return the raw text. The duplication wasn't expensive but
it was load-bearing — three separate copies of the resize size, the
prompt construction, and the timing log meant any change had to land
in three files. That's the precise spot where copy-paste drift
introduces silent behavioural divergence.

This helper owns the composition. Task modules own only what is
genuinely task-specific: the prompt template, the result type, and
the parser.

The error-handling boundary stays in the task modules because each
produces a different typed result (Task1Response vs Task2Result vs
ValidationResult). Trying to share that too would force a generic
result envelope and re-introduce stringly-typed errors — exactly the
mid-level mistake we just spent two PRs killing.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from PIL import Image

from ..config import settings
from ..engine import engine
from ..utils.image_ops import resize_image


async def run_vlm(
    prompt_text: str,
    images: List[Image.Image],
    *,
    use_cache: bool = True,
    image_ids: Optional[List[str]] = None,
    max_side: Optional[int] = None,
) -> str:
    """Resize images, build the multimodal prompt, run inference.

    Returns the raw response text. Exceptions from `engine.generate`
    propagate; the caller decides whether a failure becomes a typed
    error row or aborts the whole batch.

    `max_side` defaults to settings.vlm_input_max_side. Override per
    task only if the task has a documented reason (e.g. Task 2 needs
    higher fidelity for fine attribute detection).
    """
    side = max_side or settings.vlm_input_max_side
    processed = [resize_image(img, max_size=(side, side)) for img in images]
    prompt = engine.build_prompt(
        messages=[{"role": "user", "content": prompt_text}],
        images=processed,
    )
    return await engine.generate(
        prompt=prompt,
        images=processed,
        image_ids=image_ids,
        use_cache=use_cache,
    )


def build_prompt_text(template: str, **kwargs: Any) -> str:
    """Format a prompt template, raising clearly on missing keys.

    `str.format` raises KeyError with just the missing key — useless
    in a stack trace 400 lines deep. We wrap so the exception names
    the template and the offending substitution.
    """
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        raise ValueError(
            f"prompt template missing substitution {exc.args[0]!r}; "
            f"available keys: {list(kwargs)}"
        ) from exc
