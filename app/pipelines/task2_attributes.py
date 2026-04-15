# app/pipelines/task2_attributes.py
"""
Task 2: Garment Attributes Extraction Pipeline
Extracts detailed attributes from garment-body pairs
"""

import asyncio
from typing import List, Dict, Optional, Tuple
import json
import re
import time
import logging
from PIL import Image

from ..schemas import ErrorCode, ImageClassification, Task2Result, TaskError
from ._runner import build_prompt_text, run_vlm
from .prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)

class Task2AttributesExtractor:
    """Pipeline for Task 2: Garment attributes extraction"""

    def __init__(self, version: str = "v1.1", use_alt: bool = False):
        """
        Initialize Task2AttributesExtractor

        Args:
            version: Prompt version to use ('v1.0' or 'v1.1')
            use_alt: Use alternative format (without description) for v1.0
        """
        self.version = version
        self.use_alt = use_alt
        self.prompt_template = PromptTemplates.get_task2_prompt(version, use_alt=use_alt)
    
    async def extract_attributes(self,
                                garment_image: Image.Image,
                                body_image: Image.Image,
                                garment_id: str,
                                body_id: str,
                                use_cache: bool = True) -> Task2Result:
        """
        Extract attributes for a single garment-body pair
        
        Args:
            garment_image: PIL Image of garment
            body_image: PIL Image of body wearing garment
            garment_id: Garment image filename
            body_id: Body image filename
            use_cache: Whether to use vision encoding cache
            
        Returns:
            Task2Result object
        """
        
        logger.info(f"Extracting attributes for pair: {garment_id} + {body_id}")
        start_time = time.perf_counter()
        
        try:
            prompt_text = build_prompt_text(
                self.prompt_template,
                garment_file=garment_id,
                body_file=body_id,
            )
            response = await run_vlm(
                prompt_text=prompt_text,
                images=[garment_image, body_image],
                use_cache=use_cache,
                image_ids=[garment_id, body_id],
            )
            
            inference_time = time.perf_counter() - start_time
            logger.info(f"Task 2 inference completed in {inference_time:.3f}s")
            
            # Parse response
            result = self._parse_response(response, garment_id, body_id)
            
            return result
            
        except Exception as e:
            logger.exception("Task 2 inference failed for (%s, %s)", garment_id, body_id)
            # Explicit failure: category=None + error message, never a fake
            # "unknown" category that collides with legitimate unknowns.
            return Task2Result(
                garment_file=garment_id,
                body_file=body_id,
                category=None,
                description=None,
                attributes={},
                error=TaskError(code=ErrorCode.INFERENCE_FAILURE, detail=str(e)),
                raw_response=None,
            )
    
    async def extract_attributes_batch(self,
                                      pairs: List[Tuple[str, str]],
                                      images_dict: Dict[str, Image.Image],
                                      use_cache: bool = True,
                                      max_concurrent: int = 5) -> List[Task2Result]:
        """
        Extract attributes for multiple pairs concurrently.

        Concurrency is bounded by `max_concurrent` so vLLM's continuous
        batching can fuse requests without oversubscribing the scheduler.
        Order is preserved (gather preserves input order) and missing
        images yield a placeholder result instead of being dropped — the
        silent `continue` in the old impl made downstream counts lie.
        """

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _run(pair: Tuple[str, str]) -> Optional[Task2Result]:
            garment_id, body_id = pair
            garment_image = images_dict.get(garment_id)
            body_image = images_dict.get(body_id)

            if garment_image is None or body_image is None:
                logger.warning("task2 missing images for pair (%s, %s)", garment_id, body_id)
                return Task2Result(
                    garment_file=garment_id,
                    body_file=body_id,
                    category=None,
                    description=None,
                    attributes={},
                    error=TaskError(code=ErrorCode.MISSING_IMAGE),
                )

            async with semaphore:
                return await self.extract_attributes(
                    garment_image=garment_image,
                    body_image=body_image,
                    garment_id=garment_id,
                    body_id=body_id,
                    use_cache=use_cache,
                )

        return await asyncio.gather(*(_run(p) for p in pairs))
    
    def _parse_response(self, response: str, garment_id: str, body_id: str) -> Task2Result:
        """Parse the model response into Task2Result.

        Required fields: Category, JSON Attributes. If either is missing,
        the result is returned with `error="parse_failed"` and
        category/description set to None — never defaulted to "unknown".
        """
        cat_match = re.search(
            r'\*\*Category:\*\*\s*(.*?)\s*(?=\n\*\*|$)',
            response, re.DOTALL | re.IGNORECASE,
        )
        desc_match = re.search(
            r'\*\*Attribute Description:\*\*\s*(.*?)\s*(?=\n\*\*|$)',
            response, re.DOTALL | re.IGNORECASE,
        )
        json_section = re.search(
            r'\*\*JSON Attributes:\*\*\s*(.*)',
            response, re.DOTALL | re.IGNORECASE,
        )

        attributes_dict: Dict = {}
        json_parse_error: Optional[str] = None
        if json_section:
            json_match = re.search(r'(\{.*\})', json_section.group(1), re.DOTALL)
            if json_match:
                try:
                    attributes_dict = json.loads(json_match.group(1))
                except json.JSONDecodeError as exc:
                    json_parse_error = f"json_decode: {exc}"

        category = cat_match.group(1).strip() if cat_match else None
        description = desc_match.group(1).strip() if desc_match else None

        missing = []
        if category is None:
            missing.append("category")
        if not attributes_dict:
            missing.append("attributes")
        error: Optional[TaskError] = None
        if missing or json_parse_error:
            parts = missing + ([json_parse_error] if json_parse_error else [])
            error = TaskError(code=ErrorCode.PARSE_FAILED, detail=", ".join(parts))
            logger.warning("task2 parse failure for (%s, %s): %s", garment_id, body_id, error.detail)

        return Task2Result(
            garment_file=garment_id,
            body_file=body_id,
            category=category,
            description=description,
            attributes=attributes_dict,
            raw_response=response or None,
            error=error,
        )

    def select_best_pairs(self,
                         classifications: List[ImageClassification],
                         max_pairs: int = 5) -> List[Tuple[str, str]]:
        """
        Select best garment-body pairs based on classifications
        
        Args:
            classifications: List of image classifications from Task 1
            max_pairs: Maximum number of pairs to return
            
        Returns:
            List of (garment_filename, body_filename) tuples
        """
        
        # Separate garments and bodies
        garments = []
        bodies = []
        
        for cls in classifications:
            if cls.img_class == 'garment':
                garments.append(cls)
            elif cls.img_class == 'body':
                bodies.append(cls)
        
        # Sort by priority score
        garments.sort(key=lambda x: x.priority_score or 0, reverse=True)
        bodies.sort(key=lambda x: x.priority_score or 0, reverse=True)
        
        # Create pairs
        pairs = []
        num_pairs = min(len(garments), len(bodies), max_pairs)
        
        for i in range(num_pairs):
            pairs.append((garments[i].filename, bodies[i].filename))
        
        logger.info(f"Selected {len(pairs)} best pairs from {len(garments)} garments and {len(bodies)} bodies")
        
        return pairs