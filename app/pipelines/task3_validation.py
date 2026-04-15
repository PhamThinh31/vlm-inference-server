# app/pipelines/task3_validation.py
"""
Task 3: Garment-Body Pair Validation Pipeline
Validates if garment-body pairs are suitable for VTON
"""

import asyncio
from typing import List, Dict, Optional, Tuple, Any
import time
import logging
from PIL import Image

from ..schemas import ErrorCode, ImageClassification, TaskError, ValidationResult
from ._runner import build_prompt_text, run_vlm
from .prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)

class Task3Validator:
    """Pipeline for Task 3: Pair validation"""

    def __init__(self, version: str = "v1.1"):
        """
        Initialize Task3Validator

        Args:
            version: Prompt version to use ('v1.0' or 'v1.1')
        """
        self.version = version
        self.prompt_template = PromptTemplates.get_task3_prompt(version)
    
    async def validate_pair(self,
                           garment_image: Image.Image,
                           body_image: Image.Image,
                           garment_id: str,
                           body_id: str,
                           use_cache: bool = True) -> ValidationResult:
        """
        Validate a single garment-body pair
        
        Args:
            garment_image: PIL Image of garment
            body_image: PIL Image of body
            garment_id: Garment image filename
            body_id: Body image filename
            use_cache: Whether to use vision encoding cache
            
        Returns:
            ValidationResult object
        """
        
        logger.info(f"Validating pair: {garment_id} + {body_id}")
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
            logger.info(f"Task 3 validation completed in {inference_time:.3f}s")
            
            # Parse response
            result = self._parse_response(response, garment_id, body_id)
            
            return result
            
        except Exception as e:
            logger.exception("Task 3 inference failed for (%s, %s)", garment_id, body_id)
            return ValidationResult(
                garment_file=garment_id,
                body_file=body_id,
                is_valid=False,
                confidence=None,
                reasoning=None,
                raw_response=None,
                parse_failed=True,
                error=TaskError(code=ErrorCode.INFERENCE_FAILURE, detail=str(e)),
            )
    
    async def validate_pairs_batch(self,
                                  pairs: List[Tuple[str, str]],
                                  images_dict: Dict[str, Image.Image],
                                  use_cache: bool = True,
                                  max_concurrent: int = 5) -> List[ValidationResult]:
        """
        Validate multiple pairs with concurrent processing
        
        Args:
            pairs: List of (garment_id, body_id) tuples
            images_dict: Dictionary mapping image IDs to PIL Images
            use_cache: Whether to use vision encoding cache
            max_concurrent: Maximum concurrent validations
            
        Returns:
            List of ValidationResult objects
        """
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def validate_with_semaphore(pair):
            async with semaphore:
                garment_id, body_id = pair
                garment_image = images_dict.get(garment_id)
                body_image = images_dict.get(body_id)
                
                if not garment_image or not body_image:
                    logger.warning("task3 missing images for pair (%s, %s)", garment_id, body_id)
                    return ValidationResult(
                        garment_file=garment_id,
                        body_file=body_id,
                        is_valid=False,
                        confidence=None,
                        reasoning=None,
                        parse_failed=True,
                        error=TaskError(code=ErrorCode.MISSING_IMAGE),
                    )

                return await self.validate_pair(
                    garment_image=garment_image,
                    body_image=body_image,
                    garment_id=garment_id,
                    body_id=body_id,
                    use_cache=use_cache,
                )

        # return_exceptions=True: a single model timeout on one pair should
        # not tear down the whole asyncio.gather and lose the other 19
        # successful validations. Failed pairs get a parse_failed=True row.
        raw = await asyncio.gather(
            *(validate_with_semaphore(p) for p in pairs), return_exceptions=True
        )
        results: List[ValidationResult] = []
        for pair, r in zip(pairs, raw):
            if isinstance(r, BaseException):
                logger.exception("task3 pair (%s, %s) failed: %s", pair[0], pair[1], r)
                results.append(ValidationResult(
                    garment_file=pair[0], body_file=pair[1],
                    is_valid=False, confidence=None,
                    reasoning=None,
                    parse_failed=True,
                    error=TaskError(code=ErrorCode.INFERENCE_FAILURE, detail=str(r)),
                ))
            else:
                results.append(r)
        return results
    
    def _parse_response(self, response: str, garment_id: str, body_id: str) -> ValidationResult:
        """Parse the model response into ValidationResult

        Supports both v1.0 and v1.1 formats:
        - v1.0: "Valid: Yes/No" with optional confidence and reasoning
        - v1.1: "Pose Quality: <score>\nValid Pair: Yes/No"
        """

        # Strict, field-anchored regex. The previous "does the word 'yes'
        # appear anywhere" fallback caused false positives whenever the
        # model hedged ("the garment would look yes-adjacent to ...") and
        # tripled the apparent valid_rate on ambiguous batches.
        #
        # We also robustly extract the first numeric token on a line, so
        # "Pose Quality: 85 / 100" and "Confidence: ~0.72" don't silently
        # hit the except branch.
        import re

        _num_re = re.compile(r"[-+]?\d*\.?\d+")

        def _first_float(line: str) -> Optional[float]:
            tail = line.split(":", 1)[1] if ":" in line else line
            m = _num_re.search(tail)
            return float(m.group(0)) if m else None

        is_valid: Optional[bool] = None  # tri-state: None means "model didn't say"
        confidence: Optional[float] = None
        reasoning = ""
        pose_quality: Optional[float] = None

        for raw in response.splitlines():
            low = raw.lower().strip()
            if not low:
                continue

            if low.startswith("pose quality:"):
                val = _first_float(raw)
                if val is not None:
                    pose_quality = max(0.0, min(100.0, val))

            elif low.startswith("valid pair:"):
                is_valid = "yes" in low.split(":", 1)[1]

            elif low.startswith("valid:"):
                is_valid = "yes" in low.split(":", 1)[1]

            elif low.startswith("confidence:"):
                val = _first_float(raw)
                if val is not None:
                    confidence = max(0.0, min(1.0, val))

            elif low.startswith("reasoning:"):
                reasoning = raw.split(":", 1)[1].strip()

        parse_failed = is_valid is None
        if parse_failed:
            logger.warning(
                "task3 parse failure for (%s, %s): no Valid/Valid Pair field found",
                garment_id, body_id,
            )
            is_valid = False

        # v1.1 normalises pose_quality (0-100) into the confidence slot.
        # If neither pose_quality nor an explicit confidence field is
        # present, confidence stays None — callers MUST treat None as
        # "unreported" rather than substituting a default, otherwise
        # downstream aggregations silently average fabricated values.
        if pose_quality is not None:
            confidence = pose_quality / 100.0

        return ValidationResult(
            garment_file=garment_id,
            body_file=body_id,
            is_valid=is_valid,
            confidence=confidence,
            reasoning=reasoning or None,
            raw_response=response,
            pose_quality=pose_quality,
            parse_failed=parse_failed,
            error=TaskError(code=ErrorCode.PARSE_FAILED) if parse_failed else None,
        )
    
    def generate_all_pairs(self,
                          classifications: List[ImageClassification],
                          max_pairs: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        Generate all possible garment-body pairs
        
        Args:
            classifications: List of image classifications from Task 1
            max_pairs: Maximum number of pairs to generate
            
        Returns:
            List of (garment_filename, body_filename) tuples
        """
        
        # Separate garments and bodies
        garments = [c for c in classifications if c.img_class == 'garment']
        bodies = [c for c in classifications if c.img_class == 'body']
        
        # Sort by priority to process best pairs first
        garments.sort(key=lambda x: x.priority_score or 0, reverse=True)
        bodies.sort(key=lambda x: x.priority_score or 0, reverse=True)
        
        # Generate all pairs
        pairs = []
        for garment in garments:
            for body in bodies:
                pairs.append((garment.filename, body.filename))
                
                if max_pairs and len(pairs) >= max_pairs:
                    break
            
            if max_pairs and len(pairs) >= max_pairs:
                break
        
        logger.info(f"Generated {len(pairs)} pairs from {len(garments)} garments and {len(bodies)} bodies")
        
        return pairs
    
    def filter_valid_pairs(self,
                          validations: List[ValidationResult],
                          min_confidence: float = 0.5) -> List[ValidationResult]:
        """Filter to only valid pairs above confidence threshold.

        Pairs with `confidence is None` are excluded — unknown confidence
        is not a pass. If you want to include them, set min_confidence=0
        AND branch explicitly.
        """
        return [
            v for v in validations
            if v.is_valid and v.confidence is not None and v.confidence >= min_confidence
        ]

    def sort_by_pose_quality(self,
                            validations: List[ValidationResult],
                            descending: bool = True) -> List[ValidationResult]:
        """
        Sort validation results by pose quality score (v1.1)

        Args:
            validations: List of validation results
            descending: Sort in descending order (highest scores first)

        Returns:
            Sorted list of validation results
        """
        # Separate results with and without pose_quality
        with_score = [v for v in validations if v.pose_quality is not None]
        without_score = [v for v in validations if v.pose_quality is None]

        # Sort those with scores
        with_score.sort(key=lambda x: x.pose_quality, reverse=descending)

        # Combine: scored items first, then unscored
        return with_score + without_score

    def get_best_valid_pairs(self,
                            validations: List[ValidationResult],
                            max_count: int = 10,
                            min_pose_quality: float = 40.0,
                            sort_by: str = "pose_quality") -> List[ValidationResult]:
        """
        Get best valid pairs sorted by quality

        Args:
            validations: List of validation results
            max_count: Maximum number of pairs to return
            min_pose_quality: Minimum pose quality score (0-100) for v1.1
            sort_by: Sorting criteria - "pose_quality" (v1.1) or "confidence" (v1.0)

        Returns:
            List of best validation results
        """
        # Filter to only valid pairs
        valid_pairs = [v for v in validations if v.is_valid]

        # Apply pose quality filter if using v1.1
        if sort_by == "pose_quality":
            valid_pairs = [
                v for v in valid_pairs
                if v.pose_quality is not None and v.pose_quality >= min_pose_quality
            ]

        # Sort based on criteria
        if sort_by == "pose_quality":
            # v1.1: Sort by pose quality score (0-100)
            valid_pairs.sort(key=lambda x: x.pose_quality or 0, reverse=True)
        else:
            # v1.0: Sort by confidence (0-1). None sinks to bottom.
            valid_pairs.sort(key=lambda x: x.confidence if x.confidence is not None else -1, reverse=True)

        # Return top N
        return valid_pairs[:max_count]

    def get_statistics(self, validations: List[ValidationResult]) -> dict:
        """
        Get statistics about validation results

        Returns:
            Dictionary with statistics
        """
        valid_pairs = [v for v in validations if v.is_valid]
        invalid_pairs = [v for v in validations if not v.is_valid]

        stats = {
            "total_pairs": len(validations),
            "valid_pairs": len(valid_pairs),
            "invalid_pairs": len(invalid_pairs),
            "valid_percentage": len(valid_pairs) / len(validations) * 100 if validations else 0,
        }

        # Add pose quality statistics if available (v1.1)
        pose_scores = [v.pose_quality for v in validations if v.pose_quality is not None]
        if pose_scores:
            stats["avg_pose_quality"] = sum(pose_scores) / len(pose_scores)
            stats["max_pose_quality"] = max(pose_scores)
            stats["min_pose_quality"] = min(pose_scores)
            stats["pose_quality_std"] = (
                sum((x - stats["avg_pose_quality"]) ** 2 for x in pose_scores) / len(pose_scores)
            ) ** 0.5

        # Add confidence statistics — exclude None so we don't average
        # "unreported" as zero and silently drag the mean down.
        confidences = [v.confidence for v in validations if v.confidence is not None]
        if confidences:
            stats["avg_confidence"] = sum(confidences) / len(confidences)
            stats["max_confidence"] = max(confidences)
            stats["min_confidence"] = min(confidences)

        return stats