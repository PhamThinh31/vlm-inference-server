# app/pipelines/gis_processor.py
"""
GIS Pipeline Processor - Orchestrates all three tasks for folder processing
"""

from typing import Any, Dict, List, Optional
import time
import logging

from .task1_classifier import Task1Classifier
from .task2_attributes import Task2AttributesExtractor
from .task3_validation import Task3Validator
from ..config import settings
from ..schemas import (
    ImageClassification,
    Task2Result,
    ValidationResult,
    GISPipelineResponse,
)
from ..session import Session, session_manager

logger = logging.getLogger(__name__)

class GISProcessor:
    """Orchestrates the complete GIS pipeline"""
    
    def __init__(self) -> None:
        # batch_size is sourced from settings so ops can tune without a
        # code change. The hardcoded 5 we used to ship was a relic of the
        # original token-overflow incident; see config.task1_batch_size.
        self.task1 = Task1Classifier(version="v1.0", batch_size=settings.task1_batch_size)
        self.task2 = Task2AttributesExtractor(version="v1.1")
        self.task3 = Task3Validator(version="v1.1")
    
    async def process_folder(self,
                            session: Session,
                            max_pairs_task2: int = 5,
                            max_pairs_task3: int = 10,
                            use_cache: bool = True,
                            priority_weights: Optional[Dict[str, float]] = None) -> GISPipelineResponse:
        """
        Process all images in a session through the complete pipeline
        
        Args:
            session: Session object containing images
            max_pairs_task2: Maximum pairs for Task 2
            max_pairs_task3: Maximum pairs for Task 3
            use_cache: Whether to use vision encoding cache
            priority_weights: Custom weights for priority calculation
            
        Returns:
            GISPipelineResponse with all results
        """
        
        logger.info(f"Starting GIS pipeline for session {session.session_id}")
        pipeline_start = time.perf_counter()

        # Snapshot real cache stats before/after (replaces the old
        # fabricated counter that just multiplied pair counts).
        from ..engine import engine as _engine
        stats_before = await _engine.get_cache_stats()

        # Reset name mapper for fresh inference
        session.name_mapper.reset()

        # Use the SessionManager's bounded-concurrency decoder instead of
        # opening images inline on the event loop. The previous loop did a
        # blocking PIL.Image.open per image in async context; at 12+ images
        # that serialised disk I/O that should have been parallel.
        images_by_simple, image_ids = await session_manager.get_images_for_inference(
            session_id=session.session_id,
            use_simple_names=True,
        )

        # Restore the original-name view the rest of this function relies on
        # for Task 2/3 remapping, without re-decoding.
        images: List[Any] = []
        original_names: List[str] = []
        images_dict: Dict[str, Any] = {}
        for simple, img in zip(image_ids, images_by_simple):
            original = session.name_mapper.get_original_name(simple) or simple
            images.append(img)
            original_names.append(original)
            images_dict[original] = img

        logger.info("processing %d images through GIS pipeline", len(images))
        logger.debug("name mappings: %s", session.name_mapper.original_to_simple)

        # Task 1: Classification with simple names
        task1_start = time.perf_counter()

        classifications, response = await self.task1.classify_images(
            images=images,
            image_ids=image_ids,
            use_cache=use_cache,
        )
        
        # Remap classifications back to original names
        for cls in classifications:
            original_name = session.name_mapper.get_original_name(cls.filename)
            if original_name:
                cls.simple_name = cls.filename  # Store simple name for reference
                cls.filename = original_name  # Use original name in results
        
        task1_time = time.perf_counter() - task1_start
        logger.info(f"Task 1 completed in {task1_time:.3f}s")
        
        # Store classifications in session with original names
        for cls in classifications:
            if cls.filename in session.images:
                session.images[cls.filename].classification = cls.dict()
        
        # Task 2: Garment Attributes - use original names for pairs
        task2_start = time.perf_counter()

        # Drop parse-failed rows before pairing. A placeholder UNKNOWN row
        # with parse_failed=True doesn't carry a real class judgement, so
        # feeding it into select_best_pairs would fabricate pairs from
        # undetermined images and inflate task3 validation_rate.
        reliable_classifications = [c for c in classifications if not c.parse_failed]
        if len(reliable_classifications) < len(classifications):
            logger.warning(
                "dropping %d parse_failed rows before pair selection",
                len(classifications) - len(reliable_classifications),
            )

        task2_pairs = self.task2.select_best_pairs(
            classifications=reliable_classifications,
            max_pairs=max_pairs_task2,
        )
        
        logger.info(f"Selected {len(task2_pairs)} pairs for Task 2")
        
        # For Task 2 inference, temporarily map to simple names
        task2_pairs_simple = []
        for garment_orig, body_orig in task2_pairs:
            garment_simple = session.name_mapper.get_simple_name(garment_orig)
            body_simple = session.name_mapper.get_simple_name(body_orig)
            if garment_simple and body_simple:
                task2_pairs_simple.append((garment_simple, body_simple))
        
        # Create images_dict with simple names for task2
        images_dict_simple = {}
        for orig_name, img in images_dict.items():
            simple_name = session.name_mapper.get_simple_name(orig_name)
            if simple_name:
                images_dict_simple[simple_name] = img
        
        # Run Task 2 with simple names
        task2_results_simple = await self.task2.extract_attributes_batch(
            pairs=task2_pairs_simple,
            images_dict=images_dict_simple,
            use_cache=use_cache
        )
        
        # Remap Task 2 results back to original names
        task2_results = []
        for result in task2_results_simple:
            garment_orig = session.name_mapper.get_original_name(result.garment_file)
            body_orig = session.name_mapper.get_original_name(result.body_file)
            if garment_orig and body_orig:
                result.garment_file = garment_orig
                result.body_file = body_orig
            task2_results.append(result)
        
        task2_time = time.perf_counter() - task2_start
        logger.info(f"Task 2 completed in {task2_time:.3f}s")
        
        # Task 3: Pair Validation
        logger.info("Starting Task 3: Pair Validation")
        task3_start = time.perf_counter()
        
        # Generate pairs for Task 3 based on priority (original names only)
        # Same parse_failed filter as Task 2 — see comment above.
        task3_pairs = self.task3.generate_all_pairs(
            classifications=reliable_classifications,
            max_pairs=max_pairs_task3,
        )
        
        logger.info(f"Validating {len(task3_pairs)} pairs for Task 3")
        
        # Convert to simple names for inference
        task3_pairs_simple = []
        for garment_orig, body_orig in task3_pairs:
            garment_simple = session.name_mapper.get_simple_name(garment_orig)
            body_simple = session.name_mapper.get_simple_name(body_orig)
            if garment_simple and body_simple:
                task3_pairs_simple.append((garment_simple, body_simple))
        
        # Run Task 3 with simple names
        task3_results_simple = await self.task3.validate_pairs_batch(
            pairs=task3_pairs_simple,
            images_dict=images_dict_simple,
            use_cache=use_cache,
            max_concurrent=5
        )
        
        # Remap Task 3 results back to original names
        task3_results = []
        for result in task3_results_simple:
            garment_orig = session.name_mapper.get_original_name(result.garment_file)
            body_orig = session.name_mapper.get_original_name(result.body_file)
            if garment_orig and body_orig:
                result.garment_file = garment_orig
                result.body_file = body_orig
            task3_results.append(result)
        
        task3_time = time.perf_counter() - task3_start
        logger.info(f"Task 3 completed in {task3_time:.3f}s")
        
        # Calculate summary statistics (using original names)
        total_time = time.perf_counter() - pipeline_start
        stats_after = await _engine.get_cache_stats()
        
        summary = self._generate_summary(
            classifications=classifications,  # Has original names
            task2_results=task2_results,  # Has original names
            task3_results=task3_results,  # Has original names
            task_times={
                'task1': task1_time,
                'task2': task2_time,
                'task3': task3_time,
                'total': total_time
            }
        )
        
        # Add name mapping info to summary
        summary['name_mapping'] = {
            'total_mappings': len(session.name_mapper.original_to_simple),
            'sample_mappings': dict(list(session.name_mapper.original_to_simple.items())[:5])  # Show first 5
        }
        # Real prefix-cache numbers sourced from vLLM, not fabricated.
        summary['cache_stats'] = {
            'before': stats_before,
            'after': stats_after,
        }

        # Surface vLLM's prefix-cache hit rate verbatim. None means the
        # engine didn't expose it — never a synthesised 0, which would
        # be indistinguishable from a real cache miss in dashboards.
        raw_rate = (stats_after or {}).get('gpu_prefix_cache_hit_rate')
        cache_hit_rate = float(raw_rate) if isinstance(raw_rate, (int, float)) else None

        response = GISPipelineResponse(
            session_id=session.session_id,
            task1_results=classifications,  # Original names
            task2_results=task2_results,  # Original names
            task3_results=task3_results,  # Original names
            summary=summary,
            total_processing_time=total_time,
            cache_hit_rate=cache_hit_rate,
        )

        # Free the decoded PIL handles now that we're done. Without this
        # they live in session.images until the 24h TTL.
        freed = await session_manager.evict_decoded_images(session.session_id)
        logger.info(
            "GIS pipeline completed in %.3fs (prefix hit-rate: %s, evicted=%d)",
            total_time, cache_hit_rate, freed,
        )
        return response
    
    def _generate_summary(self,
                         classifications: List[ImageClassification],
                         task2_results: List[Task2Result],
                         task3_results: List[ValidationResult],
                         task_times: Dict[str, float]) -> Dict[str, Any]:
        """Generate summary statistics"""
        
        # Count by class
        class_counts = {
            'garment': 0,
            'body': 0,
            'unknown': 0
        }
        
        for cls in classifications:
            class_counts[cls.img_class] += 1
        
        # Validation stats — exclude parse_failed rows from validation_rate.
        # They are not model judgements; counting them as invalid pairs
        # would understate quality; counting as valid would inflate it.
        clean_task3 = [v for v in task3_results if not v.parse_failed]
        valid_pairs = [v for v in clean_task3 if v.is_valid]
        task1_parse_failed = sum(1 for c in classifications if c.parse_failed)
        task3_parse_failed = sum(1 for v in task3_results if v.parse_failed)
        
        # Garment categories from Task 2
        categories = {}
        for result in task2_results:
            if result.category not in categories:
                categories[result.category] = 0
            categories[result.category] += 1
        
        # Best matches: valid pairs with a reported confidence, sorted
        # high-to-low. Pairs without a confidence are excluded from the
        # leaderboard rather than ranked alongside as 0.0.
        best_matches = sorted(
            [v for v in task3_results if v.is_valid and v.confidence is not None],
            key=lambda x: x.confidence,
            reverse=True,
        )[:5]
        
        summary = {
            'image_statistics': {
                'total': len(classifications),
                'garments': class_counts['garment'],
                'bodies': class_counts['body'],
                'unknown': class_counts['unknown']
            },
            'task2_statistics': {
                'pairs_analyzed': len(task2_results),
                'categories': categories,
                'unique_categories': len(categories)
            },
            'task3_statistics': {
                'pairs_validated': len(task3_results),
                'valid_pairs': len(valid_pairs),
                'invalid_pairs': len(clean_task3) - len(valid_pairs),
                # Rate is over *parseable* rows only; see clean_task3.
                'validation_rate': (len(valid_pairs) / len(clean_task3) * 100) if clean_task3 else 0,
                'parse_failed_pairs': task3_parse_failed,
            },
            'data_quality': {
                'task1_parse_failed': task1_parse_failed,
                'task3_parse_failed': task3_parse_failed,
            },
            'best_matches': [
                {
                    'garment': m.garment_file,
                    'body': m.body_file,
                    'confidence': m.confidence
                }
                for m in best_matches
            ],
            'performance': {
                'task1_time': f"{task_times['task1']:.3f}s",
                'task2_time': f"{task_times['task2']:.3f}s",
                'task3_time': f"{task_times['task3']:.3f}s",
                'total_time': f"{task_times['total']:.3f}s",
                'avg_time_per_image': f"{task_times['total'] / len(classifications):.3f}s" if classifications else "N/A"
            }
        }
        
        return summary
    
