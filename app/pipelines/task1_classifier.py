# app/pipelines/task1_classifier.py
"""
Task 1: Image Type Classification Pipeline
Classifies images as garment, body, or unknown with attributes
"""

import time
import logging
from typing import List, Optional, Tuple

from PIL import Image

from ..config import settings
from ..schemas import ImageClassification, ImageClass
from ..utils.parser import extract_json_from_response
from ..utils.priority import calculate_priority_score
from ..utils.image_mapping import ImageNameMapper
from ._runner import build_prompt_text, run_vlm
from .prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class Task1Classifier:
    """Multi-image classification (garment / body / unknown).

    Batching is bounded by the prompt-token budget of the VLM, not by
    throughput. See config.task1_batch_size for the empirically-derived
    ceiling.
    """

    def __init__(
        self,
        version: str = "v1.0",
        batch_size: Optional[int] = None,
        input_max_side: Optional[int] = None,
    ):
        self.version = version
        self.batch_size = batch_size or settings.task1_batch_size
        self.input_max_side = input_max_side or settings.vlm_input_max_side
        self.prompt_template = PromptTemplates.get_task1_prompt(version)

    async def classify_images(
        self,
        images: List[Image.Image],
        image_ids: List[str],
        name_mapper: Optional[ImageNameMapper] = None,
        use_cache: bool = True,
    ) -> Tuple[List[ImageClassification], str]:
        """Classify multiple images with automatic batching.

        Returns (classifications, raw_response). The raw response is the
        concatenation of every batch's raw text — kept so callers can
        log, audit, or replay parser bugs without re-running inference.
        Previously the annotation said ``List[ImageClassification]`` but
        the body always returned a tuple; callers like
        ``main.classify_images`` already destructure the tuple, so
        fixing the annotation is the safe direction.
        """

        if not images or not image_ids:
            return [], ""

        if len(images) != len(image_ids):
            raise ValueError("Number of images and IDs must match")

        total_images = len(images)
        logger.info(f"Classifying {total_images} images with batch size {self.batch_size}")

        # Check if batching is needed
        if total_images <= self.batch_size:
            # Process all at once
            return await self._classify_batch(
                images, image_ids, name_mapper, use_cache
            )

        # Batch processing for large number of images
        logger.info(f"Processing {total_images} images in batches of {self.batch_size}")
        return await self._classify_in_batches(
            images, image_ids, name_mapper, use_cache
        )

    async def _classify_batch(
        self,
        images: List[Image.Image],
        image_ids: List[str],
        name_mapper: Optional[ImageNameMapper] = None,
        use_cache: bool = True,
    ) -> Tuple[List[ImageClassification], str]:
        """
        Classify a single batch of images (internal method)

        Args:
            images: List of PIL Image objects
            image_ids: List of image filenames/IDs
            name_mapper: Optional mapper for image names
            use_cache: Whether to use vision encoding cache

        Returns:
            Tuple of (List of ImageClassification objects, raw response)
        """

        start_time = time.perf_counter()

        prompt_text = build_prompt_text(
            self.prompt_template, image_list=", ".join(image_ids),
        )
        # Let exceptions from engine.generate() propagate. The caller in
        # `_classify_in_batches` is the one that decides whether a single
        # bad batch should poison the whole run — not this function.
        response = await run_vlm(
            prompt_text=prompt_text,
            images=images,
            use_cache=use_cache,
            image_ids=image_ids,
            max_side=self.input_max_side,
        )
        logger.info(
            "task1 batch of %d classified in %.3fs", len(images),
            time.perf_counter() - start_time,
        )

        classifications = self._parse_response(response, image_ids, images)
        if name_mapper:
            for cls in classifications:
                original = name_mapper.get_original_name(cls.filename)
                if original:
                    cls.simple_name = cls.filename
                    cls.filename = original
        return classifications, response

    async def _classify_in_batches(
        self,
        images: List[Image.Image],
        image_ids: List[str],
        name_mapper: Optional[ImageNameMapper] = None,
        use_cache: bool = True,
    ) -> Tuple[List[ImageClassification], str]:
        """
        Classify images in multiple batches and merge results

        Args:
            images: List of PIL Image objects
            image_ids: List of image filenames/IDs
            name_mapper: Optional mapper for image names
            use_cache: Whether to use vision encoding cache

        Returns:
            Tuple of (List of ImageClassification objects, concatenated responses)
        """

        total_images = len(images)
        all_classifications = []
        all_responses = []

        # Calculate number of batches
        num_batches = (total_images + self.batch_size - 1) // self.batch_size

        logger.info(f"Processing {total_images} images in {num_batches} batches")

        overall_start_time = time.perf_counter()

        # Process each batch
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_images)

            batch_images = images[start_idx:end_idx]
            batch_ids = image_ids[start_idx:end_idx]

            logger.info(f"Processing batch {batch_idx + 1}/{num_batches}: "
                       f"images {start_idx + 1}-{end_idx}/{total_images}")

            try:
                # Classify this batch
                batch_classifications, batch_response = await self._classify_batch(
                    batch_images,
                    batch_ids,
                    name_mapper,
                    use_cache
                )

                # Accumulate results
                all_classifications.extend(batch_classifications)
                all_responses.append(f"=== Batch {batch_idx + 1}/{num_batches} ===\n{batch_response}")

                logger.info(f"Batch {batch_idx + 1}/{num_batches} completed: "
                           f"{len(batch_classifications)} classifications")

            except Exception as e:
                # Degrade the batch, not the whole session. Every fallback
                # row is tagged parse_failed=True so downstream code
                # (GISProcessor, metrics) can distinguish a failed batch
                # from a genuine UNKNOWN judgement.
                logger.exception(
                    "task1 batch %d/%d failed; degrading %d images to parse_failed=UNKNOWN",
                    batch_idx + 1, num_batches, len(batch_ids),
                )
                all_classifications.extend(
                    self._create_fallback_classifications(batch_ids, batch_images)
                )
                all_responses.append(
                    f"=== Batch {batch_idx + 1}/{num_batches} (FAILED) ===\nError: {e}"
                )

        overall_time = time.perf_counter() - overall_start_time
        logger.info(f"All batches completed in {overall_time:.3f}s. "
                   f"Total classifications: {len(all_classifications)}")

        # Merge all responses
        merged_response = "\n\n".join(all_responses)

        return all_classifications, merged_response

    def _parse_response(self,
                       response: str,
                       image_ids: List[str],
                       images: List[Image.Image]) -> List[ImageClassification]:
        """Parse the model response into classifications"""

        classifications = []

        try:
            json_data = extract_json_from_response(response)

            if not isinstance(json_data, dict):
                logger.error(f"Expected dict, got {type(json_data)}")
                return self._create_fallback_classifications(image_ids, images)

            # Identity-matching only. The old positional fallback silently
            # paired the Nth response with the Nth request id whenever the
            # model dropped or renamed a key — a single missing entry would
            # cascade-mislabel every subsequent image. We now require the
            # model to name the image explicitly (exact or basename match)
            # or the row is marked parse_failed.
            for image_id, image in zip(image_ids, images):
                classification_data = json_data.get(image_id)

                if classification_data is None:
                    base_name = image_id.split('.')[0]
                    for key, value in json_data.items():
                        if key.split('.')[0] == base_name:
                            classification_data = value
                            break

                if classification_data:
                    try:
                        # Create classification object
                        cls = ImageClassification(
                            filename=image_id,
                            img_class=classification_data.get('class', 'unknown'),
                            view=classification_data.get('view'),
                            body_crop=classification_data.get('body_crop'),
                            angle=classification_data.get('angle'),
                            resolution=image.size if image else None,
                            
                        )

                        # Calculate priority score
                        cls.priority_score = calculate_priority_score({
                            'class': cls.img_class,
                            'view': cls.view,
                            'body_crop': cls.body_crop,
                            'angle': cls.angle,
                            'width': image.size[0] if image else 512,
                            'height': image.size[1] if image else 512
                        })

                        classifications.append(cls)

                    except Exception as e:
                        logger.warning("task1 parse error for %s: %s", image_id, e)
                        classifications.append(
                            self._create_unknown_classification(image_id, image, parse_failed=True)
                        )
                else:
                    logger.warning("task1 no classification for %s in model JSON", image_id)
                    classifications.append(
                        self._create_unknown_classification(image_id, image, parse_failed=True)
                    )

        except Exception as e:
            logger.error(f"Error parsing Task 1 response: {str(e)}")
            return self._create_fallback_classifications(image_ids, images)

        return classifications

    def _create_unknown_classification(
        self,
        image_id: str,
        image: Optional[Image.Image],
        parse_failed: bool = False,
    ) -> ImageClassification:
        return ImageClassification(
            filename=image_id,
            img_class=ImageClass.UNKNOWN,
            resolution=image.size if image else None,
            priority_score=0.0,
            parse_failed=parse_failed,
        )

    def _create_fallback_classifications(
        self,
        image_ids: List[str],
        images: List[Image.Image],
    ) -> List[ImageClassification]:
        # parse_failed=True is load-bearing: GISProcessor filters these
        # before feeding into Task 2/3 so we never pair a "probably-a-body"
        # placeholder with a real garment and log it as a valid match.
        return [
            self._create_unknown_classification(img_id, img, parse_failed=True)
            for img_id, img in zip(image_ids, images)
        ]

    def filter_by_class(self,
                       classifications: List[ImageClassification],
                       target_class: ImageClass) -> List[ImageClassification]:
        """Filter classifications by class type"""
        return [c for c in classifications if c.img_class == target_class]

    def get_best_garments(self,
                         classifications: List[ImageClassification],
                         max_count: int = 5) -> List[ImageClassification]:
        """Get best garment images sorted by priority"""
        garments = self.filter_by_class(classifications, ImageClass.GARMENT)
        garments.sort(key=lambda x: x.priority_score or 0, reverse=True)
        return garments[:max_count]

    def get_best_bodies(self,
                       classifications: List[ImageClassification],
                       max_count: int = 5) -> List[ImageClassification]:
        """Get best body images sorted by priority"""
        bodies = self.filter_by_class(classifications, ImageClass.BODY)
        bodies.sort(key=lambda x: x.priority_score or 0, reverse=True)
        return bodies[:max_count]
