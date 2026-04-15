# app/utils/priority.py
"""
Priority calculation utilities for ranking images and pairs
"""

from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_priority_score(image_info: Dict[str, Any],
                            weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate priority score for an image
    
    Args:
        image_info: Dictionary with image metadata
        weights: Optional custom weights for scoring
        
    Returns:
        Priority score between 0 and 1
    """
    
    # Default weights
    default_weights = {
        'resolution': 0.3,
        'view_angle': 0.5,
        'body_crop': 0.2
    }
    
    weights = weights or default_weights
    score = 0.0
    
    # Resolution factor (normalized 0-1)
    if 'width' in image_info and 'height' in image_info:
        pixels = image_info['width'] * image_info['height']
        # Cap at 4MP (2048x2048)
        resolution_score = min(pixels / (2048 * 2048), 1.0)
        score += resolution_score * weights.get('resolution', 0.3)
    
    # Class-specific scoring
    img_class = image_info.get('class', 'unknown')
    
    if img_class == 'body':
        # Angle scoring
        angle_scores = {
            'front': 1.0,
            'front-left': 0.8,
            'front-right': 0.8,
            'left side': 0.5,
            'right side': 0.5,
            'back-left': 0.3,
            'back-right': 0.3,
            'back': 0.3
        }
        angle = image_info.get('angle', '')
        angle_score = angle_scores.get(angle, 0.0)
        score += angle_score * weights.get('view_angle', 0.5)
        
        # Body crop scoring
        crop_scores = {
            'full_body': 1.0,
            'half_top': 0.6,
            'half_bottom': 0.4
        }
        crop = image_info.get('body_crop', '')
        crop_score = crop_scores.get(crop, 0.0)
        score += crop_score * weights.get('body_crop', 0.2)
    
    elif img_class == 'garment':
        # View scoring for garments
        view_scores = {
            'front': 1.0,
            'back': 0.7,
            'unknown': 0.5
        }
        view = image_info.get('view', '')
        view_score = view_scores.get(view, 0.0)
        score += view_score * weights.get('view_angle', 0.7)
    
    return min(score, 1.0)  # Ensure score doesn't exceed 1.0

def rank_pairs_by_priority(pairs: List[Tuple[str, str]],
                          classifications: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Rank garment-body pairs by combined priority
    
    Args:
        pairs: List of (garment_id, body_id) tuples
        classifications: Dictionary mapping image IDs to their classification data
        
    Returns:
        Sorted list of pairs
    """
    
    def pair_score(pair):
        garment_id, body_id = pair
        garment_score = calculate_priority_score(
            classifications.get(garment_id, {})
        )
        body_score = calculate_priority_score(
            classifications.get(body_id, {})
        )
        # Combined score (average)
        return (garment_score + body_score) / 2
    
    return sorted(pairs, key=pair_score, reverse=True)

def filter_low_quality_images(classifications: List[Dict[str, Any]],
                             min_resolution: int = 512) -> List[Dict[str, Any]]:
    """
    Filter out low quality images
    
    Args:
        classifications: List of classification dictionaries
        min_resolution: Minimum resolution (width or height)
        
    Returns:
        Filtered list of classifications
    """
    
    filtered = []
    
    for cls in classifications:
        # Check resolution
        if 'resolution' in cls:
            width, height = cls['resolution']
            if width < min_resolution or height < min_resolution:
                logger.warning(f"Filtering low resolution image: {cls.get('filename')} ({width}x{height})")
                continue
        
        # Check if it's unknown class
        if cls.get('img_class') == 'unknown':
            logger.info(f"Filtering unknown class image: {cls.get('filename')}")
            continue
        
        filtered.append(cls)
    
    return filtered