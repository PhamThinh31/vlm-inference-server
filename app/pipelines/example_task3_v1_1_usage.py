# app/pipelines/example_task3_v1_1_usage.py
"""
Example demonstrating Task 3 v1.1 with pose quality scoring
"""

import asyncio
from typing import List
from PIL import Image

from .task3_validation import Task3Validator

async def example_task3_v1_1():
    """Example showing how to use Task 3 v1.1 with pose quality scoring"""

    # Initialize validator with v1.1
    validator = Task3Validator(version="v1.1")

    # Assuming you have images and pairs ready
    images_dict = {
        "garment1.jpg": Image.open("path/to/garment1.jpg"),
        "body1.jpg": Image.open("path/to/body1.jpg"),
        # ... more images
    }

    pairs = [
        ("garment1.jpg", "body1.jpg"),
        ("garment1.jpg", "body2.jpg"),
        # ... more pairs
    ]

    # Validate all pairs
    validations = await validator.validate_pairs_batch(
        pairs=pairs,
        images_dict=images_dict,
        use_cache=True
    )

    print(f"Validated {len(validations)} pairs")
    print("\n" + "="*80)

    # Display all results
    for v in validations:
        print(f"\nPair: {v.garment_file} + {v.body_file}")
        print(f"  Valid: {v.is_valid}")
        print(f"  Pose Quality: {v.pose_quality:.2f}/100" if v.pose_quality else "  Pose Quality: N/A")
        print(f"  Confidence: {v.confidence:.2f}")
        print(f"  Raw Response: {v.raw_response}")

    print("\n" + "="*80)

    # Get statistics
    stats = validator.get_statistics(validations)
    print("\n Statistics:")
    print(f"  Total Pairs: {stats['total_pairs']}")
    print(f"  Valid Pairs: {stats['valid_pairs']} ({stats['valid_percentage']:.1f}%)")
    print(f"  Invalid Pairs: {stats['invalid_pairs']}")

    if 'avg_pose_quality' in stats:
        print(f"\n  Pose Quality Stats:")
        print(f"    Average: {stats['avg_pose_quality']:.2f}")
        print(f"    Max: {stats['max_pose_quality']:.2f}")
        print(f"    Min: {stats['min_pose_quality']:.2f}")
        print(f"    Std Dev: {stats['pose_quality_std']:.2f}")

    print("\n" + "="*80)

    # Get best valid pairs sorted by pose quality
    best_pairs = validator.get_best_valid_pairs(
        validations=validations,
        max_count=5,
        min_pose_quality=40.0,  # Only pairs with pose quality >= 40
        sort_by="pose_quality"   # Sort by pose quality (v1.1)
    )

    print("\n Top 5 Valid Pairs (sorted by pose quality):")
    for i, v in enumerate(best_pairs, 1):
        print(f"\n{i}. {v.garment_file} + {v.body_file}")
        print(f"   Pose Quality: {v.pose_quality:.2f}/100")
        print(f"   Confidence: {v.confidence:.2f}")

    print("\n" + "="*80)

    # Sort all results by pose quality
    sorted_by_quality = validator.sort_by_pose_quality(validations, descending=True)

    print("\n All Pairs Sorted by Pose Quality:")
    for i, v in enumerate(sorted_by_quality[:10], 1):  # Show top 10
        status = "" if v.is_valid else ""
        pose_str = f"{v.pose_quality:.2f}" if v.pose_quality else "N/A"
        print(f"{i}. {status} {v.garment_file} + {v.body_file} | Pose: {pose_str}")

    print("\n" + "="*80)

    # Filter valid pairs with minimum pose quality
    high_quality_pairs = validator.filter_valid_pairs(validations, min_confidence=0.7)
    print(f"\n High Quality Pairs (confidence >= 0.7): {len(high_quality_pairs)}")

    return validations

async def example_comparison_v1_vs_v1_1():
    """Compare v1.0 and v1.1 results"""

    # Same pairs, different versions
    validator_v1_0 = Task3Validator(version="v1.0")
    validator_v1_1 = Task3Validator(version="v1.1")

    images_dict = {
        "garment1.jpg": Image.open("path/to/garment1.jpg"),
        "body1.jpg": Image.open("path/to/body1.jpg"),
    }

    pairs = [("garment1.jpg", "body1.jpg")]

    # Validate with both versions
    results_v1_0 = await validator_v1_0.validate_pairs_batch(pairs, images_dict)
    results_v1_1 = await validator_v1_1.validate_pairs_batch(pairs, images_dict)

    print("=" * 80)
    print("Comparison: v1.0 vs v1.1")
    print("=" * 80)

    for i, (v1_0, v1_1) in enumerate(zip(results_v1_0, results_v1_1)):
        print(f"\nPair {i+1}: {v1_0.garment_file} + {v1_0.body_file}")
        print(f"\n  v1.0 Results:")
        print(f"    Valid: {v1_0.is_valid}")
        print(f"    Confidence: {v1_0.confidence:.2f}")
        print(f"    Response: {v1_0.raw_response}")

        print(f"\n  v1.1 Results:")
        print(f"    Valid: {v1_1.is_valid}")
        print(f"    Pose Quality: {v1_1.pose_quality:.2f}/100" if v1_1.pose_quality else "    Pose Quality: N/A")
        print(f"    Confidence: {v1_1.confidence:.2f}")
        print(f"    Response: {v1_1.raw_response}")

# Example usage with different sorting strategies
async def example_different_sorting():
    """Example showing different sorting strategies"""

    validator = Task3Validator(version="v1.1")

    # ... setup images_dict and pairs ...

    # Validate
    validations = []  # await validator.validate_pairs_batch(...)

    # Strategy 1: Sort by pose quality (v1.1)
    best_by_pose = validator.get_best_valid_pairs(
        validations,
        max_count=10,
        min_pose_quality=40.0,
        sort_by="pose_quality"
    )

    # Strategy 2: Sort by confidence (for backward compatibility with v1.0)
    best_by_confidence = validator.get_best_valid_pairs(
        validations,
        max_count=10,
        sort_by="confidence"
    )

    # Strategy 3: Custom filtering - only high-quality poses
    ideal_poses = [
        v for v in validations
        if v.is_valid and v.pose_quality and v.pose_quality >= 70.0
    ]

    # Strategy 4: Challenging poses (for augmentation dataset)
    challenging_poses = [
        v for v in validations
        if v.is_valid and v.pose_quality and 40.0 <= v.pose_quality < 70.0
    ]

    print(f"Best by pose quality: {len(best_by_pose)}")
    print(f"Best by confidence: {len(best_by_confidence)}")
    print(f"Ideal poses (70-100): {len(ideal_poses)}")
    print(f"Challenging poses (40-70): {len(challenging_poses)}")

if __name__ == "__main__":
    # Run examples
    asyncio.run(example_task3_v1_1())
    # asyncio.run(example_comparison_v1_vs_v1_1())
    # asyncio.run(example_different_sorting())
