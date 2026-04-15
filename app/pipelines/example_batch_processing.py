# app/pipelines/example_batch_processing.py
"""
Example demonstrating Task 1 batch processing for large image sets
"""

import asyncio
from typing import List
from PIL import Image
from pathlib import Path

from .task1_classifier import Task1Classifier

async def example_small_batch():
    """Example with small number of images (< 10) - no batching"""

    print("\n" + "="*80)
    print("EXAMPLE 1: Small Batch (< 10 images)")
    print("="*80)

    # Initialize classifier with default batch size (10)
    classifier = Task1Classifier(version="v1.0", batch_size=10)

    # Simulate 8 images
    images = [Image.new('RGB', (512, 512), color='red') for _ in range(8)]
    image_ids = [f"image_{i:03d}.jpg" for i in range(1, 9)]

    print(f"\nProcessing {len(images)} images...")
    print(f"Batch size: {classifier.batch_size}")
    print(f"Expected: Single batch (no splitting)")

    # Classify
    classifications, response = await classifier.classify_images(
        images=images,
        image_ids=image_ids,
        use_cache=True
    )

    print(f"\nResults:")
    print(f"  Total classifications: {len(classifications)}")
    print(f"  Response length: {len(response)} chars")

async def example_large_batch():
    """Example with large number of images (> 10) - automatic batching"""

    print("\n" + "="*80)
    print("EXAMPLE 2: Large Batch (> 10 images)")
    print("="*80)

    # Initialize classifier with batch size 10
    classifier = Task1Classifier(version="v1.0", batch_size=10)

    # Simulate 25 images
    num_images = 25
    images = [Image.new('RGB', (512, 512), color='blue') for _ in range(num_images)]
    image_ids = [f"image_{i:03d}.jpg" for i in range(1, num_images + 1)]

    print(f"\nProcessing {len(images)} images...")
    print(f"Batch size: {classifier.batch_size}")
    print(f"Expected: 3 batches (10 + 10 + 5)")

    # Classify
    classifications, response = await classifier.classify_images(
        images=images,
        image_ids=image_ids,
        use_cache=True
    )

    print(f"\nResults:")
    print(f"  Total classifications: {len(classifications)}")
    print(f"  Number of batches in response: {response.count('=== Batch')}")

    # Show sample classifications
    print(f"\n  Sample classifications:")
    for i, cls in enumerate(classifications[:5]):
        print(f"    {i+1}. {cls.filename}: {cls.img_class}")

async def example_custom_batch_size():
    """Example with custom batch size"""

    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Batch Size")
    print("="*80)

    # Initialize classifier with custom batch size of 5
    classifier = Task1Classifier(version="v1.0", batch_size=5)

    # Simulate 17 images
    num_images = 17
    images = [Image.new('RGB', (512, 512), color='green') for _ in range(num_images)]
    image_ids = [f"image_{i:03d}.jpg" for i in range(1, num_images + 1)]

    print(f"\nProcessing {len(images)} images...")
    print(f"Batch size: {classifier.batch_size}")
    print(f"Expected: 4 batches (5 + 5 + 5 + 2)")

    # Classify
    classifications, response = await classifier.classify_images(
        images=images,
        image_ids=image_ids,
        use_cache=True
    )

    print(f"\nResults:")
    print(f"  Total classifications: {len(classifications)}")
    print(f"  Number of batches: {response.count('=== Batch')}")

async def example_real_images():
    """Example with real images from directory"""

    print("\n" + "="*80)
    print("EXAMPLE 4: Real Images from Directory")
    print("="*80)

    # Path to images directory
    image_dir = Path("path/to/your/images")

    if not image_dir.exists():
        print(f"\nSkipping: Directory {image_dir} not found")
        print("Update the path in the example code to use real images")
        return

    # Load images
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    if not image_files:
        print(f"\nNo images found in {image_dir}")
        return

    print(f"\nFound {len(image_files)} images in {image_dir}")

    # Load first 50 images (or all if less)
    image_files = image_files[:50]
    images = [Image.open(f) for f in image_files]
    image_ids = [f.name for f in image_files]

    # Initialize classifier
    classifier = Task1Classifier(version="v1.0", batch_size=10)

    print(f"Processing {len(images)} images with batch size {classifier.batch_size}...")

    # Classify
    classifications, response = await classifier.classify_images(
        images=images,
        image_ids=image_ids,
        use_cache=True
    )

    print(f"\nResults:")
    print(f"  Total classifications: {len(classifications)}")

    # Count by class
    from collections import Counter
    class_counts = Counter(cls.img_class for cls in classifications)

    print(f"\n  Classification breakdown:")
    for img_class, count in class_counts.items():
        print(f"    {img_class}: {count} ({count/len(classifications)*100:.1f}%)")

async def example_batch_statistics():
    """Example showing batch processing statistics"""

    print("\n" + "="*80)
    print("EXAMPLE 5: Batch Processing Statistics")
    print("="*80)

    # Test different scenarios
    test_cases = [
        (5, 10),    # 5 images, batch 10 -> 1 batch
        (10, 10),   # 10 images, batch 10 -> 1 batch
        (15, 10),   # 15 images, batch 10 -> 2 batches
        (25, 10),   # 25 images, batch 10 -> 3 batches
        (50, 10),   # 50 images, batch 10 -> 5 batches
        (100, 10),  # 100 images, batch 10 -> 10 batches
    ]

    print(f"\n{'Images':<10} {'Batch Size':<15} {'Expected Batches':<20} {'Actual Batches':<20}")
    print("-" * 70)

    for num_images, batch_size in test_cases:
        classifier = Task1Classifier(version="v1.0", batch_size=batch_size)

        images = [Image.new('RGB', (128, 128)) for _ in range(num_images)]
        image_ids = [f"img_{i:03d}.jpg" for i in range(num_images)]

        classifications, response = await classifier.classify_images(
            images=images,
            image_ids=image_ids,
            use_cache=True
        )

        expected_batches = (num_images + batch_size - 1) // batch_size
        actual_batches = max(1, response.count('=== Batch'))

        print(f"{num_images:<10} {batch_size:<15} {expected_batches:<20} {actual_batches:<20}")

async def example_error_handling():
    """Example showing error handling in batch processing"""

    print("\n" + "="*80)
    print("EXAMPLE 6: Error Handling")
    print("="*80)

    classifier = Task1Classifier(version="v1.0", batch_size=10)

    # Simulate 25 images
    num_images = 25
    images = [Image.new('RGB', (512, 512)) for _ in range(num_images)]
    image_ids = [f"image_{i:03d}.jpg" for i in range(1, num_images + 1)]

    print(f"\nProcessing {len(images)} images...")
    print("If any batch fails, it will create fallback 'unknown' classifications")

    try:
        classifications, response = await classifier.classify_images(
            images=images,
            image_ids=image_ids,
            use_cache=True
        )

        print(f"\nResults:")
        print(f"  Total classifications: {len(classifications)}")

        # Check for fallback classifications
        unknown_count = sum(1 for cls in classifications if cls.img_class == 'unknown')
        print(f"  Unknown classifications: {unknown_count}")

        if "FAILED" in response:
            print(f"  Warning: Some batches failed (check logs)")

    except Exception as e:
        print(f"\nError: {e}")
        print("The entire operation failed")

# Comparison function
def compare_processing_modes():
    """Compare single-batch vs multi-batch processing"""

    print("\n" + "="*80)
    print("COMPARISON: Single-batch vs Multi-batch")
    print("="*80)

    print("""
    Single-batch (≤10 images):
     Faster - single inference call
     Single response to parse
     Limited to 10 images

    Multi-batch (>10 images):
     Can process unlimited images
     Automatic chunking and merging
     Error isolation (one batch fails, others continue)
      Multiple inference calls (slower overall)
      Multiple responses to parse

    When to use:
    • Use default batch_size=10 for optimal model performance
    • Reduce batch_size if running into memory issues
    • Increase batch_size if model can handle more images
    """)

# Main runner
async def main():
    """Run all examples"""

        print("            Task 1: Batch Processing Examples                            ")
    
    # Run examples
    await example_small_batch()
    await example_large_batch()
    await example_custom_batch_size()
    await example_batch_statistics()
    await example_error_handling()

    # Show comparison
    compare_processing_modes()

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)

if __name__ == "__main__":
    # Uncomment to run examples
    # asyncio.run(main())

    # Or run individual examples:
    # asyncio.run(example_large_batch())
    pass
