# client/example_direct_inference.py
"""
Example: Direct Task 2 and Task 3 without Task 1
"""

import asyncio
from vlm4gis_client import VLM4GISClient

async def example_direct_task2():
    """Run Task 2 directly with image paths"""

    async with VLM4GISClient() as client:
        # Direct pairs - no session or Task 1 needed!
        image_pairs = [
            ('/path/to/your/garment_image.jpg',
            '/path/to/your/body_image.jpg')
        ]

        print("Running Task 2 directly with image paths...")
        result = await client.extract_attributes_direct(image_pairs)
        print("Task 2 Results:")
        print(result)
        # print(f"\nProcessed {result['pairs_processed']} pairs")
        for item in result['results']:
            print(f"\nPair: {item['garment_file']} + {item['body_file']}")
            print(f"Category: {item['category']}")
            print(f"Description: {item['description']}")
            print(f"Attributes: {item['attributes']}")

async def example_direct_task3():
    """Run Task 3 directly with image paths"""

    async with VLM4GISClient() as client:
        # Check if these pairs are valid for VTON
        image_pairs = [
            ('/path/to/your/garment_image.jpg',
            '/path/to/your/body_image.jpg')
        ]

        print("Running Task 3 validation directly...")
        result = await client.validate_pairs_direct(image_pairs)
        print("Task 3 Validation Results:")
        print(result)
        print(f"\nValidated {result['total_validated']} pairs")
        print(f"Valid: {result['valid_count']}, Invalid: {result['invalid_count']}")

        for validation in result['validations']:
            status = "" if validation['is_valid'] else ""
            print(f"{status} {validation['garment_file']} + {validation['body_file']}")
            print(f"   Confidence: {validation['confidence']:.2f}")

async def example_mixed_workflow():
    """Example: Mix direct and session-based approaches"""

    async with VLM4GISClient() as client:
        # Option 1: Quick validation of specific pairs
        print("Quick validation without session...")
        quick_pairs = [
            ("test/garment.jpg", "test/body.jpg"),
        ]
        quick_result = await client.validate_pairs_direct(quick_pairs)
        print(f"Quick check: {'Valid' if quick_result['validations'][0]['is_valid'] else 'Invalid'}")

        # Option 2: Full pipeline for batch processing
        print("\nFull pipeline for folder...")
        full_result = await client.process_folder("./batch_images")
        print(f"Batch processed: {full_result['summary']['image_statistics']['total']} images")

if __name__ == "__main__":
    asyncio.run(example_direct_task2())
    # asyncio.run(example_direct_task3())
    # asyncio.run(example_mixed_workflow())
