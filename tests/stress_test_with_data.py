"""
Stress Test Script for VLM4GIS API using real data
Tests with images from ./data
"""

import asyncio
import aiohttp
import time
import statistics
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import sys
import base64


@dataclass
class TestConfig:
    """Test configuration"""
    api_url: str = "http://localhost:8000"
    data_dir: Path = Path("./data")
    max_images: int = 50  # Number of images to test with
    timeout: int = 600  # 10 minutes


@dataclass
class TestMetrics:
    """Metrics for test results"""
    test_name: str
    total_images: int
    duration: float
    success: bool
    error: str = None

    # Task-specific metrics
    classifications: int = 0
    garments: int = 0
    bodies: int = 0
    unknown: int = 0

    # Performance
    images_per_second: float = 0.0
    avg_time_per_image: float = 0.0


class DataStressTest:
    """Stress test runner using real data"""

    def __init__(self, config: TestConfig):
        self.config = config
        self.session: aiohttp.ClientSession = None

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def find_images(self, max_images: int = None) -> List[Path]:
        """Find images in data directory"""
        max_images = max_images or self.config.max_images

        print(f"\n Searching for images in: {self.config.data_dir}")

        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        all_images = []

        for ext in image_extensions:
            all_images.extend(self.config.data_dir.rglob(ext))

        all_images = list(set(all_images))  # Remove duplicates
        all_images.sort()

        print(f"Found {len(all_images)} total images")

        if max_images and len(all_images) > max_images:
            all_images = all_images[:max_images]
            print(f"Using first {max_images} images for testing")

        return all_images

    async def create_session(self) -> str:
        """Create a new session"""
        url = f"{self.config.api_url}/session/create"

        try:
            async with self.session.post(url) as response:
                if response.status == 200:
                    data = await response.json()
                    session_id = data.get("session_id")
                    print(f" Created session: {session_id}")
                    return session_id
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to create session: {response.status} - {error_text}")
        except Exception as e:
            print(f" Error creating session: {e}")
            raise

    async def upload_images(self, session_id: str, image_paths: List[Path]) -> bool:
        """Upload images to session"""
        url = f"{self.config.api_url}/session/{session_id}/upload"

        print(f"\n Uploading {len(image_paths)} images...")

        try:
            form_data = aiohttp.FormData()

            for img_path in image_paths:
                form_data.add_field(
                    'files',
                    open(img_path, 'rb'),
                    filename=img_path.name,
                    content_type='image/jpeg'
                )

            async with self.session.post(url, data=form_data) as response:
                if response.status == 200:
                    data = await response.json()
                    uploaded = data.get("total_uploaded", 0)
                    print(f" Uploaded {uploaded} images")
                    return True
                else:
                    error_text = await response.text()
                    print(f" Upload failed: {response.status} - {error_text}")
                    return False
        except Exception as e:
            print(f" Error uploading images: {e}")
            return False

    async def test_task1(self, session_id: str, num_images: int) -> TestMetrics:
        """Test Task 1: Classification"""
        print("\n" + "="*80)
        print("TEST: Task 1 Classification")
        print("="*80)

        url = f"{self.config.api_url}/task/classify"
        data = {"session_id": session_id}

        start_time = time.time()

        try:
            async with self.session.post(url, json=data) as response:
                duration = time.time() - start_time

                if response.status == 200:
                    result = await response.json()

                    classifications = result.get("classifications", [])

                    # Count by class
                    garments = sum(1 for c in classifications if c.get("class") == "garment")
                    bodies = sum(1 for c in classifications if c.get("class") == "body")
                    unknown = sum(1 for c in classifications if c.get("class") == "unknown")

                    metrics = TestMetrics(
                        test_name="Task 1",
                        total_images=num_images,
                        duration=duration,
                        success=True,
                        classifications=len(classifications),
                        garments=garments,
                        bodies=bodies,
                        unknown=unknown,
                        images_per_second=len(classifications) / duration if duration > 0 else 0,
                        avg_time_per_image=duration / len(classifications) if classifications else 0
                    )

                    print(f"\n Task 1 completed in {duration:.2f}s")
                    print(f"Classifications: {len(classifications)}")
                    print(f"- Garments: {garments}")
                    print(f"- Bodies: {bodies}")
                    print(f"- Unknown: {unknown}")
                    print(f"Performance: {metrics.images_per_second:.2f} images/sec")

                    return metrics
                else:
                    error_text = await response.text()
                    return TestMetrics(
                        test_name="Task 1",
                        total_images=num_images,
                        duration=duration,
                        success=False,
                        error=f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            duration = time.time() - start_time
            return TestMetrics(
                test_name="Task 1",
                total_images=num_images,
                duration=duration,
                success=False,
                error=str(e)
            )

    async def test_task2(self, session_id: str) -> TestMetrics:
        """Test Task 2: Attribute Extraction"""
        print("\n" + "="*80)
        print("TEST: Task 2 Attribute Extraction")
        print("="*80)

        url = f"{self.config.api_url}/task/attributes"
        data = {"session_id": session_id}

        start_time = time.time()

        try:
            async with self.session.post(url, json=data) as response:
                duration = time.time() - start_time

                if response.status == 200:
                    result = await response.json()
                    results_list = result.get("results", [])

                    metrics = TestMetrics(
                        test_name="Task 2",
                        total_images=len(results_list) * 2,  # pairs
                        duration=duration,
                        success=True
                    )

                    print(f"\n Task 2 completed in {duration:.2f}s")
                    print(f"Pairs processed: {len(results_list)}")

                    return metrics
                else:
                    error_text = await response.text()
                    return TestMetrics(
                        test_name="Task 2",
                        total_images=0,
                        duration=duration,
                        success=False,
                        error=f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            duration = time.time() - start_time
            return TestMetrics(
                test_name="Task 2",
                total_images=0,
                duration=duration,
                success=False,
                error=str(e)
            )

    async def test_task3(self, session_id: str) -> TestMetrics:
        """Test Task 3: Validation"""
        print("\n" + "="*80)
        print("TEST: Task 3 Validation")
        print("="*80)

        url = f"{self.config.api_url}/task/validate"
        data = {"session_id": session_id, "max_pairs": 10}

        start_time = time.time()

        try:
            async with self.session.post(url, json=data) as response:
                duration = time.time() - start_time

                if response.status == 200:
                    result = await response.json()
                    validations = result.get("validations", [])
                    valid_count = result.get("valid_count", 0)

                    metrics = TestMetrics(
                        test_name="Task 3",
                        total_images=len(validations) * 2,  # pairs
                        duration=duration,
                        success=True
                    )

                    print(f"\n Task 3 completed in {duration:.2f}s")
                    print(f"Pairs validated: {len(validations)}")
                    print(f"Valid pairs: {valid_count}")

                    return metrics
                else:
                    error_text = await response.text()
                    return TestMetrics(
                        test_name="Task 3",
                        total_images=0,
                        duration=duration,
                        success=False,
                        error=f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            duration = time.time() - start_time
            return TestMetrics(
                test_name="Task 3",
                total_images=0,
                duration=duration,
                success=False,
                error=str(e)
            )

    async def test_full_pipeline(self, session_id: str) -> TestMetrics:
        """Test Full Pipeline"""
        print("\n" + "="*80)
        print("TEST: Full GIS Pipeline")
        print("="*80)

        url = f"{self.config.api_url}/task/gis-pipeline"
        data = {
            "session_id": session_id,
            "max_pairs_task2": 5,
            "max_pairs_task3": 10,
            "use_cache": True
        }

        start_time = time.time()

        try:
            async with self.session.post(url, json=data) as response:
                duration = time.time() - start_time

                if response.status == 200:
                    result = await response.json()

                    task1_results = result.get("task1_results", [])
                    task2_results = result.get("task2_results", [])
                    task3_results = result.get("task3_results", [])

                    metrics = TestMetrics(
                        test_name="Full Pipeline",
                        total_images=len(task1_results),
                        duration=duration,
                        success=True
                    )

                    print(f"\n Full pipeline completed in {duration:.2f}s")
                    print(f"Task 1 classifications: {len(task1_results)}")
                    print(f"Task 2 extractions: {len(task2_results)}")
                    print(f"Task 3 validations: {len(task3_results)}")

                    return metrics
                else:
                    error_text = await response.text()
                    return TestMetrics(
                        test_name="Full Pipeline",
                        total_images=0,
                        duration=duration,
                        success=False,
                        error=f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            duration = time.time() - start_time
            return TestMetrics(
                test_name="Full Pipeline",
                total_images=0,
                duration=duration,
                success=False,
                error=str(e)
            )


async def run_stress_test(config: TestConfig, test_type: str = "all"):
    """Run stress test with real data"""

    print("")
    print("VLM4GIS API STRESS TEST (Real Data)                         ")
    print("")

    print(f"\nConfiguration:")
    print(f"  API URL:         {config.api_url}")
    print(f"  Data Directory:  {config.data_dir}")
    print(f"  Max Images:      {config.max_images}")
    print(f"  Test Type:       {test_type}")
    print(f"  Timeout:         {config.timeout}s")

    all_metrics = []

    async with DataStressTest(config) as tester:
        # Find images
        image_paths = tester.find_images()

        if not image_paths:
            print("\n No images found!")
            return

        num_images = len(image_paths)

        # Create session
        try:
            session_id = await tester.create_session()
        except Exception as e:
            print(f"\n Failed to create session: {e}")
            return

        # Upload images
        upload_success = await tester.upload_images(session_id, image_paths)

        if not upload_success:
            print("\n Failed to upload images!")
            return

        # Run tests
        if test_type in ["all", "task1"]:
            metrics = await tester.test_task1(session_id, num_images)
            all_metrics.append(metrics)

        if test_type in ["all", "task2"]:
            metrics = await tester.test_task2(session_id)
            all_metrics.append(metrics)

        if test_type in ["all", "task3"]:
            metrics = await tester.test_task3(session_id)
            all_metrics.append(metrics)

        if test_type in ["all", "pipeline"]:
            metrics = await tester.test_full_pipeline(session_id)
            all_metrics.append(metrics)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for metrics in all_metrics:
        status = " PASS" if metrics.success else " FAIL"
        print(f"\n{metrics.test_name}: {status}")
        print(f"  Duration: {metrics.duration:.2f}s")

        if metrics.success:
            if metrics.classifications > 0:
                print(f"  Classifications: {metrics.classifications}")
                print(f"  Performance: {metrics.images_per_second:.2f} images/sec")
        else:
            print(f"  Error: {metrics.error}")

    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="VLM4GIS Stress Test with Real Data")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--max-images", type=int, default=50, help="Maximum images to test")
    parser.add_argument("--timeout", type=int, default=600, help="Request timeout (seconds)")
    parser.add_argument("--test", choices=["all", "task1", "task2", "task3", "pipeline"],
                       default="all", help="Which test to run")

    args = parser.parse_args()

    config = TestConfig(
        api_url=args.url,
        data_dir=Path(args.data_dir),
        max_images=args.max_images,
        timeout=args.timeout
    )

    await run_stress_test(config, args.test)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n️  Test interrupted by user")
        sys.exit(1)
