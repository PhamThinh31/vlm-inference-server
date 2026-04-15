"""
Test concurrent request handling capacity
Measures how many simultaneous requests the server can handle
"""

import asyncio
import aiohttp
import time
import statistics
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ConcurrencyResult:
    """Result from a concurrency test"""
    concurrency_level: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    errors: List[str]


class ConcurrentRequestTester:
    """Test concurrent request handling"""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.session = None

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=300)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def single_request(self, request_id: int, endpoint: str = "/health") -> Dict:
        """Make a single request and measure time"""
        start_time = time.time()

        try:
            url = f"{self.api_url}{endpoint}"
            async with self.session.get(url) as response:
                duration = time.time() - start_time
                return {
                    "request_id": request_id,
                    "success": response.status == 200,
                    "status_code": response.status,
                    "duration": duration,
                    "error": None
                }
        except Exception as e:
            duration = time.time() - start_time
            return {
                "request_id": request_id,
                "success": False,
                "status_code": 0,
                "duration": duration,
                "error": str(e)
            }

    async def test_concurrency_level(
        self,
        concurrency: int,
        total_requests: int = 100,
        endpoint: str = "/health"
    ) -> ConcurrencyResult:
        """
        Test a specific concurrency level

        Args:
            concurrency: Number of concurrent requests
            total_requests: Total number of requests to make
            endpoint: API endpoint to test
        """
        print(f"\n{'='*80}")
        print(f"Testing Concurrency Level: {concurrency}")
        print(f"Total Requests: {total_requests}")
        print(f"{'='*80}")

        semaphore = asyncio.Semaphore(concurrency)
        results = []
        errors = []

        async def bounded_request(req_id: int):
            async with semaphore:
                return await self.single_request(req_id, endpoint)

        # Start timer
        overall_start = time.time()

        # Create and run all requests
        tasks = [bounded_request(i) for i in range(total_requests)]
        results = await asyncio.gather(*tasks)

        # Calculate metrics
        overall_time = time.time() - overall_start

        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        if successful:
            durations = [r["duration"] for r in successful]
            avg_time = statistics.mean(durations)
            min_time = min(durations)
            max_time = max(durations)
        else:
            avg_time = min_time = max_time = 0.0

        for r in failed:
            if r["error"]:
                errors.append(r["error"])

        requests_per_second = total_requests / overall_time if overall_time > 0 else 0

        result = ConcurrencyResult(
            concurrency_level=concurrency,
            total_requests=total_requests,
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_time=overall_time,
            avg_response_time=avg_time,
            min_response_time=min_time,
            max_response_time=max_time,
            requests_per_second=requests_per_second,
            errors=list(set(errors))[:5]  # Unique errors, max 5
        )

        # Print results
        self._print_result(result)

        return result

    def _print_result(self, result: ConcurrencyResult):
        """Print concurrency test result"""
        success_rate = (result.successful_requests / result.total_requests * 100) if result.total_requests > 0 else 0

        print(f"\n Results:")
        print(f"Successful:     {result.successful_requests}/{result.total_requests} ({success_rate:.1f}%)")
        print(f"Failed:         {result.failed_requests}")
        print(f"Total Time:     {result.total_time:.2f}s")
        print(f"Throughput:     {result.requests_per_second:.2f} req/s")

        if result.successful_requests > 0:
            print(f"\n️  Response Times:")
            print(f"Average:        {result.avg_response_time:.3f}s")
            print(f"Min:            {result.min_response_time:.3f}s")
            print(f"Max:            {result.max_response_time:.3f}s")

        if result.errors:
            print(f"\n Sample Errors:")
            for err in result.errors[:3]:
                print(f"- {err}")

    async def find_max_concurrency(
        self,
        start_concurrency: int = 1,
        max_concurrency: int = 100,
        step: int = 5,
        requests_per_test: int = 50,
        success_threshold: float = 0.95
    ) -> int:
        """
        Find maximum concurrency the server can handle

        Args:
            start_concurrency: Starting concurrency level
            max_concurrency: Maximum concurrency to test
            step: Increment step
            requests_per_test: Number of requests per test
            success_threshold: Success rate threshold (0.95 = 95%)

        Returns:
            Maximum safe concurrency level
        """
        print(f"\n{'='*80}")
        print(f"FINDING MAXIMUM CONCURRENCY")
        print(f"{'='*80}")
        print(f"Testing from {start_concurrency} to {max_concurrency} (step: {step})")
        print(f"Success threshold: {success_threshold * 100:.0f}%")

        max_safe_concurrency = start_concurrency
        all_results = []

        for concurrency in range(start_concurrency, max_concurrency + 1, step):
            result = await self.test_concurrency_level(
                concurrency=concurrency,
                total_requests=requests_per_test,
                endpoint="/health"
            )

            all_results.append(result)

            success_rate = result.successful_requests / result.total_requests

            if success_rate >= success_threshold:
                max_safe_concurrency = concurrency
                print(f" Passed: {concurrency} concurrent requests")
            else:
                print(f" Failed: {concurrency} concurrent requests (success rate: {success_rate*100:.1f}%)")
                break

        # Print summary
        print(f"\n{'='*80}")
        print(f"CONCURRENCY TEST SUMMARY")
        print(f"{'='*80}")
        print(f"\n Maximum Safe Concurrency: {max_safe_concurrency}")

        print(f"\n All Results:")
        print(f"{'Concurrency':<15} {'Success Rate':<15} {'Throughput':<15} {'Avg Time':<15}")
        print(f"{'-'*60}")
        for r in all_results:
            success_rate = (r.successful_requests / r.total_requests * 100) if r.total_requests > 0 else 0
            print(f"{r.concurrency_level:<15} {success_rate:<14.1f}% {r.requests_per_second:<14.2f} {r.avg_response_time:<14.3f}s")

        return max_safe_concurrency


async def run_quick_test(api_url: str):
    """Quick test with preset concurrency levels"""
    print("")
    print("Concurrent Request Capacity Test                            ")
    print("")
    print(f"\nAPI URL: {api_url}")

    # Test preset concurrency levels
    concurrency_levels = [1, 5, 10, 20, 30, 50]

    async with ConcurrentRequestTester(api_url) as tester:
        # Check if API is available
        print("\n Checking API availability...")
        result = await tester.single_request(0, "/health")
        if not result["success"]:
            print(f" API not available: {result['error']}")
            return

        print(" API is available")

        results = []
        for concurrency in concurrency_levels:
            result = await tester.test_concurrency_level(
                concurrency=concurrency,
                total_requests=50,
                endpoint="/health"
            )
            results.append(result)

            # Stop if success rate drops below 90%
            success_rate = result.successful_requests / result.total_requests
            if success_rate < 0.9:
                print(f"\n️  Success rate dropped below 90% at concurrency={concurrency}")
                break

        # Find best concurrency level
        successful_results = [r for r in results if r.successful_requests / r.total_requests >= 0.95]
        if successful_results:
            best = max(successful_results, key=lambda x: x.concurrency_level)
            print(f"\n{'='*80}")
            print(f" RECOMMENDATION")
            print(f"{'='*80}")
            print(f"Maximum safe concurrency: {best.concurrency_level} concurrent requests")
            print(f"Expected throughput:      {best.requests_per_second:.2f} requests/sec")
            print(f"Average response time:    {best.avg_response_time:.3f}s")
        else:
            print(f"\n️  Server cannot handle even 1 concurrent request reliably")


async def run_max_finder(api_url: str, max_concurrency: int):
    """Find maximum concurrency with binary search approach"""
    print("")
    print("Find Maximum Concurrent Request Capacity                       ")
    print("")
    print(f"\nAPI URL: {api_url}")

    async with ConcurrentRequestTester(api_url) as tester:
        max_safe = await tester.find_max_concurrency(
            start_concurrency=1,
            max_concurrency=max_concurrency,
            step=5,
            requests_per_test=50,
            success_threshold=0.95
        )

        print(f"\n{'='*80}")
        print(f" Test Complete!")
        print(f"{'='*80}")
        print(f"\n Your server can handle up to {max_safe} concurrent requests")
        print(f"with 95%+ success rate")


async def main():
    parser = argparse.ArgumentParser(description="Test concurrent request capacity")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--mode", choices=["quick", "find-max"], default="quick",
                       help="Test mode: 'quick' for preset levels, 'find-max' to find maximum")
    parser.add_argument("--max-concurrency", type=int, default=100,
                       help="Maximum concurrency to test (for find-max mode)")

    args = parser.parse_args()

    if args.mode == "quick":
        await run_quick_test(args.url)
    else:
        await run_max_finder(args.url, args.max_concurrency)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n️  Test interrupted by user")
        sys.exit(1)
