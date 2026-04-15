"""
Visualize stress test results from JSON output
Requires matplotlib: pip install matplotlib
"""

import json
import sys
from pathlib import Path
from typing import Dict, List


def print_comparison_table(results_files: List[str]):
    """Print comparison table of multiple test results"""

    all_data = []

    for file_path in results_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_data.append({
                'file': Path(file_path).name,
                'timestamp': data.get('timestamp', 'N/A'),
                'results': data.get('results', {})
            })

    if not all_data:
        print("No data to display")
        return

    # Print header
    print("\n" + "="*120)
    print("STRESS TEST COMPARISON")
    print("="*120)

    # Get all unique test names
    test_names = set()
    for data in all_data:
        test_names.update(data['results'].keys())

    for test_name in sorted(test_names):
        print(f"\n {test_name.upper()}")
        print("-"*120)

        # Table header
        print(f"{'File':<25} {'Timestamp':<20} {'Requests':<10} {'Success%':<10} {'RPS':<10} {'Avg(s)':<10} {'P95(s)':<10} {'P99(s)':<10}")
        print("-"*120)

        # Table rows
        for data in all_data:
            if test_name in data['results']:
                result = data['results'][test_name]
                total = result.get('total_requests', 0)
                success = result.get('successful_requests', 0)
                success_pct = (success / total * 100) if total > 0 else 0

                print(f"{data['file']:<25} "
                      f"{data['timestamp'][:19]:<20} "
                      f"{total:<10} "
                      f"{success_pct:<10.1f} "
                      f"{result.get('requests_per_second', 0):<10.2f} "
                      f"{result.get('avg_latency', 0):<10.3f} "
                      f"{result.get('p95_latency', 0):<10.3f} "
                      f"{result.get('p99_latency', 0):<10.3f}")

    print("\n" + "="*120)


def print_detailed_report(result_file: str):
    """Print detailed report for a single test result"""

    with open(result_file, 'r') as f:
        data = json.load(f)

    print("\n" + "="*80)
    print("DETAILED STRESS TEST REPORT")
    print("="*80)

    print(f"\n Timestamp: {data.get('timestamp', 'N/A')}")

    config = data.get('config', {})
    print(f"\n️  Configuration:")
    print(f"URL:          {config.get('url', 'N/A')}")
    print(f"Requests:     {config.get('requests', 'N/A')}")
    print(f"Concurrency:  {config.get('concurrency', 'N/A')}")
    print(f"Timeout:      {config.get('timeout', 'N/A')}s")

    results = data.get('results', {})

    for test_name, result in results.items():
        print(f"\n{'='*80}")
        print(f"TEST: {test_name.upper()}")
        print(f"{'='*80}")

        total = result.get('total_requests', 0)
        success = result.get('successful_requests', 0)
        failed = result.get('failed_requests', 0)

        print(f"\n Overview:")
        print(f"Total Requests:      {total}")
        print(f"Successful:          {success} ({success/total*100:.1f}%)" if total > 0 else "Successful:          0")
        print(f"Failed:              {failed} ({failed/total*100:.1f}%)" if total > 0 else "Failed:              0")
        print(f"Total Duration:      {result.get('total_duration', 0):.2f}s")

        print(f"\n Throughput:")
        print(f"Requests/Second:     {result.get('requests_per_second', 0):.2f} req/s")

        print(f"\n️  Latency (seconds):")
        print(f"Min:                 {result.get('min_latency', 0):.3f}s")
        print(f"Max:                 {result.get('max_latency', 0):.3f}s")
        print(f"Average:             {result.get('avg_latency', 0):.3f}s")
        print(f"Median:              {result.get('median_latency', 0):.3f}s")
        print(f"95th Percentile:     {result.get('p95_latency', 0):.3f}s")
        print(f"99th Percentile:     {result.get('p99_latency', 0):.3f}s")

        errors = result.get('errors', [])
        if errors:
            print(f"\n Errors ({len(errors)}):")
            error_counts = {}
            for error in errors:
                error_counts[error] = error_counts.get(error, 0) + 1
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"[{count}x] {error}")

    print("\n" + "="*80)


def print_ascii_chart(values: List[float], title: str, width: int = 60):
    """Print simple ASCII bar chart"""

    if not values:
        return

    max_val = max(values)
    min_val = min(values)

    print(f"\n{title}")
    print("-" * width)

    for i, val in enumerate(values):
        bar_len = int((val - min_val) / (max_val - min_val) * (width - 20)) if max_val > min_val else 0
        bar = "#" * bar_len
        print(f"{i+1:3d} | {bar} {val:.3f}s")


def main():
    """Main function"""

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize_results.py <result_file.json>              # Detailed report")
        print("  python visualize_results.py <file1.json> <file2.json> ...  # Comparison")
        sys.exit(1)

    result_files = sys.argv[1:]

    # Check if files exist
    for file_path in result_files:
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)

    if len(result_files) == 1:
        # Single file - detailed report
        print_detailed_report(result_files[0])
    else:
        # Multiple files - comparison table
        print_comparison_table(result_files)


if __name__ == "__main__":
    main()
