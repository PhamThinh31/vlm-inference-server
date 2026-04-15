"""
Walk a data directory, run the full VLM pipeline on every subfolder
containing images, and dump per-folder + aggregate metrics to JSON.

Used to produce stress_test_metrics.json.
"""
import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from client.vlm4gis_client import VLM4GISClient

ROOT_DIR = "./data"
RESULT_FILENAME = "gis_result.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class PerformanceMonitor:
    def __init__(self):
        self.results = []
        self.folder_times = []
        self.folder_rates = []
        self.total_images = 0
        self.total_time = 0.0
        self.failed = 0

    def record(self, folder, num_images, duration, success, error=None):
        if success:
            rate = num_images / duration if duration > 0 else 0
            self.total_images += num_images
            self.total_time += duration
            self.folder_times.append(duration)
            self.folder_rates.append(rate)
            self.results.append({
                "folder": folder, "images": num_images,
                "duration": duration, "rate": rate, "success": True,
            })
            print(f"  ok: {num_images} images in {duration:.2f}s ({rate:.2f} img/s)")
        else:
            self.failed += 1
            self.results.append({
                "folder": folder, "images": num_images,
                "duration": duration, "success": False, "error": error,
            })
            print(f"  fail: {error}")

    def summary(self):
        print("\n=== summary ===")
        print(f"folders: {len(self.results)} ({self.failed} failed)")
        print(f"total images: {self.total_images}")
        print(f"total time: {self.total_time:.2f}s")
        if self.folder_rates:
            print(f"throughput avg/median/max: "
                  f"{statistics.mean(self.folder_rates):.2f} / "
                  f"{statistics.median(self.folder_rates):.2f} / "
                  f"{max(self.folder_rates):.2f} img/s")
        if self.total_time > 0:
            print(f"overall: {self.total_images / self.total_time:.2f} img/s")

    def save(self, path="performance_report.json"):
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "folders": len(self.results),
                "failed": self.failed,
                "total_images": self.total_images,
                "total_time": self.total_time,
                "avg_throughput": statistics.mean(self.folder_rates) if self.folder_rates else 0,
                "avg_latency": statistics.mean(self.folder_times) if self.folder_times else 0,
            },
            "folders": self.results,
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {path}")


def discover(root):
    folders = []
    for dirpath, _, files in os.walk(root):
        imgs = [f for f in files if Path(f).suffix.lower() in IMAGE_EXTS]
        if imgs:
            folders.append((Path(dirpath), len(imgs)))
    return folders


async def run():
    folders = discover(ROOT_DIR)
    print(f"found {len(folders)} folders ({sum(n for _, n in folders)} images) under {ROOT_DIR}")
    if not folders:
        return

    mon = PerformanceMonitor()
    async with VLM4GISClient() as client:
        try:
            await client.health_check()
        except Exception as e:
            print(f"server not reachable: {e}")
            return

        for i, (folder, n) in enumerate(folders, 1):
            name = f"{folder.parent.name}/{folder.name}"
            print(f"[{i}/{len(folders)}] {name} ({n} images)")
            t0 = time.time()
            try:
                result = await client.process_folder(
                    folder_path=str(folder),
                    session_name=name,
                )
                with open(folder / RESULT_FILENAME, "w") as f:
                    json.dump(result, f, indent=2)
                mon.record(name, n, time.time() - t0, True)
            except Exception as e:
                mon.record(name, n, time.time() - t0, False, str(e))

    mon.summary()
    mon.save()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        sys.exit(1)
