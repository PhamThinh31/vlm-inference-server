"""
Quick async load test against a single endpoint.

Usage:
  python quick_load_test.py <url> [requests=50] [concurrency=5]
"""
import asyncio
import statistics
import sys
import time

import aiohttp


async def fetch(session, url, rid):
    t0 = time.time()
    try:
        async with session.get(url) as r:
            await r.read()
            return {"id": rid, "dt": time.time() - t0,
                    "status": r.status, "ok": r.status < 400}
    except Exception as e:
        return {"id": rid, "dt": time.time() - t0,
                "status": 0, "ok": False, "err": str(e)}


async def run(url, n, conc):
    print(f"{url}  requests={n} concurrency={conc}")
    sem = asyncio.Semaphore(conc)
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=conc)

    async def bounded(i):
        async with sem:
            return await fetch(session, url, i)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        t0 = time.time()
        results = await asyncio.gather(*(bounded(i) for i in range(n)))
        total = time.time() - t0

    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    dts = sorted(r["dt"] for r in ok)

    if not dts:
        print("all requests failed")
        for r in fail[:5]:
            print(" ", r.get("err"))
        return

    print(f"time={total:.2f}s ok={len(ok)}/{n} "
          f"throughput={len(ok)/total:.2f} req/s")
    print(f"latency min/median/avg/p95/p99/max = "
          f"{dts[0]:.3f}/{statistics.median(dts):.3f}/"
          f"{statistics.mean(dts):.3f}/"
          f"{dts[int(len(dts)*0.95)]:.3f}/"
          f"{dts[int(len(dts)*0.99)]:.3f}/{dts[-1]:.3f} s")
    if fail:
        print(f"errors: {len(fail)} (showing first 5)")
        for r in fail[:5]:
            print(f"  [{r['id']}] {r.get('err', r['status'])}")


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)
    url = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    c = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    try:
        asyncio.run(run(url, n, c))
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()
