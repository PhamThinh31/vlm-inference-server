# Tests

Stress-test harness for the inference server. These are not unit tests — they
hit a running server over HTTP and measure throughput/latency.

Start the server first (see the project README), then:

```bash
pip install -r tests/requirements-test.txt
```

### Quick check

```bash
python tests/quick_load_test.py http://localhost:8000/health 100 10
```

Arguments: `<url> [requests=50] [concurrency=5]`.

### Concurrent-request correctness

```bash
python tests/test_concurrent_requests.py
```

Fires overlapping pipeline requests to make sure session state and the
vision-encoding cache stay consistent under concurrency.

### Full data run (produces `stress_test_metrics.json`)

Walks every subfolder of `./data` that contains images, runs the full pipeline
on each, writes a per-folder `gis_result.json` plus an aggregate metrics file.

```bash
python tests/process_folders_with_metrics.py
```

There's also `stress_test_with_data.py` for a lower-level single-task loader
that exercises upload + classify + validate without the pipeline wrapper —
useful when isolating which stage is the bottleneck.

### Visualising results

```bash
python tests/visualize_results.py performance_report.json
```
