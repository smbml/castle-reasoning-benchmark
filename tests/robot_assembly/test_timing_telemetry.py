"""Sample actual local process counters without changing collector execution."""

import json
import os
import time

import castle_reasoning.runtime.diagnostics.telemetry as telemetry


def test_linux_process_counters_and_missing_pid_are_explicit():
    record = telemetry.process(os.getpid())
    assert record["pid"] == os.getpid()
    assert record["cpu_seconds"] >= 0
    assert record["rss_bytes"] > 0
    assert "read_bytes" in record["io"]
    assert telemetry.process(-1)["missing"]


def test_sampler_stops_and_records_monotonic_samples(tmp_path):
    with telemetry.Monitor(tmp_path, interval=0.01, gpu=False):
        time.sleep(0.035)
    records = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()
    ]
    assert len(records) >= 2
    assert all(b["time_ns"] >= a["time_ns"] for a, b in zip(records, records[1:]))
    assert all(r["processes"][0]["pid"] == os.getpid() for r in records)
    assert all("cgroup" in r for r in records)
    assert records[-1]["final"]
