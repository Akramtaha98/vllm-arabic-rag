"""
Smoke test for the network-aware edge-cloud experiment code, meant to be
run BEFORE renting a GPU. Exercises every module that does not require a
live vLLM server or a real network link: imports, the controller's
decision logic, the manifest generator's reproducibility, the network
profile ping-parser, the edge gateway's request/response/logging cycle
(against a deliberately unreachable cloud URL, so failure handling is what
gets tested), and the metrics scraper's and warm-up script's graceful
failure paths.

This does NOT start vLLM, does NOT apply a real tc netem shape, and does
NOT need a GPU. It exists to catch import errors, broken CLI wiring, and
logic bugs cheaply, so pod time isn't spent debugging things that don't
need a GPU to debug. Run it again inside the pod, right after cloning the
repo and installing dependencies, as a first sanity check before starting
vLLM -- the environment (Python version, installed packages) can differ
from wherever this was last run.

Usage:
    pip install -r benchmark/requirements-benchmark.txt
    python benchmark/smoke_test.py

Exit code 0 = all checks passed. Non-zero = at least one check failed;
the specific failure is printed, not swallowed.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = []


def check(name, fn):
    print(f"\n--- {name} ---")
    try:
        fn()
        print(f"PASS: {name}")
        RESULTS.append((name, True, None))
    except Exception as e:
        print(f"FAIL: {name}: {e}")
        RESULTS.append((name, False, str(e)))


def check_imports():
    import benchmark.sample_queries  # noqa
    import benchmark.request_manifest  # noqa
    import benchmark.network_controller  # noqa
    import benchmark.network_profiles  # noqa
    import benchmark.metrics_scraper  # noqa
    import benchmark.warmup  # noqa
    # edge_gateway needs CLOUD_VLLM_URL set to import (module-level env read)
    os.environ.setdefault("CLOUD_VLLM_URL", "http://127.0.0.1:1")
    os.environ.setdefault("PRECOMPUTED_CONTEXTS_PATH", str(_dummy_precomputed()))
    os.environ.setdefault("REQUEST_LOG_PATH", str(Path(tempfile.mkdtemp()) / "smoke_log.jsonl"))
    import benchmark.edge_gateway  # noqa


def _dummy_precomputed():
    p = Path(tempfile.mkdtemp()) / "empty_precomputed.json"
    p.write_text("{}", encoding="utf-8")
    return p


def check_manifest_reproducible():
    from benchmark.request_manifest import build_manifest
    from benchmark.sample_queries import SAMPLE_QUERIES
    m1 = build_manifest(200, 12345, SAMPLE_QUERIES)
    m2 = build_manifest(200, 12345, SAMPLE_QUERIES)
    assert m1 == m2, "same seed produced different manifests"
    assert len(m1) == 200


def check_controller_logic():
    from benchmark.network_controller import NetworkAwareController
    c = NetworkAwareController()
    best = c.decide(bandwidth_mbps=100, rtt_ms=1, queue_length=0, concurrency=1, kv_cache_usage_pct=0)
    assert best.ratio_selected == 0.7, f"expected 0.7, got {best.ratio_selected}"
    worst = c.decide(bandwidth_mbps=1, rtt_ms=150, queue_length=50, concurrency=50, kv_cache_usage_pct=100)
    assert worst.ratio_selected == 0.3, f"expected 0.3, got {worst.ratio_selected}"
    missing = c.decide(bandwidth_mbps=None, rtt_ms=20, queue_length=None, concurrency=5, kv_cache_usage_pct=40)
    assert missing.fallback_used
    try:
        NetworkAwareController(weights={"bandwidth_mbps": 1, "rtt_ms": 1, "queue_length": 0,
                                          "concurrency": 0, "kv_cache_usage_pct": 0})
        raise AssertionError("bad weights should have raised ValueError")
    except ValueError:
        pass


def check_network_profile_parser():
    from benchmark.network_profiles import _PING_RTT_RE, _PING_LOSS_RE, PROFILES
    assert len(PROFILES) == 4, f"expected 4 profiles, got {len(PROFILES)}"
    sample = "64 bytes from x: icmp_seq=1 ttl=64 time=59.8 ms\n3 packets transmitted, 3 received, 0% packet loss"
    rtts = [float(m) for m in _PING_RTT_RE.findall(sample)]
    assert rtts == [59.8]
    loss = _PING_LOSS_RE.search(sample)
    assert loss.group(1) == "0"


def check_metrics_scraper_graceful_failure():
    from benchmark.metrics_scraper import scrape_once, scrape_gpu_util
    val, err = scrape_once("http://127.0.0.1:1/metrics")
    assert val is None and err is not None
    util, mem, total, gpu_err = scrape_gpu_util()  # fine either way (GPU present or not), must not raise


def check_warmup_fails_cleanly():
    result = subprocess.run(
        [sys.executable, str(ROOT / "benchmark" / "warmup.py"), "--host", "http://127.0.0.1:1", "--n", "1", "--timeout", "2"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode != 0, "warmup.py should exit non-zero when every request fails"


def check_edge_gateway_live_server():
    import httpx
    port = 9877
    env = os.environ.copy()
    env["CLOUD_VLLM_URL"] = "http://127.0.0.1:1"
    log_path = Path(tempfile.mkdtemp()) / "gw_smoke_log.jsonl"
    env["REQUEST_LOG_PATH"] = str(log_path)
    env["PRECOMPUTED_CONTEXTS_PATH"] = str(_dummy_precomputed())

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "benchmark.edge_gateway:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.time() + 10
        up = False
        while time.time() < deadline:
            try:
                r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
                if r.status_code == 200:
                    up = True
                    break
            except Exception:
                time.sleep(0.3)
        assert up, "edge gateway did not come up within 10s"

        r = httpx.post(f"http://127.0.0.1:{port}/generate", json={
            "query": "ما هو مركز أبحاث الذكاء الاصطناعي؟", "method": "raw", "profile_name": "smoke",
        }, timeout=10.0)
        assert r.status_code == 200
        data = r.json()
        assert "error" in data, f"expected a graceful error against unreachable cloud, got: {data}"

        assert log_path.exists() and log_path.stat().st_size > 0, "gateway did not write to REQUEST_LOG_PATH"
        line = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
        assert line["status"] == "exception"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    check("imports", check_imports)
    check("request manifest reproducibility", check_manifest_reproducible)
    check("network-aware controller logic", check_controller_logic)
    check("network profile ping-parser", check_network_profile_parser)
    check("metrics scraper graceful failure", check_metrics_scraper_graceful_failure)
    check("warmup.py fails cleanly on unreachable host", check_warmup_fails_cleanly)
    check("edge gateway live server + logging", check_edge_gateway_live_server)

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = len(RESULTS) - n_pass
    print(f"\n=== Smoke test: {n_pass}/{len(RESULTS)} passed ===")
    if n_fail:
        print("FAILED checks:")
        for name, ok, err in RESULTS:
            if not ok:
                print(f"  - {name}: {err}")
        sys.exit(1)
    print("All checks passed. Safe to proceed to renting a GPU.")


if __name__ == "__main__":
    main()
