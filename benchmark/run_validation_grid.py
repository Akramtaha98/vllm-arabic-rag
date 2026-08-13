"""
Orchestrates the SMALL VALIDATION GRID specified in Task 6 of the
network-aware experiment plan (paper/NETWORK_SPECIALIZATION_PLAN.md):

    - 2 network profiles: edge_lan, constrained_wireless
    - 3 concurrency levels: 1, 10, 50
    - 5 methods: raw, naive, fixed_lspm, kv_aware, network_aware
    - 3 repeats per cell
    = 2 x 3 x 5 x 3 = 90 cells

This is deliberately the SMALL validation grid, not the full Deliverable-5
experimental grid (which also sweeps ratio and all four network profiles).
Task 7 is explicit that the point of this run is to check whether the
instrumentation is trustworthy and whether the previously observed
high-concurrency KV-cache reversal (paper Section 5.8) persists in this new
edge-cloud topology, BEFORE spending GPU budget on the full grid.

REQUIRES:
    - A running cloud-tier vLLM server (same setup as BENCHMARK_INSTRUCTIONS.md)
    - A running edge gateway (benchmark/edge_gateway.py) reachable at --edge-host
    - A request manifest (benchmark/request_manifest.py), long enough for the
      grid's total request count (this script checks and refuses to proceed
      on an under-length manifest rather than silently wrapping it)
    - EITHER root/CAP_NET_ADMIN on the edge machine to shape --edge-iface with
      tc netem (--emulation-mode tc, the default), OR the edge gateway started
      with NETWORK_EMULATION_MODE=application (--emulation-mode application),
      for environments without CAP_NET_ADMIN -- confirmed necessary on the
      RunPod pod this was first deployed on: both `tc qdisc add ... netem`
      and `unshare --net ...` fail there with "Operation not permitted".
      See network_profiles.py's module docstring for what application-level
      emulation does and does not reproduce, and its required disclosure.

This script NEVER fabricates a missing measurement. Every cell's outcome is
one of: "ok" (locust + metrics scraper both exited cleanly and produced
non-empty output), or "failed" with a specific recorded reason (profile
apply/verify failed, warm-up failed, locust non-zero exit, metrics scraper
produced zero samples, etc.). Failed cells are retried up to --max-retries
times and then left as recorded failures -- never silently dropped,
replaced with a neighboring cell's numbers, or interpolated.

Usage:
    python benchmark/run_validation_grid.py \
        --edge-host http://localhost:9000 \
        --cloud-metrics-url http://<cloud-host>:8000/metrics \
        --edge-iface eth0 \
        --ping-target <cloud-host> \
        --run-time 60 \
        --repeats 3 \
        --out-dir results/validation_grid
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmark import network_profiles  # noqa: E402

PROFILES = ["edge_lan", "constrained_wireless"]
CONCURRENCY = [1, 10, 50]
METHODS = ["raw", "naive", "fixed_lspm", "kv_aware", "network_aware"]

# Rough upper bound on requests a single cell could issue, for the
# manifest-length check: 50 users x 1 req per ~0.5-2s wait_time x run_time.
# At run_time=60s and the tightest wait_time (0.5s), that's up to ~6000
# requests for the single largest cell alone.
MAX_REQUESTS_PER_CELL_ESTIMATE = 6000


def cell_tag(profile, method, users, rep):
    return f"{profile}_{method}_c{users}_rep{rep}"


def run_cmd(cmd, **kwargs):
    print("  $ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-host", required=True, help="Edge gateway base URL, e.g. http://localhost:9000")
    ap.add_argument("--cloud-metrics-url", required=True, help="Cloud vLLM /metrics URL, for KV-cache sampling")
    ap.add_argument("--edge-iface", required=True, help="Network interface on the edge machine facing the cloud tier, shaped by tc netem")
    ap.add_argument("--ping-target", required=True, help="Cloud-tier host, for profile verification (ping RTT)")
    ap.add_argument("--run-time", type=int, default=60)
    ap.add_argument("--spawn-rate", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup-requests", type=int, default=10)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--manifest-path", default=str(ROOT / "benchmark" / "request_manifest.json"))
    ap.add_argument("--out-dir", default="results/validation_grid")
    ap.add_argument("--gpu-util", action="store_true", help="Also sample nvidia-smi during each cell (see metrics_scraper.py)")
    ap.add_argument("--emulation-mode", choices=["tc", "application"], default="tc",
                     help="'tc' (default): kernel-level tc netem on --edge-iface, requires CAP_NET_ADMIN. "
                          "'application': skip tc entirely and rely on the edge gateway process having been "
                          "started with NETWORK_EMULATION_MODE=application, which injects delay/jitter/loss/"
                          "bandwidth-throttle in software per request. Use 'application' if "
                          "`python benchmark/network_profiles.py --iface <iface> --check-capability` reports "
                          "tc is NOT AVAILABLE.")
    args = ap.parse_args()

    if args.emulation_mode == "application":
        print("=" * 78)
        print("--emulation-mode application: this script will NOT touch --edge-iface at all.")
        print("Network shaping only happens if the EDGE GATEWAY PROCESS was started with")
        print("NETWORK_EMULATION_MODE=application. If it was started with that unset (or 'none'),")
        print("every cell below will silently run over the real UNSHAPED link. Verify now with:")
        print(f"  curl -s {args.edge_host.rstrip('/')}/health")
        print("and confirm the response includes \"network_emulation_mode\": \"application\" before proceeding.")
        print("=" * 78)

    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        sys.exit(f"Manifest not found at {manifest_path}. Run: python benchmark/request_manifest.py --out {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_cells = len(PROFILES) * len(CONCURRENCY) * len(METHODS) * args.repeats
    needed = n_cells * MAX_REQUESTS_PER_CELL_ESTIMATE  # conservative: every cell reads from position 0 again
    # NOTE: each locust invocation below is a fresh process, and locustfile_edge.py's
    # _MANIFEST_INDEX restarts at 0 each time -- so every cell independently needs
    # up to MAX_REQUESTS_PER_CELL_ESTIMATE manifest entries, not the grid total.
    if manifest["length"] < MAX_REQUESTS_PER_CELL_ESTIMATE:
        print(f"WARNING: manifest length ({manifest['length']}) is smaller than the conservative "
              f"single-cell upper bound ({MAX_REQUESTS_PER_CELL_ESTIMATE}). At c=50 with a fast "
              f"wait_time draw, the largest cells could exhaust the manifest and raise IndexError "
              f"(locustfile_edge.py deliberately does not wrap). Regenerate with --length "
              f">= {MAX_REQUESTS_PER_CELL_ESTIMATE} if any cell fails with 'manifest exhausted'.")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "grid_run_log.jsonl"

    def _append_log(record: dict):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    results = []
    for profile in PROFILES:
        if args.emulation_mode == "tc":
            print(f"\n=== Applying network profile: {profile} on {args.edge_iface} (tc netem) ===")
            try:
                applied = network_profiles.apply_profile(args.edge_iface, profile)
            except Exception as e:
                print(f"FATAL: could not apply profile {profile!r}: {e}")
                print("Every cell under this profile will be recorded as failed with reason 'profile_apply_failed'. "
                      "NOT substituting an unshaped link and pretending the profile was applied.")
                for method in METHODS:
                    for users in CONCURRENCY:
                        for rep in range(1, args.repeats + 1):
                            rec = {"cell": cell_tag(profile, method, users, rep), "status": "failed",
                                   "reason": "profile_apply_failed", "detail": str(e)}
                            results.append(rec)
                            _append_log(rec)
                continue

            verified = network_profiles.verify_profile(profile, args.ping_target, count=15)
            print(f"  Verified: requested_delay={verified.requested_delay_ms}ms "
                  f"measured_rtt_mean={verified.measured_rtt_mean_ms}ms "
                  f"measured_rtt_stddev={verified.measured_rtt_stddev_ms}ms loss={verified.ping_loss_pct}%")
            if verified.measured_rtt_mean_ms is None:
                print(f"FATAL: profile {profile!r} applied but verification ping failed entirely. "
                      "Recording every cell under this profile as failed rather than trusting an unverified link.")
                for method in METHODS:
                    for users in CONCURRENCY:
                        for rep in range(1, args.repeats + 1):
                            rec = {"cell": cell_tag(profile, method, users, rep), "status": "failed",
                                   "reason": "profile_verify_failed"}
                            results.append(rec)
                            _append_log(rec)
                continue

            _append_log({"event": "profile_applied", "profile": profile, "emulation_mode": "tc",
                          "requested": applied.requested, "verified": verified.__dict__})
        else:
            # application mode: no tc call at all. The gateway process (started
            # separately with NETWORK_EMULATION_MODE=application) applies
            # delay/jitter/loss/throttle per request, keyed by the profile_name
            # sent in each request -- see locustfile_edge.py / edge_gateway.py.
            # We still ping the real link for disclosure of the UNSHAPED
            # baseline, but explicitly label it as such so it is never mistaken
            # for the applied condition.
            print(f"\n=== Profile: {profile} (application-level emulation, gateway-side) ===")
            baseline = network_profiles.verify_profile(profile, args.ping_target, count=15)
            print(f"  Baseline (UNSHAPED) link to {args.ping_target}: "
                  f"measured_rtt_mean={baseline.measured_rtt_mean_ms}ms loss={baseline.ping_loss_pct}% "
                  f"-- actual emulated delay/jitter/loss is applied inside the gateway process, "
                  f"see its REQUEST_LOG_PATH's applied_emulation field per request.")
            _append_log({"event": "profile_baseline_measured_application_mode", "profile": profile,
                          "emulation_mode": "application",
                          "note": "baseline is the real unshaped link; applied emulation is logged "
                                  "per-request by edge_gateway.py, not here",
                          "baseline_unshaped": baseline.__dict__})

        for users in CONCURRENCY:
            for method in METHODS:
                for rep in range(1, args.repeats + 1):
                    tag = cell_tag(profile, method, users, rep)
                    attempt = 0
                    cell_result = None
                    while attempt <= args.max_retries and cell_result is None:
                        attempt += 1
                        print(f"\n[{tag}] attempt {attempt}/{args.max_retries + 1}")

                        warm = run_cmd([sys.executable, str(ROOT / "benchmark" / "warmup.py"),
                                         "--host", args.edge_host, "--n", str(args.warmup_requests),
                                         "--gateway-mode", "--method", method],
                                        cwd=str(ROOT))
                        if warm.returncode != 0:
                            cell_result = {"cell": tag, "status": "failed", "reason": "warmup_failed", "attempt": attempt}
                            continue

                        metrics_csv = out_dir / f"metrics_{tag}.csv"
                        metrics_cmd = [sys.executable, str(ROOT / "benchmark" / "metrics_scraper.py"),
                                       "--url", args.cloud_metrics_url, "--interval", "1.0",
                                       "--duration", str(args.run_time + 10), "--out", str(metrics_csv)]
                        if args.gpu_util:
                            metrics_cmd.append("--gpu-util")
                        metrics_proc = subprocess.Popen(metrics_cmd)
                        time.sleep(1)

                        locust_prefix = str(out_dir / f"locust_{tag}")
                        locust_cmd = [
                            "locust", "-f", str(ROOT / "benchmark" / "locustfile_edge.py"),
                            "--host", args.edge_host, "--users", str(users),
                            "--spawn-rate", str(args.spawn_rate), "--run-time", f"{args.run_time}s",
                            "--headless", "--csv", locust_prefix,
                        ]
                        full_env = os.environ.copy()
                        full_env.update({"METHOD": method, "PROFILE_NAME": profile, "MANIFEST_PATH": args.manifest_path})
                        locust_result = run_cmd(locust_cmd, cwd=str(ROOT), env=full_env)
                        metrics_proc.wait(timeout=args.run_time + 30)

                        stats_path = Path(f"{locust_prefix}_stats.csv")
                        metrics_ok = metrics_csv.exists() and metrics_csv.stat().st_size > 0
                        locust_ok = locust_result.returncode == 0 and stats_path.exists() and stats_path.stat().st_size > 0

                        if not locust_ok:
                            cell_result = {"cell": tag, "status": "failed", "reason": "locust_failed_or_empty",
                                           "locust_returncode": locust_result.returncode, "attempt": attempt}
                        elif not metrics_ok:
                            cell_result = {"cell": tag, "status": "failed", "reason": "metrics_scraper_empty", "attempt": attempt}
                        else:
                            cell_result = {"cell": tag, "status": "ok", "attempt": attempt,
                                           "locust_stats": str(stats_path), "metrics_csv": str(metrics_csv)}

                    results.append(cell_result)
                    _append_log(cell_result)
                    print(f"[{tag}] -> {cell_result['status']}" + (f" ({cell_result.get('reason')})" if cell_result["status"] == "failed" else ""))

        if args.emulation_mode == "tc":
            network_profiles.reset_profile(args.edge_iface)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\n=== Validation grid complete: {n_ok} ok, {n_failed} failed, {len(results)} total cells ===")
    if n_failed:
        print("Failed cells (reason, count):")
        from collections import Counter
        reasons = Counter(r.get("reason", "?") for r in results if r["status"] == "failed")
        for reason, count in reasons.most_common():
            print(f"  {reason}: {count}")
    print(f"Full per-cell log: {log_path}")


if __name__ == "__main__":
    main()
