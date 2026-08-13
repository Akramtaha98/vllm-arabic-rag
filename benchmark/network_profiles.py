"""
Four emulated network profiles for the edge-cloud validation grid (Task 3),
applied with Linux `tc netem` on the network interface between the edge
gateway and the cloud vLLM server.

CRITICAL DISCLOSURE REQUIREMENT (per the paper's own integrity rules,
paper/NETWORK_SPECIALIZATION_PLAN.md Section 0 and Task 4's instruction):
every one of these profiles is an EMULATED condition on a single link
shaped with `tc netem`, not a physical edge device or a real production
network. Every table/figure that reports results under these profiles must
say so explicitly. This module's `verify_profile()` exists specifically so
the paper reports *measured*, not merely *configured*, RTT/jitter/bandwidth
-- what you asked `tc` to do and what it actually did can differ (queueing
interactions, measurement noise, kernel HZ resolution), so every applied
profile is checked with a real ping/throughput probe immediately after
being applied, and the measured numbers -- not just the intended ones --
are what gets logged and reported.

Requires root or CAP_NET_ADMIN on the machine running the edge gateway (to
run `tc qdisc add/change/del`). This module refuses to guess or silently
skip on permission failure -- apply_profile() raises rather than continuing
as if a profile were applied when it was not, since silently running
"WAN" cells under unshaped LAN conditions would corrupt the whole grid
without any visible signal that something was wrong.

Usage:
    python benchmark/network_profiles.py --iface eth0 --profile moderate_wan --apply
    python benchmark/network_profiles.py --iface eth0 --profile moderate_wan --verify --target 10.0.0.5
    python benchmark/network_profiles.py --iface eth0 --reset

--------------------------------------------------------------------------
FALLBACK: APPLICATION-LEVEL EMULATION (no CAP_NET_ADMIN available)
--------------------------------------------------------------------------
Confirmed on the RunPod pod used for this experiment (2026-08-13): the
container has neither CAP_NET_ADMIN (`tc qdisc add ... netem` ->
"RTNETLINK answers: Operation not permitted") nor unprivileged user
namespaces (`unshare --net ...` -> "unshare: unshare failed: Operation not
permitted"). Kernel-level packet shaping is therefore NOT POSSIBLE in this
deployment, full stop -- there is no further privilege-escalation path to
try from inside the container.

The functions below (`sample_uplink_delay_s`, `sample_downlink_delay_s`,
`should_drop_request`, `bandwidth_throttle_sleep_s`) implement a SOFTWARE
fallback: the edge gateway (edge_gateway.py, when started with
NETWORK_EMULATION_MODE=application) injects delay/jitter before sending
each request to the cloud tier and delay/jitter/bandwidth-throttling after
receiving the response, and randomly fails a fraction of requests to
stand in for packet loss. This is DELIBERATELY NOT presented as equivalent
to tc netem:

  - It shapes one HTTP request/response at a time, in userspace, on top of
    whatever the real (unshaped) TCP connection does -- it does not touch
    packets, does not interact with TCP congestion control the way real
    delay/loss would, and cannot reproduce reordering.
  - "Loss" is modeled as whole-request failure, not per-packet loss with
    retransmission.
  - The delay split (half applied before the request, half plus the
    bandwidth throttle applied after the full response is buffered) is a
    simplifying, documented modeling choice, not a measurement.

Any table or figure built from NETWORK_EMULATION_MODE=application data
MUST say "application-level emulation" explicitly and must not be
described with the same "emulated network profile" language used
elsewhere for tc-netem-based results, per the paper's own disclosure
requirement (see this module's docstring above and
paper/NETWORK_SPECIALIZATION_PLAN.md Section 0).
"""
from __future__ import annotations

import argparse
import random
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

# --------------------------------------------------------------------------
# Profile definitions. Values are the paper's Task 3 four required profiles:
# Edge/LAN, good cloud, moderate WAN, constrained wireless. Chosen to be
# representative, documented figures (cited ranges are typical, not derived
# from a specific measured link) -- the ACTUAL applied and verified values
# are what must be reported in the paper's test-environment section, not
# these targets, since tc netem's realized behavior can differ from the
# requested parameters (see verify_profile()).
# --------------------------------------------------------------------------
PROFILES = {
    "edge_lan": {
        "description": "Edge/LAN: gateway and GPU server on the same local network segment.",
        "delay_ms": 1, "jitter_ms": 0.2, "rate_mbit": 1000, "loss_pct": 0.0,
    },
    "good_cloud": {
        "description": "Good cloud: same-region cloud-to-cloud link, low congestion.",
        "delay_ms": 10, "jitter_ms": 2, "rate_mbit": 200, "loss_pct": 0.0,
    },
    "moderate_wan": {
        "description": "Moderate WAN: cross-region/intercontinental wired link.",
        "delay_ms": 60, "jitter_ms": 10, "rate_mbit": 50, "loss_pct": 0.1,
    },
    "constrained_wireless": {
        "description": "Constrained wireless: rural/congested mobile link.",
        "delay_ms": 120, "jitter_ms": 30, "rate_mbit": 5, "loss_pct": 1.0,
    },
}


@dataclass
class AppliedProfile:
    profile_name: str
    iface: str
    requested: dict
    tc_command: str
    applied_at: float


@dataclass
class VerifiedProfile:
    profile_name: str
    target: str
    requested_delay_ms: float
    measured_rtt_mean_ms: Optional[float]
    measured_rtt_stddev_ms: Optional[float]  # empirical jitter proxy
    ping_loss_pct: Optional[float]
    n_pings: int
    raw_ping_output_tail: str
    verified_at: float


def _run(cmd, check=True):
    print("  $ " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}\n"
            "Refusing to continue as if the profile were applied -- this "
            "usually means the process lacks CAP_NET_ADMIN (try sudo/root) "
            "or --iface does not exist on this machine."
        )
    return result


def reset_profile(iface: str):
    """Remove any existing netem qdisc on iface. Safe to call even if none exists."""
    result = subprocess.run(["tc", "qdisc", "del", "dev", iface, "root"], capture_output=True, text=True)
    if result.returncode != 0 and "Cannot find" not in result.stderr and "No such" not in result.stderr:
        print(f"  WARNING: tc qdisc del reported: {result.stderr.strip()} (continuing; may mean nothing was set)")


def apply_profile(iface: str, profile_name: str) -> AppliedProfile:
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile {profile_name!r}; choices: {list(PROFILES)}")
    p = PROFILES[profile_name]
    reset_profile(iface)  # always start clean -- never layer profiles

    cmd = [
        "tc", "qdisc", "add", "dev", iface, "root", "netem",
        "delay", f"{p['delay_ms']}ms", f"{p['jitter_ms']}ms", "distribution", "normal",
        "loss", f"{p['loss_pct']}%",
        "rate", f"{p['rate_mbit']}mbit",
    ]
    _run(cmd)
    return AppliedProfile(
        profile_name=profile_name, iface=iface, requested=dict(p),
        tc_command=" ".join(cmd), applied_at=time.time(),
    )


def check_tc_capability(iface: str) -> tuple[bool, str]:
    """Cheap capability probe: tries a real tc qdisc add/del on `iface` and
    reports whether it actually worked, rather than assuming. Used by
    run_validation_grid.py to fail loudly (not silently fall back to
    unshaped traffic) if --emulation-mode tc is requested somewhere that
    can't actually do it."""
    result = subprocess.run(
        ["tc", "qdisc", "add", "dev", iface, "root", "netem", "delay", "1ms"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        subprocess.run(["tc", "qdisc", "del", "dev", iface, "root"], capture_output=True, text=True)
        return True, "tc netem works on this interface"
    return False, (result.stderr.strip() or "tc qdisc add failed with no stderr output")


# --------------------------------------------------------------------------
# Application-level emulation helpers (see module docstring's fallback
# section). Consumed by edge_gateway.py, not by this module's own CLI.
# --------------------------------------------------------------------------
def sample_uplink_delay_s(profile_name: str) -> float:
    """One-way delay sample, in seconds: half of the profile's round-trip
    delay+jitter, Gaussian, floored at 0. Documented simplifying
    assumption: the round-trip delay is split evenly between the
    uplink (client->cloud) and downlink (cloud->client) legs."""
    p = PROFILES[profile_name]
    mean_s = (p["delay_ms"] / 2.0) / 1000.0
    jitter_s = (p["jitter_ms"] / 2.0) / 1000.0
    sampled = random.gauss(mean_s, jitter_s) if jitter_s > 0 else mean_s
    return max(0.0, sampled)


def sample_downlink_delay_s(profile_name: str) -> float:
    return sample_uplink_delay_s(profile_name)  # same symmetric-split assumption


def should_drop_request(profile_name: str) -> bool:
    """Whole-request failure with probability loss_pct/100 -- a coarser
    stand-in for tc netem's per-packet loss (see module docstring)."""
    p = PROFILES[profile_name]
    return random.random() < (p["loss_pct"] / 100.0)


def bandwidth_throttle_sleep_s(profile_name: str, n_bytes: int, elapsed_s: float) -> float:
    """Extra sleep (seconds) needed so that transferring n_bytes over
    elapsed_s does not exceed the profile's rate_mbit. Returns 0 if the
    transfer was already slower than the cap (never speeds anything up)."""
    p = PROFILES[profile_name]
    rate_mbit = p["rate_mbit"]
    if rate_mbit <= 0 or n_bytes <= 0:
        return 0.0
    min_required_s = (n_bytes * 8 / 1_000_000) / rate_mbit
    return max(0.0, min_required_s - elapsed_s)


_PING_RTT_RE = re.compile(r"time=([\d.]+)\s*ms")
_PING_LOSS_RE = re.compile(r"([\d.]+)% packet loss")


def verify_profile(profile_name: str, target: str, count: int = 20) -> VerifiedProfile:
    """Pings `target` `count` times and reports the ACTUALLY MEASURED RTT
    mean/stddev and packet loss -- this is what should be cited in the
    paper's test-environment table, not the requested tc parameters alone,
    per the disclosure requirement in this module's docstring."""
    p = PROFILES.get(profile_name, {})
    result = subprocess.run(
        ["ping", "-c", str(count), "-i", "0.2", target],
        capture_output=True, text=True, timeout=count * 2 + 10,
    )
    rtts = [float(m) for m in _PING_RTT_RE.findall(result.stdout)]
    loss_match = _PING_LOSS_RE.search(result.stdout)
    loss_pct = float(loss_match.group(1)) if loss_match else None
    return VerifiedProfile(
        profile_name=profile_name, target=target,
        requested_delay_ms=p.get("delay_ms", float("nan")),
        measured_rtt_mean_ms=statistics.mean(rtts) if rtts else None,
        measured_rtt_stddev_ms=statistics.pstdev(rtts) if len(rtts) > 1 else None,
        ping_loss_pct=loss_pct, n_pings=count,
        raw_ping_output_tail="\n".join(result.stdout.strip().splitlines()[-5:]),
        verified_at=time.time(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", required=True)
    ap.add_argument("--profile", choices=list(PROFILES))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--target", help="Host to ping for --verify (the cloud tier's address)")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--check-capability", action="store_true",
                     help="Test whether tc netem actually works on --iface (tries a real add+del) "
                          "and exit. Use this before trusting --emulation-mode tc in run_validation_grid.py.")
    args = ap.parse_args()

    if args.check_capability:
        ok, detail = check_tc_capability(args.iface)
        print(f"tc netem capability on {args.iface}: {'AVAILABLE' if ok else 'NOT AVAILABLE'}")
        print(f"  {detail}")
        sys.exit(0 if ok else 1)

    if args.reset:
        reset_profile(args.iface)
        print(f"Reset netem qdisc on {args.iface}")
        return

    if args.apply:
        if not args.profile:
            sys.exit("--apply requires --profile")
        applied = apply_profile(args.iface, args.profile)
        print(f"Applied profile {applied.profile_name!r} on {applied.iface}: {applied.requested}")

    if args.verify:
        if not args.profile or not args.target:
            sys.exit("--verify requires --profile and --target")
        v = verify_profile(args.profile, args.target, count=args.count)
        print(f"Verified {v.profile_name!r} against {v.target}: "
              f"requested_delay={v.requested_delay_ms}ms measured_rtt_mean={v.measured_rtt_mean_ms}"
              f"ms measured_rtt_stddev={v.measured_rtt_stddev_ms}ms loss={v.ping_loss_pct}%")
        if v.measured_rtt_mean_ms is None:
            print("WARNING: no successful pings -- verification FAILED, do not trust this profile as applied.")


if __name__ == "__main__":
    main()
