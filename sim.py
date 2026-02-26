#!/usr/bin/env python3
"""Quick CLI to simulate the CS amp with explicit W1, L1, RD values.

Usage:
    python sim.py --W1 10e-6 --L1 0.25e-6 --RD 6400
    python sim.py --W1 10u --L1 0.25u --RD 6.4k   # SI suffixes also accepted
"""

import argparse
import sys

from circuitrl.simulators.ngspice_runner import NGSpiceRunner

TEMPLATE = "circuitrl/envs/cs_amp_template.sp"
TARGETS  = {"gain_db": 20.0, "bandwidth": 50e6}


def parse_si(s: str) -> float:
    """Parse a float string with optional SI suffix (u, n, p, k, M, G)."""
    suffixes = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6,
                "m": 1e-3,  "k": 1e3,   "M": 1e6,  "G": 1e9}
    s = s.strip()
    if s[-1] in suffixes:
        return float(s[:-1]) * suffixes[s[-1]]
    return float(s)


def main():
    parser = argparse.ArgumentParser(description="Simulate CS amp")
    parser.add_argument("--W1", required=True, help="NMOS width  (e.g. 10e-6 or 10u)")
    parser.add_argument("--L1", required=True, help="NMOS length (e.g. 0.25e-6 or 0.25u)")
    parser.add_argument("--RD",  required=True, help="Drain resistor in Ω (e.g. 6400 or 6.4k)")
    args = parser.parse_args()

    W1 = parse_si(args.W1)
    L1 = parse_si(args.L1)
    RD = parse_si(args.RD)

    print(f"Parameters:  W1={W1:.3e}  L1={L1:.3e}  RD={RD:.1f}  (W/L={W1/L1:.1f})")

    runner = NGSpiceRunner(TEMPLATE, timeout=15, expected_metrics=("gain_db", "bandwidth"))
    result = runner.run({"W1": f"{W1:.6e}", "L1": f"{L1:.6e}", "RD": f"{RD:.6e}"})

    if result is None:
        print("Simulation failed (timeout or parse error).")
        sys.exit(1)

    print()
    print(f"{'Metric':<12} {'Simulated':>14} {'Target':>14} {'%err':>8}")
    print("-" * 52)
    for name, target in TARGETS.items():
        val = result[name]
        pct_err = 100.0 * abs(val - target) / abs(target) if target != 0 else float("inf")
        print(f"{name:<12} {val:>14.4g} {target:>14.4g} {pct_err:>7.1f}%")


if __name__ == "__main__":
    main()
