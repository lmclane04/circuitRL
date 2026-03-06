#!/usr/bin/env python3
"""Analytical CS amp solver: given gain (dB) and bandwidth (Hz), compute W/L and RD.

Constants from cs_amp_template.sp:
  KP     = 200e-6  A/V²   (NMOS KP)
  VTO    = 0.5     V      (NMOS threshold)
  LAMBDA = 0.04    V⁻¹   (channel-length modulation)
  CL     = 0.5e-12 F      (load cap, fixed in netlist)
  VG     = 0.7     V      (DC gate bias, Vin DC)

Equations (from small-signal analysis):
  Rout = 1 / (2π B CL)                         [BW sets output pole]
  ro   = 1 / (λ π B CL G (VG − VT))            [MOSFET output resistance]
  RD   = Rout·ro / (ro − Rout)                  [RD || ro = Rout]
  W/L  = 2π B CL G / (KP (VG − VT))            [gm = G/Rout]

Usage:
  python solve.py                                # default targets from config
  python solve.py --gain-db 20 --bw 50e6
  python solve.py --gain-db 20 --bw 50e6 --simulate
"""

import argparse
import math
import sys

import numpy as np

# ── Circuit constants (from cs_amp_template.sp) ──────────────────────────────
KP     = 200e-6   # A/V²
VTO    = 0.5      # V
LAMBDA = 0.04     # V⁻¹
CL     = 0.5e-12  # F
VG     = 0.7      # V  (DC gate bias)
VGS_VT = VG - VTO  # = 0.2 V

# ── Discrete parameter grids (from cs_amp.yaml) ───────────────────────────────
W1_GRID = np.arange(1e-6,  50e-6  + 0.5e-6,  1e-6)   # 1–50 µm, 1 µm step
L1_GRID = np.arange(0.18e-6, 5e-6 + 0.05e-6, 0.1e-6)  # 0.18–5 µm, 0.1 µm step
RD_GRID = np.arange(1000,  50000  + 250,      500)     # 1–50 kΩ, 500 Ω step
W1_MIN_RATIO = 5  # constraint: W1 >= 5 * L1


def solve(gain_db: float, bw_hz: float) -> dict:
    G = 10 ** (gain_db / 20.0)   # linear voltage gain
    B = bw_hz

    Rout = 1.0 / (2 * math.pi * B * CL)
    ro   = 1.0 / (LAMBDA * math.pi * B * CL * G * VGS_VT)
    WL   = (2 * math.pi * B * CL * G) / (KP * VGS_VT)

    feasible = ro > Rout
    RD = (Rout * ro) / (ro - Rout) if feasible else math.nan

    return dict(G=G, Rout=Rout, ro=ro, WL=WL, RD=RD, feasible=feasible)


def snap_to_grid(WL: float, RD: float) -> dict | None:
    """Find the closest valid (W1, L1, RD) on the discrete grids."""
    best = None
    best_err = math.inf

    RD_snap = RD_GRID[np.argmin(np.abs(RD_GRID - RD))]

    for L1 in L1_GRID:
        W1_ideal = WL * L1
        if W1_ideal < W1_MIN_RATIO * L1 or W1_ideal > W1_GRID[-1]:
            continue
        W1_snap = W1_GRID[np.argmin(np.abs(W1_GRID - W1_ideal))]
        if W1_snap < W1_MIN_RATIO * L1:
            continue
        WL_actual = W1_snap / L1
        err = abs(WL_actual - WL) / WL + abs(RD_snap - RD) / RD
        if err < best_err:
            best_err = err
            best = dict(W1=W1_snap, L1=L1, RD=RD_snap, WL=WL_actual)

    return best


def main():
    parser = argparse.ArgumentParser(description="Analytical CS amp solver")
    parser.add_argument("--gain-db", type=float, default=20.0,
                        help="Target voltage gain in dB (default: 20)")
    parser.add_argument("--bw",      type=float, default=50e6,
                        help="Target bandwidth in Hz (default: 50e6)")
    parser.add_argument("--simulate", action="store_true",
                        help="Run NGSpice on the snapped solution to verify")
    args = parser.parse_args()

    s = solve(args.gain_db, args.bw)

    print(f"Targets:  gain = {args.gain_db} dB  (G = {s['G']:.3f}×)   BW = {args.bw/1e6:.1f} MHz")
    print()
    print(f"  Rout  = {s['Rout']:>10.1f} Ω   (required output impedance)")
    print(f"  ro    = {s['ro']:>10.1f} Ω   (MOSFET output resistance)")
    print()

    if not s["feasible"]:
        print("ERROR: ro < Rout — targets are not simultaneously achievable with this topology.")
        print(f"  ro={s['ro']:.1f} Ω must exceed Rout={s['Rout']:.1f} Ω.")
        print("  Try reducing gain or increasing bandwidth.")
        sys.exit(1)

    print(f"  W/L   = {s['WL']:>10.2f}")
    print(f"  RD    = {s['RD']:>10.1f} Ω")
    print()

    grid = snap_to_grid(s["WL"], s["RD"])
    if grid is None:
        print("WARNING: no valid grid point found (constraint W1 >= 5·L1 may be infeasible).")
    else:
        print("Nearest discrete grid point (W1 >= 5·L1 enforced):")
        print(f"  W1  = {grid['W1']*1e6:.1f} µm")
        print(f"  L1  = {grid['L1']*1e9:.0f} nm   (W/L = {grid['WL']:.1f})")
        print(f"  RD  = {grid['RD']:.0f} Ω")

        if args.simulate:
            print()
            print("Running NGSpice simulation …")
            from circuitrl.simulators.ngspice_runner import NGSpiceRunner
            runner = NGSpiceRunner(
                "circuitrl/envs/cs_amp_template.sp", timeout=15,
                expected_metrics=("gain_db", "bandwidth"),
            )
            result = runner.run({
                "W1": f"{grid['W1']:.6e}",
                "L1": f"{grid['L1']:.6e}",
                "RD": f"{grid['RD']:.6e}",
            })
            if result is None:
                print("Simulation failed.")
            else:
                targets = {"gain_db": args.gain_db, "bandwidth": args.bw}
                print(f"\n  {'Metric':<12} {'Simulated':>14} {'Target':>14} {'%err':>8}")
                print("  " + "-" * 50)
                for name, target in targets.items():
                    val = result[name]
                    pct = 100.0 * abs(val - target) / abs(target)
                    print(f"  {name:<12} {val:>14.4g} {target:>14.4g} {pct:>7.1f}%")


if __name__ == "__main__":
    main()
