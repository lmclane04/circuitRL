"""Validate ngspice wrapper (it should return expected metrics and not timeout)"""

import numpy as np

from circuitrl.simulators.ngspice_runner import NGSpiceRunner

TEMPLATE = "circuitrl/envs/netlist_template.sp"

DEFAULT_PARAMS = {
    "W1": "10u", "L1": "0.5u",
    "W3": "20u", "L3": "0.5u",
    "W5": "10u", "L5": "0.5u",
    "W7": "1u",  "L7": "2u",
    "Cc": "1p",  "Ib": "10u",
}


def test_run_default_params():
    runner = NGSpiceRunner(TEMPLATE)
    result = runner.run(DEFAULT_PARAMS)
    assert result is not None, "Simulation returned None"
    expected_keys = ("gain_db", "ugbw", "phase_margin", "power")
    for key in expected_keys:
        assert key in result, f"Missing metric: {key}"
        assert np.isfinite(result[key]), f"Non-finite value for {key}: {result[key]}"


def test_gain_positive():
    runner = NGSpiceRunner(TEMPLATE)
    result = runner.run(DEFAULT_PARAMS)
    assert result is not None
    assert result["gain_db"] > 0, f"Expected positive gain, got {result['gain_db']} dB"


def test_timeout_handling():
    runner = NGSpiceRunner(TEMPLATE, timeout=0.001)
    result = runner.run(DEFAULT_PARAMS)
    assert result is None, "Expected None for timed-out simulation"
