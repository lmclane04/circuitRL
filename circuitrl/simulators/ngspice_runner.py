"""NGSpice subprocess wrapper for circuit simulation."""

import os
import re
import subprocess
import tempfile


class NGSpiceRunner:
    """Fills a parameterized SPICE netlist template, runs NGSpice in batch mode,
    and parses measurement results from stdout."""

    def __init__(self, template_path: str, timeout: int = 30, expected_metrics: tuple | None = None):
        with open(template_path) as f:
            self._template = f.read()
        self._timeout = timeout
        self._expected_metrics = expected_metrics

    def run(self, params: dict) -> dict | None:
        """Run a simulation with the given parameters.

        Args:
            params: dict mapping parameter names (W1, L1, …) to string values
                    suitable for SPICE (e.g. '10u', '0.5u', '1p').

        Returns:
            Dict of metric name → float, or None if the simulation failed.
        """
        netlist = self._template.format(**params)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sp", delete=False
        ) as tmp:
            tmp.write(netlist)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                ["ngspice", "-b", tmp_path],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            return self._parse_output(result.stdout)
        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None
        finally:
            os.unlink(tmp_path)

    def _parse_output(self, stdout: str) -> dict | None:
        """Parse NGSpice stdout for MEAS_ prefixed echo lines.

        The netlist prints lines like:
            MEAS_gain_db = 6.02000e+01
        """
        pattern = re.compile(
            r"^MEAS_(\w+)\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
            re.MULTILINE,
        )
        parsed = {}
        for match in pattern.finditer(stdout):
            name = match.group(1).lower()
            value = float(match.group(2))
            parsed[name] = value

        if not parsed:
            return None

        # If expected metrics specified, validate and return only those
        if self._expected_metrics:
            if not all(k in parsed for k in self._expected_metrics):
                return None
            return {k: parsed[k] for k in self._expected_metrics}

        return parsed
