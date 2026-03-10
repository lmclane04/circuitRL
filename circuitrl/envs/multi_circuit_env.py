import numpy as np

from circuitrl.envs.circuit_env import CircuitEnv


class MultiCircuitEnv:
    """Wraps multiple CircuitEnvs, randomly selecting one at each reset().

    Not a proper gym.Env because obs/action spaces vary per circuit.
    Used by MultiCircuitPPOGNNAgent, which manages per-circuit buffers
    and graph data internally.

    Each reset() picks a circuit uniformly at random and returns
    circuit_id in the info dict so the agent can select the correct
    adjacency matrix and graph observation adapter.
    """

    def __init__(self, circuit_configs: list[tuple[str, str]]):
        """
        Args:
            circuit_configs: list of (circuit_id, config_path) pairs.
                circuit_id is a short name used to look up graph data
                in the agent (e.g. "cs_amp", "opamp").
        """
        if not circuit_configs:
            raise ValueError("Must provide at least one circuit config")
        self.circuit_ids: list[str] = [cid for cid, _ in circuit_configs]
        self.envs: dict[str, CircuitEnv] = {
            cid: CircuitEnv(config_path=cfg) for cid, cfg in circuit_configs
        }
        self.current_id: str | None = None
        self._rng = np.random.default_rng()

    def reset(self, seed: int | None = None, **kwargs):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        idx = int(self._rng.integers(len(self.circuit_ids)))
        self.current_id = self.circuit_ids[idx]
        obs, info = self.envs[self.current_id].reset()
        info["circuit_id"] = self.current_id
        return obs, info

    def step(self, action):
        if self.current_id is None:
            raise RuntimeError("Call reset() before step()")
        obs, reward, terminated, truncated, info = self.envs[self.current_id].step(action)
        info["circuit_id"] = self.current_id
        return obs, reward, terminated, truncated, info

    def get_env(self, circuit_id: str) -> CircuitEnv:
        return self.envs[circuit_id]

    @property
    def current_env(self) -> CircuitEnv | None:
        return self.envs.get(self.current_id) if self.current_id else None
