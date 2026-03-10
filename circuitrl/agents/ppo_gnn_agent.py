from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Global type vocabulary — fixed across ALL circuits so that node_feature_dim
# is constant, making node_encoder fully transferable between topologies.
# Any component type not in this list falls back to "generic".
# ---------------------------------------------------------------------------
GLOBAL_TYPE_VOCAB: list[str] = ["capacitor", "current", "generic", "nmos", "pmos", "resistor"]
GLOBAL_TYPE_TO_ID: dict[str, int] = {t: i for i, t in enumerate(GLOBAL_TYPE_VOCAB)}
GLOBAL_N_TYPES: int = len(GLOBAL_TYPE_VOCAB)          # 6
GLOBAL_MAX_PARAMS_PER_COMPONENT: int = 2              # W+L for MOS; maximum across all supported components


@dataclass
class GraphSpec:
    """Component-level circuit graph.

    Each node is a circuit component (e.g., transistor, resistor).
    Each component can own one or more env parameters (e.g., W/L on one NMOS node).
    """

    component_names: list[str]
    adjacency: torch.Tensor       # degree-normalized (D^{-1}A), used for message passing
    node_degrees: np.ndarray      # raw degree per node before normalization (float32, shape n_nodes)
    node_type_ids: np.ndarray
    model_features: np.ndarray
    component_param_names: list[list[str]]
    param_to_node: np.ndarray
    param_to_slot: np.ndarray
    max_params_per_component: int
    type_vocab: list[str]

    @property
    def nodes(self) -> list[str]:
        # Backward-compat alias for earlier code.
        return self.component_names

    @property
    def n_nodes(self) -> int:
        return len(self.component_names)

    @staticmethod
    def default(param_names: list[str]) -> "GraphSpec":
        # Fallback: one component per parameter.
        n = len(param_names)
        adj = np.eye(n, dtype=np.float32)
        for i in range(n - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        deg = adj.sum(axis=1)  # raw degree before normalization
        adj = adj / np.clip(deg[:, None], 1.0, None)

        return GraphSpec(
            component_names=list(param_names),
            adjacency=torch.tensor(adj, dtype=torch.float32),
            node_degrees=deg.astype(np.float32),
            node_type_ids=np.full(n, GLOBAL_TYPE_TO_ID["generic"], dtype=np.int64),
            model_features=np.zeros((n, 5), dtype=np.float32),
            component_param_names=[[p] for p in param_names],
            param_to_node=np.arange(n, dtype=np.int64),
            param_to_slot=np.zeros(n, dtype=np.int64),
            max_params_per_component=1,
            type_vocab=["generic"],
        )

    @staticmethod
    def from_path(path: str, param_names: list[str]) -> "GraphSpec":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)

        # New schema: components list.
        if "components" in raw:
            components = raw["components"]
            if not components:
                raise ValueError("Graph spec has empty components list")

            component_names = [str(c["name"]) for c in components]
            if len(set(component_names)) != len(component_names):
                raise ValueError("Component names must be unique")

            comp_types = [str(c.get("type", "generic")) for c in components]
            type_vocab = sorted(set(comp_types))  # kept for display/debugging only
            # Use global IDs so node_feature_dim is constant across all circuits.
            # Unknown types fall back to "generic".
            node_type_ids = np.array(
                [GLOBAL_TYPE_TO_ID.get(t, GLOBAL_TYPE_TO_ID["generic"]) for t in comp_types],
                dtype=np.int64,
            )

            model_feature_dim = int(raw.get("model_feature_dim", 5))
            feature_names = raw.get("model_feature_names", None)
            if feature_names is not None and len(feature_names) != model_feature_dim:
                raise ValueError(
                    f"model_feature_names length ({len(feature_names)}) "
                    f"must match model_feature_dim ({model_feature_dim})"
                )
            model_features = np.zeros((len(components), model_feature_dim), dtype=np.float32)

            component_param_names: list[list[str]] = []
            seen_params: set[str] = set()
            param_index = {p: i for i, p in enumerate(param_names)}
            param_to_node = np.zeros(len(param_names), dtype=np.int64)
            param_to_slot = np.zeros(len(param_names), dtype=np.int64)

            max_params_per_component = 0
            for node_idx, comp in enumerate(components):
                comp_params = [str(p) for p in comp.get("parameters", [])]
                if not comp_params:
                    raise ValueError(f"Component {comp['name']} has no parameters")
                component_param_names.append(comp_params)
                max_params_per_component = max(max_params_per_component, len(comp_params))

                for slot_idx, p in enumerate(comp_params):
                    if p not in param_index:
                        raise ValueError(f"Unknown parameter '{p}' in component {comp['name']}")
                    if p in seen_params:
                        raise ValueError(f"Parameter '{p}' appears in multiple components")
                    seen_params.add(p)
                    j = param_index[p]
                    param_to_node[j] = node_idx
                    param_to_slot[j] = slot_idx

                feats = comp.get("model_features", None)
                if feats is not None:
                    feats = np.asarray(feats, dtype=np.float32)
                    if feats.ndim != 1:
                        raise ValueError(f"model_features for {comp['name']} must be a 1D list")
                    if feats.shape[0] != model_feature_dim:
                        raise ValueError(
                            f"model_features for {comp['name']} has length {feats.shape[0]}, "
                            f"expected {model_feature_dim}"
                        )
                    model_features[node_idx, :] = feats

            missing = [p for p in param_names if p not in seen_params]
            if missing:
                raise ValueError(f"Graph spec missing parameters: {missing}")

            if max_params_per_component > GLOBAL_MAX_PARAMS_PER_COMPONENT:
                raise ValueError(
                    f"A component has {max_params_per_component} parameters, "
                    f"but GLOBAL_MAX_PARAMS_PER_COMPONENT={GLOBAL_MAX_PARAMS_PER_COMPONENT}. "
                    f"Increase GLOBAL_MAX_PARAMS_PER_COMPONENT to support this circuit."
                )

            name_to_idx = {n: i for i, n in enumerate(component_names)}
            n = len(component_names)
            adj = np.eye(n, dtype=np.float32)
            for edge in raw.get("edges", []):
                if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                    raise ValueError(f"Invalid edge entry: {edge}")
                a, b = str(edge[0]), str(edge[1])
                if a not in name_to_idx or b not in name_to_idx:
                    raise ValueError(f"Unknown edge endpoints: {edge}")
                ia, ib = name_to_idx[a], name_to_idx[b]
                adj[ia, ib] = 1.0
                if raw.get("undirected", True):
                    adj[ib, ia] = 1.0

            deg = adj.sum(axis=1)  # raw degree before normalization
            adj = adj / np.clip(deg[:, None], 1.0, None)

            return GraphSpec(
                component_names=component_names,
                adjacency=torch.tensor(adj, dtype=torch.float32),
                node_degrees=deg.astype(np.float32),
                node_type_ids=node_type_ids,
                model_features=model_features,
                component_param_names=component_param_names,
                param_to_node=param_to_node,
                param_to_slot=param_to_slot,
                max_params_per_component=max_params_per_component,
                type_vocab=type_vocab,
            )

        # Backward-compat fallback for old parameter-node schema.
        nodes = list(raw.get("nodes", param_names))
        missing = [p for p in param_names if p not in nodes]
        if missing:
            raise ValueError(f"Graph spec missing parameters: {missing}")
        return GraphSpec.default(param_names)


class GraphObservationAdapter:
    """Build per-component node features and a separate goal context vector.

    Node feature layout (per node, constant across all circuits):
      [degree_scalar(1) | type_onehot(GLOBAL_N_TYPES=6) | model_features(5) | param_slots(GLOBAL_MAX_PARAMS=2)]
      → node_feature_dim = 14

    degree_scalar = raw_degree / max_degree_in_graph — encodes structural role (e.g. a bias
    current source with 1 connection scores lower than a diff-pair transistor with 3).
    This replaces the old index_scalar (i / n_nodes-1) which encoded arbitrary YAML ordering
    and did not transfer meaningfully across circuit topologies.

    Returned as a tuple (node_features, spec_context) from transform():
      - node_features: (n_nodes, 14) — topology + current parameter state
      - spec_context: (2 * metric_slots,) — [norm_metrics | norm_targets]

    spec_context is kept separate (not embedded per-node) so it can be injected
    once into both actor and critic heads via spec_encoder, rather than being
    redundantly copied across every node embedding.
    """

    def __init__(
        self,
        n_params: int,
        n_metrics: int,
        graph_spec: GraphSpec,
        metric_slots: int = 4,
    ):
        self.n_params = int(n_params)
        self.n_metrics = int(n_metrics)
        self.metric_slots = int(metric_slots)
        self.graph_spec = graph_spec
        self.n_nodes = graph_spec.n_nodes
        # Use global constants so node_feature_dim is identical for all circuits,
        # enabling cross-topology transfer of the node_encoder weights.
        self.n_types = GLOBAL_N_TYPES
        self.model_feature_dim = graph_spec.model_features.shape[1]
        self.max_params_per_component = GLOBAL_MAX_PARAMS_PER_COMPONENT

        deg = graph_spec.node_degrees
        self.degree_scalar = deg / max(1.0, deg.max())

        self.type_onehot = np.zeros((self.n_nodes, self.n_types), dtype=np.float32)
        self.type_onehot[np.arange(self.n_nodes), graph_spec.node_type_ids] = 1.0

        # node_type_ids now holds global IDs → look up via GLOBAL_TYPE_VOCAB, not local type_vocab
        self.node_types = [GLOBAL_TYPE_VOCAB[i] for i in graph_spec.node_type_ids]
        self.params_per_component = [len(p) for p in graph_spec.component_param_names]

        # metrics and targets are passed as a separate spec_context vector,
        # NOT embedded per-node, so node_feature_dim is independent of metric count.
        self.node_feature_dim = (
            1 + self.n_types + self.model_feature_dim + self.max_params_per_component
        )

    def _pad(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros(self.metric_slots, dtype=np.float32)
        n = min(self.metric_slots, x.shape[0])
        out[:n] = x[:n]
        return out

    def transform(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (node_features, spec_context).

        node_features: (n_nodes, node_feature_dim) — topology + parameter state.
        spec_context: (2 * metric_slots,) — [metrics_pad | targets_pad], goal conditioning
                      kept separate so it can be injected into both actor and critic heads
                      rather than redundantly repeated across every node embedding.
        """
        p = obs[: self.n_params]
        m = obs[self.n_params: self.n_params + self.n_metrics]
        t = obs[self.n_params + self.n_metrics: self.n_params + 2 * self.n_metrics]

        component_param_slots = np.zeros((self.n_nodes, self.max_params_per_component), dtype=np.float32)
        for p_idx in range(self.n_params):
            n_idx = self.graph_spec.param_to_node[p_idx]
            s_idx = self.graph_spec.param_to_slot[p_idx]
            component_param_slots[n_idx, s_idx] = p[p_idx]

        # Use static YAML-defined model features (ro_prior, gm_efficiency, etc.).
        # These preserve pretrained semantic priors across transfer; previously this
        # was overwritten with a dynamically reconstructed array that zeroed slots 2-4.
        model_features = self.graph_spec.model_features  # (n_nodes, model_feature_dim)

        node_features = np.zeros((self.n_nodes, self.node_feature_dim), dtype=np.float32)
        col = 0
        node_features[:, col] = self.degree_scalar
        col += 1
        node_features[:, col: col + self.n_types] = self.type_onehot
        col += self.n_types
        node_features[:, col: col + self.model_feature_dim] = model_features
        col += self.model_feature_dim
        node_features[:, col: col + self.max_params_per_component] = component_param_slots

        spec_context = np.concatenate([self._pad(m), self._pad(t)])  # (2 * metric_slots,)
        return node_features, spec_context


class GraphConvLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neigh_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        neigh = torch.einsum("ij,bjh->bih", adj, h)
        return h + torch.tanh(self.self_linear(h) + self.neigh_linear(neigh))


class GraphActorCritic(nn.Module):
    """GNN actor-critic with separate actor/critic trunks.

    Improvements implemented:
      #4 — Global graph context added to actor param embeddings (residual mean-pool).
      #5 — Residual connections in GraphConvLayer (see GraphConvLayer.forward).
      #6 — Learnable type embedding (shared between trunks) injected after node encoder.
      #8 — Separate actor/critic GNN trunks; critic adapts freely during fine-tuning
           while the actor encoder preserves pretrained representations.
    """

    def __init__(
        self,
        node_feature_dim: int,
        actions_per_param: int,
        max_params_per_component: int,
        metric_slots: int = 4,
        hidden_dim: int = 128,
        gnn_layers: int = 2,
    ):
        super().__init__()
        self.metric_slots = metric_slots

        # --- Actor trunk (improvement #8) ---
        self.actor_node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_gnn_layers = nn.ModuleList([GraphConvLayer(hidden_dim) for _ in range(gnn_layers)])

        # --- Critic trunk (improvement #8: separate from actor) ---
        self.critic_node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.Tanh(),
        )
        self.critic_gnn_layers = nn.ModuleList([GraphConvLayer(hidden_dim) for _ in range(gnn_layers)])

        # --- Shared modules ---
        # Learnable type embedding (improvement #6): supplements static model_features priors.
        # Shared between actor/critic so component-type semantics are learned once.
        self.type_embedding = nn.Embedding(GLOBAL_N_TYPES, hidden_dim)

        # Distinguish W vs L within one component node.
        self.slot_embedding = nn.Embedding(max_params_per_component, hidden_dim)

        # Shared goal encoder: projects [metrics_pad | targets_pad] → hidden_dim.
        # Injected as a residual into both actor param embeddings and critic graph embedding
        # so both get explicit goal conditioning without repeating spec_context per-node.
        self.spec_encoder = nn.Linear(2 * metric_slots, hidden_dim)

        self.policy_head = nn.Linear(hidden_dim, actions_per_param)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_actor(
        self, node_obs: torch.Tensor, adj: torch.Tensor, node_type_ids: torch.Tensor
    ) -> torch.Tensor:
        h = self.actor_node_encoder(node_obs)
        h = h + self.type_embedding(node_type_ids).unsqueeze(0)  # improvement #6
        for layer in self.actor_gnn_layers:
            h = layer(h, adj)
        return h

    def _encode_critic(
        self, node_obs: torch.Tensor, adj: torch.Tensor, node_type_ids: torch.Tensor
    ) -> torch.Tensor:
        h = self.critic_node_encoder(node_obs)
        h = h + self.type_embedding(node_type_ids).unsqueeze(0)  # improvement #6
        for layer in self.critic_gnn_layers:
            h = layer(h, adj)
        return h

    def _param_hidden(
        self,
        h_nodes: torch.Tensor,
        param_to_node: torch.Tensor,
        param_to_slot: torch.Tensor,
        spec_emb: torch.Tensor,
    ) -> torch.Tensor:
        h_params = h_nodes[:, param_to_node, :]
        h_params = h_params + self.slot_embedding(param_to_slot).unsqueeze(0)
        # Improvement #4: global graph context injected as residual.
        h_graph = h_nodes.mean(dim=1, keepdim=True)  # (B, 1, hidden)
        h_params = h_params + h_graph
        # Explicit goal conditioning: same spec embedding broadcast to all params.
        h_params = h_params + spec_emb.unsqueeze(1)  # (B, 1, hidden) → broadcast
        return h_params

    def forward(
        self,
        node_obs: torch.Tensor,
        adj: torch.Tensor,
        param_to_node: torch.Tensor,
        param_to_slot: torch.Tensor,
        node_type_ids: torch.Tensor,
        spec_context: torch.Tensor,
    ):
        spec_emb = self.spec_encoder(spec_context)  # (B, hidden_dim)
        h_actor = self._encode_actor(node_obs, adj, node_type_ids)
        h_critic = self._encode_critic(node_obs, adj, node_type_ids)
        h_params = self._param_hidden(h_actor, param_to_node, param_to_slot, spec_emb)
        logits = self.policy_head(h_params)
        # Critic: graph-level mean-pool + goal conditioning.
        value = self.value_head(h_critic.mean(dim=1) + spec_emb).squeeze(-1)
        return logits, value

    def get_action(
        self,
        node_obs: torch.Tensor,
        adj: torch.Tensor,
        param_to_node: torch.Tensor,
        param_to_slot: torch.Tensor,
        node_type_ids: torch.Tensor,
        spec_context: torch.Tensor,
    ):
        logits, value = self(node_obs, adj, param_to_node, param_to_slot, node_type_ids, spec_context)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_prob_sum = dist.log_prob(actions).sum(dim=1)
        return actions, log_prob_sum, value

    def evaluate(
        self,
        node_obs: torch.Tensor,
        actions: torch.Tensor,
        adj: torch.Tensor,
        param_to_node: torch.Tensor,
        param_to_slot: torch.Tensor,
        node_type_ids: torch.Tensor,
        spec_context: torch.Tensor,
    ):
        logits, value = self(node_obs, adj, param_to_node, param_to_slot, node_type_ids, spec_context)
        dist = Categorical(logits=logits)
        log_prob_sum = dist.log_prob(actions).sum(dim=1)
        entropy = dist.entropy().mean(dim=1)
        return log_prob_sum, entropy, value


class GraphRolloutBuffer:
    def __init__(self, n_steps: int, n_nodes: int, n_params: int, node_feature_dim: int, metric_slots: int = 4):
        self.n_steps = n_steps
        self.node_obs = np.zeros((n_steps, n_nodes, node_feature_dim), dtype=np.float32)
        self.spec_contexts = np.zeros((n_steps, 2 * metric_slots), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_params), dtype=np.int64)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)
        self.ptr = 0

    def store(self, node_obs, spec_context, action, log_prob, reward, done, value):
        self.node_obs[self.ptr] = node_obs
        self.spec_contexts[self.ptr] = spec_context
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        last_adv = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_adv = delta + gamma * gae_lambda * next_non_terminal * last_adv

        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    def get_batches(self, batch_size: int):
        indices = np.arange(self.ptr)
        np.random.shuffle(indices)
        for start in range(0, self.ptr, batch_size):
            batch_idx = indices[start: start + batch_size]
            yield (
                torch.tensor(self.node_obs[batch_idx]),
                torch.tensor(self.spec_contexts[batch_idx]),
                torch.tensor(self.actions[batch_idx]),
                torch.tensor(self.log_probs[batch_idx]),
                torch.tensor(self.returns[batch_idx]),
                torch.tensor(self.advantages[batch_idx]),
            )

    def reset(self):
        self.ptr = 0


class PPOGNNAgent:
    def __init__(
        self,
        env,
        adapter: GraphObservationAdapter,
        graph_spec: GraphSpec,
        config: dict,
        hidden_dim: int = 128,
        gnn_layers: int = 2,
        init_checkpoint: str | None = None,
    ):
        self.env = env
        self.adapter = adapter
        self.graph_spec = graph_spec

        ppo_cfg = config["ppo"]
        self.lr = float(ppo_cfg["learning_rate"])
        self.n_steps = int(ppo_cfg["n_steps"])
        self.batch_size = int(ppo_cfg["batch_size"])
        self.n_epochs = int(ppo_cfg["n_epochs"])
        self.gamma = float(ppo_cfg["gamma"])
        self.total_timesteps = int(ppo_cfg["total_timesteps"])

        self.clip_eps = 0.2
        self.vf_coef = 0.5
        self.ent_coef = 0.01
        self.gae_lambda = 0.95
        self.max_grad_norm = 0.5

        self.n_nodes = graph_spec.n_nodes
        self.n_params = env._n_params
        self.node_feature_dim = adapter.node_feature_dim
        self.actions_per_param = env.n_actions_per_param
        self.hidden_dim = int(hidden_dim)
        self.gnn_layers = int(gnn_layers)

        self.metric_slots = adapter.metric_slots
        self.network = GraphActorCritic(
            node_feature_dim=self.node_feature_dim,
            actions_per_param=self.actions_per_param,
            max_params_per_component=GLOBAL_MAX_PARAMS_PER_COMPONENT,
            metric_slots=self.metric_slots,
            hidden_dim=self.hidden_dim,
            gnn_layers=self.gnn_layers,
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.buffer = GraphRolloutBuffer(
            self.n_steps, self.n_nodes, self.n_params, self.node_feature_dim, self.metric_slots
        )

        self.adj = graph_spec.adjacency
        self.param_to_node = torch.tensor(graph_spec.param_to_node, dtype=torch.long)
        self.param_to_slot = torch.tensor(graph_spec.param_to_slot, dtype=torch.long)
        self.node_type_ids = torch.tensor(graph_spec.node_type_ids, dtype=torch.long)

        if init_checkpoint:
            self._load_network_weights(init_checkpoint)

    def _load_network_weights(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        if "network" not in checkpoint:
            raise ValueError(f"Checkpoint {checkpoint_path} missing 'network' key")
        src = checkpoint["network"]
        dst = self.network.state_dict()

        # Remap old single-trunk checkpoint keys to the new separate-trunk naming.
        # Old: node_encoder.* / gnn_layers.* → New: actor_node_encoder.* / actor_gnn_layers.*
        # The critic trunk starts from random init at fine-tune time (correct behavior).
        remapped = {}
        for k, v in src.items():
            if k.startswith("node_encoder."):
                remapped["actor_node_encoder." + k[len("node_encoder."):]] = v
            elif k.startswith("gnn_layers."):
                remapped["actor_gnn_layers." + k[len("gnn_layers."):]] = v
            else:
                remapped[k] = v

        compatible = {}
        skipped_shape = []
        for k, v in remapped.items():
            if k not in dst:
                continue
            if dst[k].shape != v.shape:
                skipped_shape.append((k, tuple(v.shape), tuple(dst[k].shape)))
                continue
            compatible[k] = v

        missing, _ = self.network.load_state_dict(compatible, strict=False)
        print(f"[transfer] Loaded {len(compatible)} compatible tensors from {checkpoint_path}")
        if skipped_shape:
            print("[transfer] Skipped mismatched tensors:")
            for name, s_src, s_dst in skipped_shape:
                print(f"  - {name}: checkpoint{s_src} != current{s_dst}")
        if missing:
            print(f"[transfer] Missing keys (critic trunk starts fresh): {missing}")

    def set_encoder_frozen(self, frozen: bool):
        """Freeze/unfreeze actor encoder only; critic adapts freely during fine-tuning."""
        for p in self.network.actor_node_encoder.parameters():
            p.requires_grad = not frozen
        for layer in self.network.actor_gnn_layers:
            for p in layer.parameters():
                p.requires_grad = not frozen

    def collect_rollouts(self) -> list[dict]:
        self.buffer.reset()
        obs, _ = self.env.reset()
        node_obs, spec_ctx = self.adapter.transform(obs)

        episode_stats = []
        ep_reward = 0.0
        ep_len = 0

        for _ in range(self.n_steps):
            with torch.no_grad():
                node_obs_t = torch.tensor(node_obs, dtype=torch.float32).unsqueeze(0)
                spec_ctx_t = torch.tensor(spec_ctx, dtype=torch.float32).unsqueeze(0)
                action, log_prob, value = self.network.get_action(
                    node_obs_t,
                    self.adj,
                    self.param_to_node,
                    self.param_to_slot,
                    self.node_type_ids,
                    spec_ctx_t,
                )

            action_np = action.squeeze(0).numpy()
            next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated

            self.buffer.store(node_obs, spec_ctx, action_np, log_prob.item(), reward, float(done), value.item())

            ep_reward += reward
            ep_len += 1

            if done:
                episode_stats.append({"reward": ep_reward, "length": ep_len})
                ep_reward = 0.0
                ep_len = 0
                obs, _ = self.env.reset()
            else:
                obs = next_obs
            node_obs, spec_ctx = self.adapter.transform(obs)

        with torch.no_grad():
            node_obs_t = torch.tensor(node_obs, dtype=torch.float32).unsqueeze(0)
            spec_ctx_t = torch.tensor(spec_ctx, dtype=torch.float32).unsqueeze(0)
            _, _, last_value = self.network.get_action(
                node_obs_t,
                self.adj,
                self.param_to_node,
                self.param_to_slot,
                self.node_type_ids,
                spec_ctx_t,
            )

        self.buffer.compute_gae(last_value.item(), self.gamma, self.gae_lambda)
        return episode_stats

    def update(self) -> dict:
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for node_obs_b, spec_ctx_b, act_b, old_lp_b, ret_b, adv_b in self.buffer.get_batches(self.batch_size):
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                new_lp, entropy, values = self.network.evaluate(
                    node_obs_b,
                    act_b,
                    self.adj,
                    self.param_to_node,
                    self.param_to_slot,
                    self.node_type_ids,
                    spec_ctx_b,
                )

                ratio = torch.exp(new_lp - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, ret_b)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
        }

    def train(
        self,
        total_timesteps: int | None = None,
        callback: Callable | None = None,
        freeze_encoder_steps: int = 0,
    ):
        total = total_timesteps or self.total_timesteps
        timesteps_done = 0
        encoder_frozen = freeze_encoder_steps > 0
        if encoder_frozen:
            self.set_encoder_frozen(True)

        while timesteps_done < total:
            if encoder_frozen and timesteps_done >= freeze_encoder_steps:
                self.set_encoder_frozen(False)
                encoder_frozen = False
                print(f"[transfer] Unfroze GNN encoder at timestep {timesteps_done}")

            episode_stats = self.collect_rollouts()
            loss_stats = self.update()
            timesteps_done += self.n_steps

            if callback:
                callback(timesteps_done, episode_stats, loss_stats)

    def save(self, path: str):
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "meta": {
                    "node_feature_dim": self.node_feature_dim,
                    "hidden_dim": self.hidden_dim,
                    "gnn_layers": self.gnn_layers,
                    "actions_per_param": self.actions_per_param,
                    "metric_slots": self.adapter.metric_slots,
                    "n_nodes": self.n_nodes,
                    "n_params": self.n_params,
                    "max_params_per_component": GLOBAL_MAX_PARAMS_PER_COMPONENT,
                },
            },
            path,
        )


# ---------------------------------------------------------------------------
# Multi-circuit training
# ---------------------------------------------------------------------------

class MultiCircuitPPOGNNAgent:
    """PPO agent with a single GNN shared across multiple circuit topologies.

    At each training iteration, collects n_steps//n_circuits steps from each
    circuit in sequence, then does a joint PPO update using interleaved batches
    from all circuits. The shared network learns representations that transfer
    across topologies.

    Requires Fix #1 (GLOBAL_TYPE_VOCAB) so that node_feature_dim is identical
    for all circuits — otherwise the shared node_encoder cannot be used.

    circuits: list of dicts, each with keys:
        "id"         — short circuit name, e.g. "cs_amp"
        "env"        — CircuitEnv instance
        "adapter"    — GraphObservationAdapter instance
        "graph_spec" — GraphSpec instance
    """

    def __init__(
        self,
        circuits: list[dict],
        config: dict,
        hidden_dim: int = 128,
        gnn_layers: int = 2,
        init_checkpoint: str | None = None,
    ):
        if not circuits:
            raise ValueError("Must provide at least one circuit")
        self.circuits = circuits
        n_circuits = len(circuits)

        # Validate shared dimensions
        actions_per_param = circuits[0]["env"].n_actions_per_param
        node_feature_dim = circuits[0]["adapter"].node_feature_dim
        for c in circuits[1:]:
            if c["env"].n_actions_per_param != actions_per_param:
                raise ValueError(
                    f"All circuits must use the same n_actions_per_param. "
                    f"Circuit '{c['id']}' has {c['env'].n_actions_per_param}, "
                    f"expected {actions_per_param}. Standardize action_deltas in YAML configs."
                )
            if c["adapter"].node_feature_dim != node_feature_dim:
                raise ValueError(
                    f"All circuits must have the same node_feature_dim. "
                    f"Circuit '{c['id']}' has {c['adapter'].node_feature_dim}, "
                    f"expected {node_feature_dim}. "
                    f"Ensure GLOBAL_TYPE_VOCAB is applied (Fix #1)."
                )

        ppo_cfg = config["ppo"]
        self.lr = float(ppo_cfg["learning_rate"])
        self.n_steps = int(ppo_cfg["n_steps"])
        self.batch_size = int(ppo_cfg["batch_size"])
        self.n_epochs = int(ppo_cfg["n_epochs"])
        self.gamma = float(ppo_cfg["gamma"])
        self.total_timesteps = int(ppo_cfg["total_timesteps"])

        self.clip_eps = 0.2
        self.vf_coef = 0.5
        self.ent_coef = 0.01
        self.gae_lambda = 0.95
        self.max_grad_norm = 0.5

        self.actions_per_param = actions_per_param
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.gnn_layers = gnn_layers
        self.metric_slots = circuits[0]["adapter"].metric_slots

        self.network = GraphActorCritic(
            node_feature_dim=node_feature_dim,
            actions_per_param=actions_per_param,
            max_params_per_component=GLOBAL_MAX_PARAMS_PER_COMPONENT,
            metric_slots=self.metric_slots,
            hidden_dim=hidden_dim,
            gnn_layers=gnn_layers,
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        # Per-circuit rollout buffers (n_nodes and n_params differ per circuit).
        steps_per_circuit = max(1, self.n_steps // n_circuits)
        self.steps_per_circuit = steps_per_circuit

        self.buffers: dict[str, GraphRolloutBuffer] = {}
        self.adjs: dict[str, torch.Tensor] = {}
        self.param_to_nodes: dict[str, torch.Tensor] = {}
        self.param_to_slots: dict[str, torch.Tensor] = {}
        self.node_type_ids: dict[str, torch.Tensor] = {}

        for c in circuits:
            cid = c["id"]
            gs = c["graph_spec"]
            self.buffers[cid] = GraphRolloutBuffer(
                steps_per_circuit, gs.n_nodes, c["env"]._n_params, node_feature_dim, self.metric_slots
            )
            self.adjs[cid] = gs.adjacency
            self.param_to_nodes[cid] = torch.tensor(gs.param_to_node, dtype=torch.long)
            self.param_to_slots[cid] = torch.tensor(gs.param_to_slot, dtype=torch.long)
            self.node_type_ids[cid] = torch.tensor(gs.node_type_ids, dtype=torch.long)

        if init_checkpoint:
            self._load_network_weights(init_checkpoint)

    def _load_network_weights(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        if "network" not in checkpoint:
            raise ValueError(f"Checkpoint {checkpoint_path} missing 'network' key")
        src = checkpoint["network"]
        dst = self.network.state_dict()

        # Remap old single-trunk checkpoint keys to the new separate-trunk naming.
        remapped = {}
        for k, v in src.items():
            if k.startswith("node_encoder."):
                remapped["actor_node_encoder." + k[len("node_encoder."):]] = v
            elif k.startswith("gnn_layers."):
                remapped["actor_gnn_layers." + k[len("gnn_layers."):]] = v
            else:
                remapped[k] = v

        compatible = {}
        skipped_shape = []
        for k, v in remapped.items():
            if k not in dst:
                continue
            if dst[k].shape != v.shape:
                skipped_shape.append((k, tuple(v.shape), tuple(dst[k].shape)))
                continue
            compatible[k] = v
        missing, _ = self.network.load_state_dict(compatible, strict=False)
        print(f"[transfer] Loaded {len(compatible)} compatible tensors from {checkpoint_path}")
        if skipped_shape:
            print("[transfer] Skipped mismatched tensors:")
            for name, s_src, s_dst in skipped_shape:
                print(f"  - {name}: checkpoint{s_src} != current{s_dst}")
        if missing:
            print(f"[transfer] Missing keys (critic trunk starts fresh): {missing}")

    def set_encoder_frozen(self, frozen: bool):
        """Freeze/unfreeze actor encoder only; critic adapts freely during fine-tuning."""
        for p in self.network.actor_node_encoder.parameters():
            p.requires_grad = not frozen
        for layer in self.network.actor_gnn_layers:
            for p in layer.parameters():
                p.requires_grad = not frozen

    def collect_rollouts(self) -> list[dict]:
        """Collect steps_per_circuit steps from each circuit using the shared network."""
        all_stats = []
        for circuit in self.circuits:
            cid = circuit["id"]
            env = circuit["env"]
            adapter = circuit["adapter"]
            buf = self.buffers[cid]
            adj = self.adjs[cid]
            p2n = self.param_to_nodes[cid]
            p2s = self.param_to_slots[cid]
            nti = self.node_type_ids[cid]
            buf.reset()

            obs, _ = env.reset()
            node_obs, spec_ctx = adapter.transform(obs)
            ep_reward = 0.0
            ep_len = 0

            for _ in range(buf.n_steps):
                with torch.no_grad():
                    node_obs_t = torch.tensor(node_obs, dtype=torch.float32).unsqueeze(0)
                    spec_ctx_t = torch.tensor(spec_ctx, dtype=torch.float32).unsqueeze(0)
                    action, log_prob, value = self.network.get_action(node_obs_t, adj, p2n, p2s, nti, spec_ctx_t)

                action_np = action.squeeze(0).numpy()
                next_obs, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated

                buf.store(node_obs, spec_ctx, action_np, log_prob.item(), reward, float(done), value.item())
                ep_reward += reward
                ep_len += 1

                if done:
                    all_stats.append({"reward": ep_reward, "length": ep_len, "circuit": cid})
                    ep_reward = 0.0
                    ep_len = 0
                    obs, _ = env.reset()
                else:
                    obs = next_obs
                node_obs, spec_ctx = adapter.transform(obs)

            with torch.no_grad():
                node_obs_t = torch.tensor(node_obs, dtype=torch.float32).unsqueeze(0)
                spec_ctx_t = torch.tensor(spec_ctx, dtype=torch.float32).unsqueeze(0)
                _, _, last_value = self.network.get_action(node_obs_t, adj, p2n, p2s, nti, spec_ctx_t)
            buf.compute_gae(last_value.item(), self.gamma, self.gae_lambda)

        return all_stats

    def update(self) -> dict:
        """Joint PPO update: round-robin batches from all circuit buffers."""
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            # Build per-circuit batch iterators
            batch_iters = {
                c["id"]: iter(self.buffers[c["id"]].get_batches(self.batch_size))
                for c in self.circuits
            }
            # Round-robin: one update per circuit per pass until all exhausted
            active_ids = [c["id"] for c in self.circuits]
            while active_ids:
                next_active = []
                for cid in active_ids:
                    try:
                        node_obs_b, spec_ctx_b, act_b, old_lp_b, ret_b, adv_b = next(batch_iters[cid])
                    except StopIteration:
                        continue
                    next_active.append(cid)

                    adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
                    new_lp, entropy, values = self.network.evaluate(
                        node_obs_b, act_b,
                        self.adjs[cid], self.param_to_nodes[cid], self.param_to_slots[cid],
                        self.node_type_ids[cid], spec_ctx_b,
                    )
                    ratio = torch.exp(new_lp - old_lp_b)
                    surr1 = ratio * adv_b
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.functional.mse_loss(values, ret_b)
                    entropy_loss = -entropy.mean()
                    loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    n_updates += 1
                active_ids = next_active

        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
        }

    def train(
        self,
        total_timesteps: int | None = None,
        callback: Callable | None = None,
        freeze_encoder_steps: int = 0,
    ):
        total = total_timesteps or self.total_timesteps
        timesteps_done = 0
        steps_per_iter = self.steps_per_circuit * len(self.circuits)

        encoder_frozen = freeze_encoder_steps > 0
        if encoder_frozen:
            self.set_encoder_frozen(True)

        while timesteps_done < total:
            if encoder_frozen and timesteps_done >= freeze_encoder_steps:
                self.set_encoder_frozen(False)
                encoder_frozen = False
                print(f"[transfer] Unfroze GNN encoder at timestep {timesteps_done}")

            episode_stats = self.collect_rollouts()
            loss_stats = self.update()
            timesteps_done += steps_per_iter

            if callback:
                callback(timesteps_done, episode_stats, loss_stats)

    def save(self, path: str):
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "meta": {
                    "node_feature_dim": self.node_feature_dim,
                    "hidden_dim": self.hidden_dim,
                    "gnn_layers": self.gnn_layers,
                    "actions_per_param": self.actions_per_param,
                    "metric_slots": self.metric_slots,
                    "n_circuits": len(self.circuits),
                    "circuit_ids": [c["id"] for c in self.circuits],
                    "max_params_per_component": GLOBAL_MAX_PARAMS_PER_COMPONENT,
                },
            },
            path,
        )
