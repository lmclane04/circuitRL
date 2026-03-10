# Graph Spec Feature Schema

`model_features` is a fixed-length field in the graph spec. In the current
implementation, runtime node features are computed directly from current
component parameters (the same normalized values controlled by the policy and
fed into NGSpice).

Current schema (`model_feature_dim: 5`):

1. `active_device`: 1 for active bias/conducting devices, 0 for passive elements
2. `gm_efficiency_prior`: heuristic transconductance-efficiency prior (higher = stronger gm/Id expectation)
3. `ro_prior`: heuristic output-resistance prior (higher = stronger intrinsic gain / current-source behavior)
4. `parasitic_load_prior`: heuristic parasitic capacitance/load prior (higher = expected loading/parasitics)
5. `mismatch_sensitivity`: heuristic sensitivity to mismatch/process variation

Runtime behavior:
1. MOS nodes: `[W_norm, L_norm, 0, 0, 0]`
2. Resistor/capacitor/current/other: `[param_norm, 0, 0, 0, 0]`

Notes:
- Values are normalized to `[0, 1]` and come from current tunable parameters.
- Keep semantics identical across all circuits so transfer learning can reuse this channel.
- If you later have extracted device data, replace these with normalized technology-specific numbers.
