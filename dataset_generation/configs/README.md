# Configuration index

- `corpus/`: authoritative 20-leaf taxonomy and release state.
- `backends/`: backend capabilities and pinned source dependencies.
- `randomization/`: R0/R1/R2 policy and independent RNG streams.
- `review/`: fixed-six acceptance policy.
- `physics/`: versioned physics profiles and exploratory calibration evidence.
- `assets/`: admitted RoboCasa catalog and machine-local root template.
- `cameras/`, `schema/`, and `splits/`: canonical data contract settings.
- `pilots/` and `release_gates/`: staged scale gates; blocked configurations
  must remain fail-closed.
- `examples/`: diagnostic/non-release example inputs.

Normal operators should not edit resolved run files by hand. Plan immutable
runs through the CLI so hashes bind the effective configuration.
