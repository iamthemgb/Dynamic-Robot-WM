"""Run every unit test module: python -m generalized_physics.tests.run_all"""

import importlib
import sys
import time

MODULES = [
    "test_record_permutation",
    "test_record_padding_mask",
    "test_temporal_order",
    "test_action_alignment",
    "test_projector_contract",
    "test_zero_gate_equivalence",
    "test_shared_noise_ranking",
    "test_lora_zero_init",
]


def main():
    failures = []
    for name in MODULES:
        mod = importlib.import_module(f"{__package__}.{name}")
        fns = [getattr(mod, f) for f in dir(mod) if f.startswith("test_")]
        for fn in fns:
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name}.{fn.__name__} ({time.time() - t0:.1f}s)")
            except Exception as e:  # noqa: BLE001
                failures.append((name, fn.__name__, e))
                print(f"FAIL {name}.{fn.__name__}: {e}")
    print(f"\n{len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
