"""Precompute umT5-xxl embeddings for the synthetic prompt set.

There are only ~32-40 distinct prompts (one per appearance bucket), so the
whole table is a few MB. Computing it once here means the 11.4 GB text encoder
is never resident during training.

  python -m generalized_physics.real.t5_cache --cache <cache_dir>
"""

import argparse
import json
from pathlib import Path

import torch

from .paths import OUT_ROOT
from .prompts import prompt_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(OUT_ROOT / "cache" / "wan21_vae"))
    ap.add_argument("--also", nargs="*", default=[],
                    help="extra cache dirs to copy the table into")
    args = ap.parse_args()

    import pandas as pd
    from . import wan_loader as W

    cache = Path(args.cache)
    idx = pd.read_parquet(cache / "index.parquet")
    table = prompt_table(idx)
    print(f"{len(table)} distinct prompts")
    for pid, text in list(table.items())[:3]:
        print(f"  [{pid}] {text}")

    t5 = W.load_t5(device="cuda")
    ids = sorted(table)
    with torch.no_grad():
        embs = t5([table[i] for i in ids], torch.device("cuda"))
    out = {str(i): e.detach().to(torch.bfloat16).cpu()
           for i, e in zip(ids, embs)}
    # prompt_id -1 is the unconditional slot, for text ablations
    with torch.no_grad():
        out["-1"] = t5([""], torch.device("cuda"))[0].to(torch.bfloat16).cpu()

    for d in [cache] + [Path(p) for p in args.also]:
        (d / "t5").mkdir(parents=True, exist_ok=True)
        torch.save(out, d / "t5" / "embeddings.pt")
        (d / "t5" / "prompts.json").write_text(json.dumps(table, indent=2))
        print(f"wrote {d/'t5'}  shapes "
              f"{ {k: tuple(v.shape) for k, v in list(out.items())[:2]} }")


if __name__ == "__main__":
    main()
