"""T5-encode captions_blind.jsonl into per-episode fp16 tensors.

All 3,000 projectile captions are already cached (verified byte-identical by
captions.py); this is the idempotent refill. umt5-xxl bf16 needs a large GPU
(a100/h100/l40s).

  python projectile_smoke/data/precompute_t5.py [--overwrite]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("WAN21_ROOT", "/gpfs/radev/scratch/sous/mzl7/Wan2.1"))
sys.path.insert(0, str(REPO))

from wan.modules.t5 import T5EncoderModel  # noqa: E402

from projectile_smoke.data.episodes_proj import (  # noqa: E402
    CACHE, WAN_MODEL_DIR, file_stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(CACHE))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cache = Path(args.out_dir)
    out = cache / "t5_blind"
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(cache / "captions_blind.jsonl")]
    todo = [r for r in rows
            if args.overwrite or not (out / f"{file_stem(r['key'])}.pt").exists()]
    print(f"{len(todo)}/{len(rows)} captions to encode")
    if not todo:
        return

    encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        checkpoint_path=str(WAN_MODEL_DIR / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(WAN_MODEL_DIR / "google/umt5-xxl"),
    )

    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        with torch.no_grad():
            embs = encoder([r["caption"] for r in batch], torch.device("cuda"))
        for r, e in zip(batch, embs):
            dst = out / f"{file_stem(r['key'])}.pt"
            tmp = dst.with_suffix(".tmp")
            torch.save({"key": r["key"], "caption": r["caption"],
                        "t5": e.to(torch.float16).cpu()}, tmp)
            tmp.rename(dst)
        if (i // args.batch_size) % 20 == 0:
            print(f"{i + len(batch)}/{len(todo)}", flush=True)
    print("finished")


if __name__ == "__main__":
    main()
