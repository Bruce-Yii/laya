"""Fine-tune a Laya checkpoint on a single device (CPU or GPU).

Delegates preprocessing, training, calibration and export to the shipped `laya.train.finetune`
rather than carrying a private copy of the loop, so a fix to the loop reaches this entry point.
The Kaggle notebook is not this script and keeps its own DDP orchestration; what it shares with
this one is the building blocks, not the loop.

Preprocesses a JSONL dataset into training items, trains with RLCD, fits temperatures through
`laya.calibrate.fit_temperature_map` and saves a checkpoint that `laya.load` opens.

Each JSONL line is one case: {"state": "...", "questions": {...}, "gold": {...}}
where questions use the choice / score / noul primitives and gold carries the
teacher probabilities plus a label, matching the schema in docs/finetune.md.

Usage:
  python research/scripts/finetune_single_device.py \
      --data dataset.jsonl --model-dir /path/to/multilingual --output-dir out/
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
from huggingface_hub import snapshot_download

import laya.train
from laya.train import TrainConfig

# What this entry point has always persisted. `laya.train` writes the fitted temperatures and
# the length budgets; the rest of this dict is the rest of the old output contract, restored here
# so delegating the loop does not silently change the checkpoint it produces.
PERSISTED = {"max_tokens_per_batch": 4096, "max_len": 1024, "head_max_len": 256}


def default_model_dir():
    """The multilingual checkpoint, which is the point of this script: one device, any model."""
    return os.path.join(
        snapshot_download(
            "convaiinnovations/laya",
            allow_patterns=["multilingual/*", "multilingual/*/*"],
        ),
        "multilingual",
    )


def resolve_device(requested):
    """`auto` has always meant CUDA if it is there, otherwise CPU -- not MPS or XPU."""
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def finalize_config(output_dir, gradient_checkpointing):
    """Write the output config this entry point has always produced.

    Per-type temperatures only: the bucket map `fit_temperature_map` also fits is dropped rather
    than persisted, as before. `laya.load` reads the per-type scalars.
    """
    path = os.path.join(output_dir, "rl_agent_config.json")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("temperature_by_options", None)
    cfg["gradient_checkpointing"] = gradient_checkpointing
    cfg.update(PERSISTED)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a Laya checkpoint on one device."
    )
    parser.add_argument(
        "--data", required=True, help="JSONL dataset of {state, questions, gold} cases."
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Checkpoint directory; defaults to the multilingual subfolder.",
    )
    parser.add_argument("--output-dir", default="laya_finetuned")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))
    device = resolve_device(args.device)
    use_amp = torch.device(device).type == "cuda"
    model_dir = args.model_dir or default_model_dir()
    print("Device:", device)

    summary = laya.train.finetune(
        data=args.data,
        model_dir=model_dir,
        output_dir=args.output_dir,
        # The budgets this script has always trained with, rather than the checkpoint's own, and
        # `--seed` deciding the calibration split as well as the epoch order, as it always has.
        config=TrainConfig(epochs=args.epochs, seed=args.seed, calib_seed=args.seed,
                           micro_batch=8, grad_accum=1, calib_max=400, calib_frac=0.1,
                           max_len=PERSISTED["max_len"], head_max_len=PERSISTED["head_max_len"]),
        device=device,
    )
    cfg = finalize_config(args.output_dir, use_amp)
    print("Train items: {} ({} held out for calibration) | skipped: {}"
          .format(summary["train_items"], summary["calibration_items"], summary["skipped"]))
    print("Fitted temperatures (choice, score, noul):",
          [round(t, 3) for t in cfg["temperature"]])
    if summary["temperature_by_options"]:
        print("Per-bucket temperatures were fitted but are not persisted here, as before.")
    print("Saved checkpoint to", args.output_dir)


if __name__ == "__main__":
    main()
