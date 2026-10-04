"""Fine-tune a Laya checkpoint on a single device (CPU or GPU).

Runs the same loop the Kaggle notebook runs, through the shipped `laya.train` module instead of
carrying a private copy of it: preprocessing, training, calibration and export are all
`laya.train.finetune`, so a fix to the loop lands here too.

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
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
from huggingface_hub import snapshot_download

import laya.train
from laya.train import TrainConfig


def default_model_dir():
    """The multilingual checkpoint, which is the point of this script: one device, any model."""
    return os.path.join(
        snapshot_download(
            "convaiinnovations/laya",
            allow_patterns=["multilingual/*", "multilingual/*/*"],
        ),
        "multilingual",
    )


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
    model_dir = args.model_dir or default_model_dir()
    print("Device:", args.device)

    summary = laya.train.finetune(
        data=args.data,
        model_dir=model_dir,
        output_dir=args.output_dir,
        # The budgets this script has always trained with, rather than the checkpoint's own.
        config=TrainConfig(epochs=args.epochs, seed=args.seed, micro_batch=8, grad_accum=1,
                           max_len=1024, head_max_len=256, calib_max=400, calib_frac=0.1),
        device=args.device,
    )
    print("Train items: {} ({} held out for calibration) | skipped: {}"
          .format(summary["train_items"], summary["calibration_items"], summary["skipped"]))
    print("Fitted temperatures (choice, score, noul):",
          [round(t, 3) for t in summary["temperature"]])
    if summary["temperature_by_options"]:
        print("Fitted per-bucket temperatures:",
              {k: round(v, 3) for k, v in summary["temperature_by_options"].items()})
    print("Saved checkpoint to", summary["output_dir"])


if __name__ == "__main__":
    main()
