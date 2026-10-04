"""Laya typed-decisions fine-tuning on Apple Silicon.

This is a standalone replacement for the Kaggle 2xT4 notebook. It performs:
1. dependency/model/data preparation
2. local preprocessing and caching
3. single-process MPS/CPU fine-tuning
4. temperature calibration
5. checkpoint/model export

Steps 2 to 5 are `laya.train`, the loop `research/scripts/finetune_single_device.py` also uses,
so this script carries no copy of it.

Useful examples:
    python notebooks/laya_finetune_typed_decisions_mps.py --epochs 2 --micro-batch 1 --grad-accum 32
    python notebooks/laya_finetune_typed_decisions_mps.py --model-dir ./laya_base --items ./train_items.jsonl
    python notebooks/laya_finetune_typed_decisions_mps.py --device cpu
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

import laya.train
from laya.agent import _fix_tokenizer_config
from laya.train import TrainConfig

MODEL_ID = "convaiinnovations/laya"
DATASET_ID = "LocalLLaMA/typed-decisions"
DEFAULT_MODEL_DIR = "./laya_base"
DEFAULT_ITEMS = "./train_items.jsonl"
DEFAULT_OUTPUT_DIR = "./laya_finetuned_typed_decisions"

# What this entry point has always persisted. `laya.train` writes the fitted temperatures and the
# length budgets; the rest is the rest of the old output contract, restored here so delegating the
# loop does not silently change the checkpoint it produces.
PERSISTED = {"max_tokens_per_batch": 2048, "max_len": 1024, "head_max_len": 256,
             "model_name": "laya-typed-decisions"}

# This script always held the calibration split still: the old code shuffled the case order with
# `random.Random(20260922)` regardless of `--seed`, and `--seed` only ordered the epochs. Passed
# explicitly rather than inherited, so the split cannot drift with the loop's default.
CALIB_SPLIT_SEED = 20260922


def choose_device(requested):
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS is unavailable; using CPU instead.")
        return torch.device("cpu")
    return torch.device(requested)


def prepare_model(model_dir):
    model_dir = Path(model_dir)
    if not (model_dir / "model.safetensors").exists():
        print(f"Downloading {MODEL_ID} to {model_dir} ...")
        snapshot_download(MODEL_ID, local_dir=str(model_dir))
    _fix_tokenizer_config(str(model_dir))
    return str(model_dir)


def prepare_rows(items_path, force=False):
    """Cache the dataset as the JSONL of `{state, questions, gold}` cases `laya.train` reads.

    `laya.train` builds the tokenized items from rows, so what is cached here is the dataset
    rather than a second copy of the preprocessing. The sidecar key is unchanged in meaning: a
    cache built by an older version is rebuilt once, because its rows are preprocessed items
    rather than cases.
    """
    items_path = Path(items_path)
    cache_meta_path = items_path.with_name(items_path.name + ".meta.json")
    cache_key = {"dataset": DATASET_ID, "split": "train", "rows": True}
    if items_path.exists() and not force:
        if not cache_meta_path.exists():
            print(f"Rebuilding {items_path}: legacy cached training items cannot be reused.")
        else:
            try:
                with open(cache_meta_path) as f:
                    cached_key = json.load(f)
            except (OSError, json.JSONDecodeError):
                cached_key = None
            if cached_key == cache_key:
                print(f"Using cached training rows: {items_path}")
                return 0
            print("Training-row cache key changed; rebuilding the cache.")

    # Keep datasets optional when a compatible local cache is already available.
    from datasets import load_dataset

    print(f"Downloading dataset {DATASET_ID} ...")
    dataset = load_dataset(DATASET_ID, "all", split="train")

    items_path.parent.mkdir(parents=True, exist_ok=True)
    n_cases = 0
    with open(items_path, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps({
                "state": json.loads(row["state"]),
                "questions": json.loads(row["questions"]),
                "gold": json.loads(row["gold"]),
            }) + "\n")
            n_cases += 1
    with open(cache_meta_path, "w") as f:
        json.dump(cache_key, f, indent=2)
    print(f"Saved {n_cases} training cases to {items_path}")
    return n_cases


def finalize_config(output_dir, gradient_checkpointing):
    """Write the output config this entry point has always produced.

    Per-type temperatures only: the bucket map `fit_temperature_map` also fits is dropped rather
    than persisted, as before. The rolling `checkpoint_latest/rl_agent_config.json` gets the same
    config, which is what a resumed or inspected checkpoint used to find there.
    """
    path = Path(output_dir) / "rl_agent_config.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg.pop("temperature_by_options", None)
    cfg["gradient_checkpointing"] = gradient_checkpointing
    cfg.update(PERSISTED)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    latest = Path(output_dir) / "checkpoint_latest" / "rl_agent_config.json"
    if latest.exists():
        latest.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def train(args, model_dir, items_path, device):
    summary = laya.train.finetune(
        data=str(items_path),
        model_dir=model_dir,
        output_dir=args.output_dir,
        config=TrainConfig(
            epochs=args.epochs,
            micro_batch=args.micro_batch,
            grad_accum=args.grad_accum,
            calib_max=args.calib_max,
            # The budgets and the epoch order this script has always used.
            max_len=PERSISTED["max_len"],
            head_max_len=PERSISTED["head_max_len"],
            gradient_checkpointing=not args.no_checkpointing,
            seed=args.seed,
            calib_seed=CALIB_SPLIT_SEED,
        ),
        device=str(device),
    )
    cfg = finalize_config(args.output_dir, not args.no_checkpointing)
    print("Train items: {} ({} held out for calibration) | skipped: {}"
          .format(summary["train_items"], summary["calibration_items"], summary["skipped"]))
    print("Temperatures (choice, score, noul):", cfg["temperature"])
    if summary["temperature_by_options"]:
        print("Per-bucket temperatures were fitted but are not persisted here, as before.")
    print(f"Model saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Laya on Apple Silicon MPS")
    parser.add_argument("model_dir", nargs="?", default=None)
    parser.add_argument("output_dir", nargs="?", default=None)
    parser.add_argument("--model-dir", dest="model_dir_option")
    parser.add_argument("--output-dir", dest="output_dir_option")
    parser.add_argument("--items", default=DEFAULT_ITEMS)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--calib-max", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42, help="epoch shuffle base; 42 as this script has always used")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--no-checkpointing", action="store_true")
    args = parser.parse_args()

    if args.model_id != MODEL_ID:
        # Keep the requested model ID local to preparation by updating the
        # module constant before prepare_model() is called.
        globals()["MODEL_ID"] = args.model_id

    args.model_dir = args.model_dir_option or args.model_dir or DEFAULT_MODEL_DIR
    args.output_dir = args.output_dir_option or args.output_dir or DEFAULT_OUTPUT_DIR

    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    model_dir = prepare_model(args.model_dir)
    prepare_rows(args.items, force=args.force_preprocess)
    train(args, model_dir, args.items, device)
    gc.collect()


if __name__ == "__main__":
    main()
