"""The single-process fine-tuning entry points delegate to `laya.train`; Kaggle reuses its pieces.

What is asserted here is the contract each entry point has always had, not just that it calls the
library: the CLI values reach `laya.train.finetune` unchanged, `--seed` still decides the
calibration split, `auto` still means CUDA-or-CPU on the single-device script, and the persisted
output config keeps its old shape (per-type temperatures only, plus this script's own defaults).

The Kaggle notebook is not covered by the delegation tests because it keeps its DDP
orchestration; `tests/test_calibration_persistence.py` holds the invariant that none of the three
carries a private temperature fitter.

Weight-free and offline: the scripts are imported with their heavyweight work (checkpoint
downloads, dataset downloads, training) patched out.

Run: python tests/test_train_entrypoints.py
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
SINGLE_DEVICE = "research/scripts/finetune_single_device.py"
MPS = "notebooks/laya_finetune_typed_decisions_mps.py"
KAGGLE = "notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb"


def load_script(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def notebook_writefile(rel):
    """The source a notebook actually executes: its `%%writefile` cells."""
    cells = json.loads((ROOT / rel).read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(c["source"]) for c in cells
                     if "".join(c["source"]).startswith("%%writefile "))


def local_fitters(rel):
    """Entry-point functions that clamp `exp()` of a fitted log-temperature."""
    if rel.endswith(".ipynb"):
        text = notebook_writefile(rel)
    else:
        text = (ROOT / rel).read_text(encoding="utf-8")
    return [m.group(1) for m in re.finditer(r"^def (\w+)\(.*?(?=^\S|\Z)", text, re.M | re.S)
            if re.search(r"torch\.clamp\(\s*\w+\.exp\(\)", m.group(0))]


def rows_to_jsonl(path, n=3):
    rows = [{
        "state": "the invoice was charged twice",
        "questions": {"q0": {"type": "choice", "instructions": "Where does this belong?",
                             "criteria": {"billing": "Charged twice", "other": "Anything else"}}},
        "gold": {"q0": {"probabilities": {"billing": 0.9, "other": 0.1}}},
    } for _ in range(n)]
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return str(path)


def fake_run(output_dir, checkpoint_latest=False):
    """Stand in for `laya.train.finetune`, writing what the real one writes.

    It deliberately writes a `temperature_by_options` map and the library's own config values:
    the wrappers must strip and restore them, so a fake that started from the wrapper's wanted
    state would prove nothing.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if checkpoint_latest:
        (out / "checkpoint_latest").mkdir(parents=True, exist_ok=True)
        (out / "checkpoint_latest" / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    (out / "rl_agent_config.json").write_text(json.dumps({
        "max_len": 64, "head_max_len": 32, "fine_tuned": True,
        "temperature": [1.2, 1.2, 1.2],
        "temperature_by_options": {"choice:2": 1.9},
        "training": {"laya_train": {"epochs": 4}},
    }), encoding="utf-8")
    return {"train_items": 1, "calibration_items": 1, "skipped": {}, "epoch_loss": [1.0],
            "temperature": [1.2, 1.2, 1.2], "temperature_by_options": {"choice:2": 1.9},
            "n_by_bucket": {"choice:2": 1}, "output_dir": str(out)}


class SingleDeviceContractTests(unittest.TestCase):
    """`research/scripts/finetune_single_device.py`."""

    def setUp(self):
        self.module = load_script(SINGLE_DEVICE, "entry_single_device")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = rows_to_jsonl(Path(self.tmp.name) / "data.jsonl")
        self.model_dir = Path(self.tmp.name) / "model"
        self.model_dir.mkdir()
        (self.model_dir / "model.safetensors").write_bytes(b"")
        self.out = Path(self.tmp.name) / "out"

    def run_main(self, argv_extra=()):
        calls = {}

        def fake_finetune(**kwargs):
            calls.update(kwargs)
            return fake_run(kwargs["output_dir"])

        argv = ["finetune_single_device.py", "--data", self.data, "--model-dir", str(self.model_dir),
                "--output-dir", str(self.out)] + list(argv_extra)
        with patch.object(sys, "argv", argv), \
             patch("laya.train.finetune", side_effect=fake_finetune) as finetune, \
             patch("laya.train.load_checkpoint", side_effect=AssertionError("no checkpoint")), \
             patch.object(self.module, "snapshot_download",
                          side_effect=AssertionError("no download")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.module.main()
        self.assertEqual(1, finetune.call_count)
        return calls

    def test_cli_values_reach_the_library(self):
        calls = self.run_main(["--epochs", "2", "--seed", "7", "--device", "cpu"])
        self.assertEqual(self.data, calls["data"])
        self.assertEqual(str(self.model_dir), calls["model_dir"])
        self.assertEqual(str(self.out), calls["output_dir"])
        self.assertEqual("cpu", calls["device"])
        self.assertEqual(2, calls["config"].epochs)
        self.assertEqual(7, calls["config"].seed)
        # The old script stepped the optimizer on every micro-batch: one effective batch of 8.
        self.assertEqual(8, calls["config"].micro_batch)
        self.assertEqual(1, calls["config"].grad_accum)
        self.assertEqual(1024, calls["config"].max_len)
        self.assertEqual(256, calls["config"].head_max_len)

    def test_seed_also_decides_the_calibration_split(self):
        for seed in (0, 7, 99):
            with self.subTest(seed=seed):
                calls = self.run_main(["--seed", str(seed), "--device", "cpu"])
                self.assertEqual(seed, calls["config"].calib_seed)

    def test_auto_device_is_cuda_or_cpu_only(self):
        with patch.object(self.module.torch.cuda, "is_available", return_value=True):
            self.assertEqual("cuda", self.module.resolve_device("auto"))
        with patch.object(self.module.torch.cuda, "is_available", return_value=False):
            self.assertEqual("cpu", self.module.resolve_device("auto"))
        # An explicit device is left alone, including mps.
        for requested in ("cpu", "cuda", "mps"):
            self.assertEqual(requested, self.module.resolve_device(requested))
        calls = self.run_main(["--device", "auto"])
        self.assertEqual("cpu", calls["device"])

    def test_persisted_config_keeps_this_scripts_old_shape(self):
        self.run_main(["--device", "cpu"])
        cfg = json.loads((self.out / "rl_agent_config.json").read_text(encoding="utf-8"))
        self.assertEqual([1.2, 1.2, 1.2], cfg["temperature"])
        # The bucket map is fitted inside the library and dropped here, as before.
        self.assertNotIn("temperature_by_options", cfg)
        self.assertEqual(4096, cfg["max_tokens_per_batch"])
        self.assertEqual(1024, cfg["max_len"])
        self.assertEqual(256, cfg["head_max_len"])
        self.assertFalse(cfg["gradient_checkpointing"])
        self.assertTrue(cfg["fine_tuned"])

    def test_persisted_gradient_checkpointing_follows_amp(self):
        with patch.object(self.module.torch.cuda, "is_available", return_value=True):
            calls = self.run_main(["--device", "auto"])
        self.assertEqual("cuda", calls["device"])
        cfg = json.loads((self.out / "rl_agent_config.json").read_text(encoding="utf-8"))
        self.assertTrue(cfg["gradient_checkpointing"])


class MpsContractTests(unittest.TestCase):
    """`notebooks/laya_finetune_typed_decisions_mps.py`."""

    def setUp(self):
        self.module = load_script(MPS, "entry_mps")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rows = rows_to_jsonl(Path(self.tmp.name) / "train_items.jsonl")
        self.model_dir = Path(self.tmp.name) / "model"
        self.model_dir.mkdir()
        # `prepare_model` only downloads when the weights are missing; a stub keeps the test
        # offline and turns any download attempt into an assertion.
        (self.model_dir / "model.safetensors").write_bytes(b"")
        self.out = Path(self.tmp.name) / "out"

    def run_main(self, argv_extra=()):
        calls = {}

        def fake_finetune(**kwargs):
            calls.update(kwargs)
            return fake_run(kwargs["output_dir"], checkpoint_latest=True)

        argv = ["mps", "--model-dir", str(self.model_dir), "--items", self.rows,
                "--output-dir", str(self.out), "--device", "cpu"] + list(argv_extra)
        with patch.object(sys, "argv", argv), \
             patch("laya.train.finetune", side_effect=fake_finetune) as finetune, \
             patch("laya.train.load_checkpoint", side_effect=AssertionError("no checkpoint")), \
             patch.object(self.module, "snapshot_download",
                          side_effect=AssertionError("no download")), \
             patch.object(self.module, "prepare_rows", return_value=0), \
             contextlib.redirect_stdout(io.StringIO()):
            self.module.main()
        self.assertEqual(1, finetune.call_count)
        return calls

    def test_cli_values_reach_the_library(self):
        calls = self.run_main(["--epochs", "3", "--micro-batch", "2", "--grad-accum", "16"])
        self.assertEqual(self.rows, calls["data"])
        self.assertEqual(3, calls["config"].epochs)
        self.assertEqual(2, calls["config"].micro_batch)
        self.assertEqual(16, calls["config"].grad_accum)
        self.assertEqual(1024, calls["config"].max_len)
        self.assertEqual(256, calls["config"].head_max_len)

    def test_epoch_order_keeps_its_former_base(self):
        # The old script shuffled each epoch with `random.Random(42 + epoch)`; the shared loop
        # uses `config.seed + epoch`, so the default has to stay 42 for the same order.
        calls = self.run_main()
        self.assertEqual(42, calls["config"].seed)

    def test_persisted_config_keeps_this_scripts_old_shape(self):
        self.run_main()
        cfg = json.loads((self.out / "rl_agent_config.json").read_text(encoding="utf-8"))
        self.assertEqual([1.2, 1.2, 1.2], cfg["temperature"])
        self.assertNotIn("temperature_by_options", cfg)
        self.assertEqual(2048, cfg["max_tokens_per_batch"])
        self.assertEqual(1024, cfg["max_len"])
        self.assertEqual(256, cfg["head_max_len"])
        self.assertEqual("laya-typed-decisions", cfg["model_name"])
        self.assertTrue(cfg["gradient_checkpointing"])
        # The rolling checkpoint keeps the same config, as it used to.
        latest = json.loads(
            (self.out / "checkpoint_latest" / "rl_agent_config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg, latest)

    def test_no_checkpointing_flag_is_persisted_and_passed(self):
        calls = self.run_main(["--no-checkpointing"])
        self.assertFalse(calls["config"].gradient_checkpointing)
        cfg = json.loads((self.out / "rl_agent_config.json").read_text(encoding="utf-8"))
        self.assertFalse(cfg["gradient_checkpointing"])


class KaggleConvergenceTests(unittest.TestCase):
    """The notebook keeps DDP and reuses the library's building blocks."""

    def test_ddp_script_keeps_ddp_and_drops_its_private_fitter(self):
        script = notebook_writefile(KAGGLE)
        self.assertIn("DistributedDataParallel", script)
        self.assertIn("fit_temperature_map", script)
        self.assertEqual([], local_fitters(KAGGLE))


if __name__ == "__main__":
    unittest.main()
