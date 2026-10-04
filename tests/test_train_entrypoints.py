"""The fine-tuning entry points delegate to `laya.train` instead of carrying their own copies.

Weight-free and offline: the scripts are imported with their heavyweight work (checkpoint
downloads, dataset downloads, torch training) patched out, so this only checks that the wrappers
call the library rather than reimplementing it.

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
ENTRY_POINTS = (
    "research/scripts/finetune_single_device.py",
    "notebooks/laya_finetune_typed_decisions_mps.py",
    "notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb",
)


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


def runnable_source(rel):
    return notebook_writefile(rel) if rel.endswith(".ipynb") else (ROOT / rel).read_text(encoding="utf-8")


def local_fitters(rel):
    """Entry-point functions that clamp `exp()` of a fitted log-temperature."""
    found = []
    for match in re.finditer(r"^def (\w+)\(.*?(?=^\S|\Z)", runnable_source(rel), re.M | re.S):
        if re.search(r"torch\.clamp\(\s*\w+\.exp\(\)", match.group(0)):
            found.append(match.group(1))
    return found


def rows_to_jsonl(path, n=3):
    rows = []
    for i in range(n):
        rows.append({
            "state": "the invoice was charged twice",
            "questions": {"q0": {"type": "choice",
                                "instructions": "Where does this belong?",
                                "criteria": {"billing": "Charged twice",
                                             "other": "Anything else"}}},
            "gold": {"q0": {"probabilities": {"billing": 0.9, "other": 0.1}}},
        })
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return str(path)


class EntryPointDelegationTests(unittest.TestCase):
    def test_single_device_script_calls_the_library_finetune(self):
        module = load_script("research/scripts/finetune_single_device.py", "entry_single_device")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        data = rows_to_jsonl(Path(tmp.name) / "data.jsonl")
        with tempfile.TemporaryDirectory() as model_dir:
            calls = {}

            def fake_finetune(**kwargs):
                calls.update(kwargs)
                return {"train_items": 1, "calibration_items": 1, "skipped": {},
                        "epoch_loss": [1.0], "temperature": [1.2, 1.2, 1.2],
                        "temperature_by_options": {}, "n_by_bucket": {}, "output_dir": "out"}

            argv = ["finetune_single_device.py", "--data", data, "--model-dir", model_dir,
                    "--output-dir", "out", "--epochs", "2", "--seed", "7", "--device", "cpu"]
            with patch.object(sys, "argv", argv), \
                 patch("laya.train.finetune", side_effect=fake_finetune) as finetune:
                with contextlib.redirect_stdout(io.StringIO()):
                    module.main()
        self.assertEqual(1, finetune.call_count)
        self.assertEqual(data, calls["data"])
        self.assertEqual(model_dir, calls["model_dir"])
        self.assertEqual("out", calls["output_dir"])
        self.assertEqual("cpu", calls["device"])
        self.assertEqual(2, calls["config"].epochs)
        self.assertEqual(7, calls["config"].seed)

    def test_mps_script_calls_the_library_finetune(self):
        module = load_script("notebooks/laya_finetune_typed_decisions_mps.py", "entry_mps")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        rows = rows_to_jsonl(Path(tmp.name) / "train_items.jsonl")
        with tempfile.TemporaryDirectory() as model_dir:
            # `prepare_model` only downloads when the weights are missing; a stub keeps the test
            # offline and turns any download attempt into an assertion.
            (Path(model_dir) / "model.safetensors").write_bytes(b"")
            calls = {}

            def fake_finetune(**kwargs):
                calls.update(kwargs)
                out = Path(kwargs["output_dir"])
                out.mkdir(parents=True, exist_ok=True)
                (out / "rl_agent_config.json").write_text(json.dumps({"temperature": [1.0, 1.0, 1.0]}),
                                                         encoding="utf-8")
                return {"train_items": 1, "calibration_items": 1, "skipped": {},
                        "epoch_loss": [1.0], "temperature": [1.2, 1.2, 1.2],
                        "temperature_by_options": {}, "n_by_bucket": {}, "output_dir": str(out)}

            argv = ["mps", "--model-dir", model_dir, "--items", rows, "--output-dir",
                    str(Path(tmp.name) / "out"), "--epochs", "3", "--micro-batch", "2",
                    "--grad-accum", "16", "--device", "cpu"]
            with patch.object(sys, "argv", argv), \
                 patch("laya.train.finetune", side_effect=fake_finetune) as finetune, \
                 patch("laya.train.load_checkpoint", side_effect=AssertionError("no checkpoint")), \
                 patch.object(module, "snapshot_download",
                              side_effect=AssertionError("no download")), \
                 patch.object(module, "prepare_rows", return_value=0):
                with contextlib.redirect_stdout(io.StringIO()):
                    module.main()
        self.assertEqual(1, finetune.call_count)
        self.assertEqual(rows, calls["data"])
        self.assertEqual(3, calls["config"].epochs)
        self.assertEqual(2, calls["config"].micro_batch)
        self.assertEqual(16, calls["config"].grad_accum)
        self.assertEqual("cpu", calls["device"])


class NotebookConvergenceTests(unittest.TestCase):
    def test_ddp_script_keeps_ddp_and_uses_the_canonical_fit(self):
        script = notebook_writefile("notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb")
        self.assertIn("DistributedDataParallel", script)
        self.assertIn("fit_temperature_map", script)
        self.assertNotIn("TEMP_MIN", script)
        self.assertEqual([], local_fitters("notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb"))


if __name__ == "__main__":
    unittest.main()
