"""Smoke-run every example script with tiny settings (catches API drift in the examples)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run_example(script: str, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_spiral_example() -> None:
    out = run_example("spiral_mlp.py", "--epochs", "150", "--points", "100", "--no-plot")
    accuracy = float(out.split("train accuracy ")[1].split()[0])
    assert accuracy > 0.9


def test_shapes_example() -> None:
    out = run_example(
        "shapes_cnn.py", "--epochs", "1", "--train-size", "256", "--test-size", "64", "--no-plot"
    )
    assert "confusion matrix" in out
    assert "final test accuracy" in out


def test_char_gpt_example_with_checkpoint(tmp_path: Path) -> None:
    ckpt = tmp_path / "gpt.npz"
    common = ["--n-layer", "1", "--n-embd", "32", "--block-size", "16", "--batch-size", "8"]
    out = run_example(
        "char_gpt.py", "--steps", "10", "--eval-interval", "5", "--eval-batches", "2",
        "--sample-chars", "40", "--checkpoint", str(ckpt), "--no-plot", *common,
    )  # fmt: skip
    assert "--- sample ---" in out
    assert ckpt.exists()
    resumed = run_example(
        "char_gpt.py", "--steps", "2", "--eval-interval", "1", "--eval-batches", "1",
        "--sample-chars", "10", "--resume", str(ckpt), "--no-plot", *common,
    )  # fmt: skip
    assert "step     2" in resumed


def test_newton_example_converges_in_a_few_iterations() -> None:
    out = run_example("newton_logreg.py", "--samples", "400", "--gd-steps", "20", "--no-plot")
    newton_line = next(line for line in out.splitlines() if line.startswith("Newton, exact"))
    assert int(newton_line.split()[3]) <= 10  # quadratic convergence from w = 0


@pytest.mark.parametrize("script", ["spiral_mlp.py", "shapes_cnn.py", "char_gpt.py"])
def test_examples_expose_epochs_or_steps_and_seed(script: str) -> None:
    help_text = run_example(script, "--help")
    assert "--seed" in help_text
    assert "--epochs" in help_text or "--steps" in help_text
