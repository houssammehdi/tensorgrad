"""Smoke-run every example script with tiny settings (catches API drift in the examples)."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
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
    # The bundled excerpt keeps the test offline (by default the script downloads).
    common = [
        "--corpus", str(EXAMPLES / "data" / "shakespeare.txt"),
        "--n-layer", "1", "--n-embd", "32", "--block-size", "16", "--batch-size", "8",
    ]  # fmt: skip
    out = run_example(
        "char_gpt.py", "--steps", "10", "--eval-interval", "5", "--eval-batches", "2",
        "--sample-chars", "40", "--checkpoint", str(ckpt), "--no-plot", *common,
    )  # fmt: skip
    assert "--- sample (40 chars" in out
    assert ckpt.exists()
    resumed = run_example(
        "char_gpt.py", "--steps", "2", "--eval-interval", "1", "--eval-batches", "1",
        "--sample-chars", "10", "--resume", str(ckpt), "--no-plot", *common,
    )  # fmt: skip
    assert "step     2" in resumed


def test_char_gpt_windows_cover_the_last_character(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: starts were drawn from [0, n - T - 1), so the final window was never used
    # and a corpus of exactly one window (n = T + 1 tokens) raised in rng.integers.
    monkeypatch.syspath_prepend(str(EXAMPLES))
    char_gpt = importlib.import_module("char_gpt")
    rng = np.random.default_rng(0)
    assert set(char_gpt.random_starts(rng, 9, 8, 50).tolist()) == {0}
    assert set(char_gpt.random_starts(rng, 10, 8, 200).tolist()) == {0, 1}
    data = np.arange(10)
    x, y = char_gpt.windows(data, np.array([1]), 8)
    assert x.tolist() == [list(range(1, 9))]
    assert y.tolist() == [list(range(2, 10))]  # the target of the last window is data[-1]


def test_newton_example_converges_in_a_few_iterations() -> None:
    out = run_example("newton_logreg.py", "--samples", "400", "--gd-steps", "20", "--no-plot")
    newton_line = next(line for line in out.splitlines() if line.startswith("Newton, exact"))
    assert int(newton_line.split()[3]) <= 10  # quadratic convergence from w = 0


def test_maml_example_runs_all_three_methods() -> None:
    out = run_example(
        "maml_sinusoid.py", "--iterations", "3", "--meta-batch", "4", "--test-tasks", "8",
        "--no-plot",
    )  # fmt: skip
    for method in ("MAML", "first-order MAML", "pretrained"):
        assert any(line.startswith(method + " ") for line in out.splitlines()), method


def test_adding_problem_example_runs_every_cell() -> None:
    out = run_example(
        "adding_problem.py", "--length", "12", "--steps", "20", "--eval-interval", "10",
        "--test-size", "64", "--hidden", "8", "--no-plot",
    )  # fmt: skip
    for name in ("RNN", "GRU", "LSTM"):
        assert any(line.startswith(f"{name} ") and "test MSE" in line for line in out.splitlines())


@pytest.mark.parametrize(
    "script",
    ["spiral_mlp.py", "shapes_cnn.py", "char_gpt.py", "adding_problem.py"],
)
def test_examples_expose_epochs_or_steps_and_seed(script: str) -> None:
    help_text = run_example(script, "--help")
    assert "--seed" in help_text
    assert "--epochs" in help_text or "--steps" in help_text
