"""DataLoader, checkpointing and the synthetic datasets."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

import tensorgrad as tg
from tensorgrad import nn
from tensorgrad.datasets import (
    SHAPE_CLASSES,
    TINY_SHAKESPEARE_SHA256,
    CharTokenizer,
    cache_dir,
    fetch,
    make_shapes,
    make_spiral,
)
from tensorgrad.utils import DataLoader, load, save


class TestDataLoader:
    def test_batches_cover_the_data_in_order(self) -> None:
        x, y = np.arange(20).reshape(10, 2), np.arange(10)
        loader = DataLoader(x, y, batch_size=4)
        batches = list(loader)
        assert len(loader) == len(batches) == 3
        assert [b[0].shape for b in batches] == [(4, 2), (4, 2), (2, 2)]
        np.testing.assert_array_equal(np.concatenate([b[1].data for b in batches]), y)

    def test_drop_last(self) -> None:
        loader = DataLoader(np.arange(10), batch_size=4, drop_last=True)
        assert len(loader) == 2
        assert [b[0].shape[0] for b in loader] == [4, 4]

    def test_shuffle_is_seeded_and_changes_every_epoch(self) -> None:
        data = np.arange(50)
        first = [b[0].data for b in DataLoader(data, batch_size=50, shuffle=True, seed=3)]
        again = [b[0].data for b in DataLoader(data, batch_size=50, shuffle=True, seed=3)]
        np.testing.assert_array_equal(first[0], again[0])
        loader = DataLoader(data, batch_size=50, shuffle=True, seed=3)
        epoch1, epoch2 = next(iter(loader))[0].data, next(iter(loader))[0].data
        assert not np.array_equal(epoch1, epoch2)
        np.testing.assert_array_equal(np.sort(epoch1), data)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="different lengths"):
            DataLoader(np.zeros(3), np.zeros(4))
        with pytest.raises(ValueError, match="at least one"):
            DataLoader()
        with pytest.raises(ValueError, match="batch_size"):
            DataLoader(np.zeros(3), batch_size=0)


class TestSerialization:
    def test_module_round_trip(self, tmp_path: Path) -> None:
        model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
        model(tg.randn((6, 3)))
        path = tmp_path / "model.ckpt"
        save(model, path)
        assert path.exists()  # written to exactly this path, no ".npz" appended
        restored = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
        restored.load_state_dict(load(path))
        for (name, a), b in zip(
            model.state_dict().items(), restored.state_dict().values(), strict=True
        ):
            np.testing.assert_array_equal(a, b, err_msg=name)

    def test_plain_mapping_round_trip(self, tmp_path: Path) -> None:
        state = {"a.b": np.arange(3), "c": np.ones((2, 2), dtype=np.float32)}
        save(state, tmp_path / "state.npz")
        loaded = load(tmp_path / "state.npz")
        assert loaded.keys() == state.keys()
        assert loaded["c"].dtype == np.float32


class TestDatasets:
    def test_spiral(self) -> None:
        x, y = make_spiral(50, 3, seed=0)
        assert x.shape == (150, 2)
        assert x.dtype == np.float32
        assert np.bincount(y).tolist() == [50, 50, 50]
        np.testing.assert_array_equal(make_spiral(50, 3, seed=0)[0], x)

    def test_shapes(self) -> None:
        images, labels = make_shapes(40, size=16, seed=0)
        assert images.shape == (40, 1, 16, 16)
        assert images.dtype == np.float32
        assert images.min() >= 0.0 and images.max() <= 1.0
        assert np.bincount(labels).tolist() == [10] * len(SHAPE_CLASSES)
        # Every class draws a visible stroke.
        clean, _ = make_shapes(8, noise=0.0, seed=1)
        assert (clean.reshape(8, -1).max(axis=1) > 0.9).all()

    def test_char_tokenizer(self) -> None:
        tok = CharTokenizer("hello world")
        assert tok.vocab_size == 8
        ids = tok.encode("low")
        assert ids.dtype == np.int64
        assert tok.decode(ids) == "low"
        with pytest.raises(KeyError):
            tok.encode("z")
        with pytest.raises(ValueError, match="empty"):
            CharTokenizer("")


class TestFetch:
    """The downloader, exercised offline through file:// URLs."""

    @staticmethod
    def source(tmp_path: Path, content: bytes = b"to be, or not to be") -> tuple[str, str, Path]:
        src = tmp_path / "remote" / "corpus.txt"
        src.parent.mkdir()
        src.write_bytes(content)
        return src.as_uri(), hashlib.sha256(content).hexdigest(), src

    def test_downloads_verifies_and_caches(self, tmp_path: Path) -> None:
        url, digest, src = self.source(tmp_path)
        path = fetch(url, digest, "corpus.txt", cache=tmp_path / "cache")
        assert path.read_bytes() == b"to be, or not to be"
        src.unlink()  # a second call must be served from the cache
        assert fetch(url, digest, "corpus.txt", cache=tmp_path / "cache") == path

    def test_rejects_a_mismatching_download_and_leaves_nothing(self, tmp_path: Path) -> None:
        url, _, _ = self.source(tmp_path)
        cache = tmp_path / "cache"
        with pytest.raises(ValueError, match="checksum mismatch"):
            fetch(url, "0" * 64, "corpus.txt", cache=cache)
        assert list(cache.iterdir()) == []

    def test_replaces_a_corrupted_cached_copy(self, tmp_path: Path) -> None:
        url, digest, _ = self.source(tmp_path)
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "corpus.txt").write_bytes(b"truncated")
        assert fetch(url, digest, "corpus.txt", cache=cache).read_bytes() == b"to be, or not to be"

    def test_cache_directory_follows_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TENSORGRAD_CACHE", str(tmp_path / "explicit"))
        assert cache_dir() == tmp_path / "explicit"
        monkeypatch.delenv("TENSORGRAD_CACHE")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        assert cache_dir() == tmp_path / "xdg" / "tensorgrad"

    def test_pinned_checksum_is_a_sha256_hex_digest(self) -> None:
        assert len(TINY_SHAKESPEARE_SHA256) == 64
        int(TINY_SHAKESPEARE_SHA256, 16)


class TestRetainFreedMemory:
    SCRIPT = """
import resource, sys
import numpy as np
from tensorgrad.utils import retain_freed_memory
applied = retain_freed_memory() if sys.argv[1] == "tuned" else False
before = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
for _ in range(20):  # like training steps: many activations alive at once, freed together
    live = [np.ones(400_000) for _ in range(30)]  # 30 arrays of 3.2 MB, every page written
    del live
print(applied, resource.getrusage(resource.RUSAGE_SELF).ru_minflt - before)
"""

    def run(self, mode: str) -> tuple[bool, int]:
        import subprocess
        import sys

        out = subprocess.run(
            [sys.executable, "-c", self.SCRIPT, mode], capture_output=True, text=True, check=True
        ).stdout.split()
        return out[0] == "True", int(out[1])

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc tuning only")
    def test_freed_arrays_are_reused_instead_of_refaulted(self) -> None:
        # Run in subprocesses: the setting is process-wide and must not leak into this one.
        applied, tuned_faults = self.run("tuned")
        if not applied:
            pytest.skip("not a glibc system")
        _, default_faults = self.run("default")
        # By default every "step" faults its ~96 MB back in; tuned, only the first one does.
        assert tuned_faults * 5 < default_faults
