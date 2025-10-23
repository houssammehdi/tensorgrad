"""DataLoader, checkpointing and the synthetic datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tensorgrad as tg
from tensorgrad import nn
from tensorgrad.datasets import SHAPE_CLASSES, CharTokenizer, make_shapes, make_spiral
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
