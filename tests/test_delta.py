"""Low-rank dW comparison (AC-5)."""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

from loraport import Adapter, LoraModule, compare
from loraport.delta import delta_norm, dense_delta, diff_norm


def _module(rng: np.random.Generator, out: int, inn: int, r: int, scale: float) -> LoraModule:
    return LoraModule(
        rng.normal(size=(r, inn)).astype(np.float32), rng.normal(size=(out, r)).astype(np.float32), scale
    )


def test_lowrank_matches_dense() -> None:
    rng = np.random.default_rng(1)
    m1 = _module(rng, 48, 40, 4, 2.5)
    m2 = _module(rng, 48, 40, 6, 0.7)  # different rank and scale on purpose
    d1, d2 = dense_delta(m1), dense_delta(m2)
    assert abs(delta_norm(m1) - np.linalg.norm(d1)) / np.linalg.norm(d1) < 1e-10
    assert abs(diff_norm(m1, m2) - np.linalg.norm(d1 - d2)) / np.linalg.norm(d1 - d2) < 1e-10


def test_identical_modules_have_zero_difference() -> None:
    m = _module(np.random.default_rng(2), 30, 20, 3, 1.0)
    assert diff_norm(m, m) / delta_norm(m) < 1e-7  # sqrt of float64 round-off


def test_large_module_no_dense() -> None:
    """A 4096x11008 module (rank 16) compares in well under 2 s and far below the 360 MB a dense dW needs."""
    rng = np.random.default_rng(3)
    big1 = _module(rng, 4096, 11008, 16, 20.0)
    big2 = LoraModule(big1.A.copy(), big1.B * 1.01, 20.0)  # 1% bigger B -> rel err 0.01
    a = Adapter("mlx", 16, 20.0, {"model.layers.0.mlp.down_proj": big1})
    b = Adapter("peft", 16, 20.0, {"model.layers.0.mlp.down_proj": big2})
    tracemalloc.start()
    t0 = time.perf_counter()
    rep = compare(a, b, rtol=0.05)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert rep.ok and abs(rep.worst_rel_err - 0.01) < 1e-6
    assert elapsed < 2.0, f"took {elapsed:.2f}s"
    assert peak < 150e6, f"peak {peak / 1e6:.0f} MB"


def test_zero_reference_module() -> None:
    z = LoraModule(np.ones((2, 5), np.float32), np.zeros((4, 2), np.float32), 1.0)
    nz = LoraModule(np.ones((2, 5), np.float32), np.ones((4, 2), np.float32), 1.0)
    same = compare(Adapter("mlx", 2, 1.0, {"m": z}), Adapter("peft", 2, 1.0, {"m": z}))
    assert same.ok
    differ = compare(Adapter("mlx", 2, 1.0, {"m": z}), Adapter("peft", 2, 1.0, {"m": nz}))
    assert not differ.ok and differ.to_dict()["worst_rel_err"] is None  # inf is serialised as null


def test_empty_reference_never_passes() -> None:
    assert not compare(Adapter("mlx", 4, 1.0, {}), Adapter("peft", 4, 1.0, {})).ok
