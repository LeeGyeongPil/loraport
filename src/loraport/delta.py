"""Compare adapters by their weight delta dW = scale * B @ A without building dW.

For ``U = [s1*B1, -s2*B2]`` (out x (r1+r2)) and ``V = [A1; A2]`` we have
``dW1 - dW2 = U @ V`` and ``||U V||_F^2 = trace((U^T U)(V V^T))``, which only
needs (r1+r2)-square matrices.  Everything runs in float64.  Squaring limits the
precision floor of a relative error to about 1e-8, far below any real mistake.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .model import Adapter, LoraModule

_TINY = 1e-30


def _sq_norm(U: np.ndarray, V: np.ndarray) -> float:
    gram_u = U.T @ U
    gram_v = V @ V.T
    return max(float(np.sum(gram_u * gram_v)), 0.0)  # both Gram matrices are symmetric


def delta_norm(m: LoraModule) -> float:
    """Frobenius norm of dW."""
    A = m.A.astype(np.float64)
    B = m.B.astype(np.float64)
    return math.sqrt(_sq_norm(m.scale * B, A))


def diff_norm(m1: LoraModule, m2: LoraModule) -> float:
    """Frobenius norm of dW1 - dW2 (shapes must match)."""
    U = np.concatenate([m1.scale * m1.B.astype(np.float64), -m2.scale * m2.B.astype(np.float64)], axis=1)
    V = np.concatenate([m1.A.astype(np.float64), m2.A.astype(np.float64)], axis=0)
    return math.sqrt(_sq_norm(U, V))


def dense_delta(m: LoraModule) -> np.ndarray:
    """Materialise dW (float64). Only for tests and small modules."""
    return m.scale * (m.B.astype(np.float64) @ m.A.astype(np.float64))


def _shapes(m: LoraModule) -> str:
    return f"A{tuple(m.A.shape)} B{tuple(m.B.shape)}"


@dataclass
class Report:
    """Result of comparing a reference adapter ``a`` with a candidate ``b``."""

    ok: bool
    rtol: float
    n_reference: int
    missing: list[str] = field(default_factory=list)  # in a, absent in b
    extra: list[str] = field(default_factory=list)  # in b, absent in a
    shape_mismatch: list[str] = field(default_factory=list)
    key_issues: list[str] = field(default_factory=list)
    per_module: dict[str, float] = field(default_factory=dict)  # relative dW error
    worst_rel_err: float = 0.0
    worst_module: str | None = None

    @property
    def bad_modules(self) -> list[str]:
        return [n for n, e in self.per_module.items() if not e <= self.rtol]

    def headline(self) -> str:
        good = self.n_reference - len(self.missing) - len(self.shape_mismatch) - len(self.bad_modules)
        if self.ok:
            return f"PASS {self.n_reference}/{self.n_reference} modules, worst rel err {self.worst_rel_err:.1e}"
        return (
            f"FAIL {max(good, 0)}/{self.n_reference} modules match "
            f"(missing {len(self.missing)}, extra {len(self.extra)}, shape mismatch {len(self.shape_mismatch)}, "
            f"dW off {len(self.bad_modules)}, ignored/dropped tensors {len(self.key_issues)})"
        )

    def to_dict(self) -> dict[str, Any]:
        def num(x: float) -> float | None:
            return x if math.isfinite(x) else None

        return {
            "ok": self.ok,
            "rtol": self.rtol,
            "modules_in_reference": self.n_reference,
            "missing": self.missing,
            "extra": self.extra,
            "shape_mismatch": self.shape_mismatch,
            "key_issues": self.key_issues,
            "bad_modules": self.bad_modules,
            "worst_rel_err": num(self.worst_rel_err),
            "worst_module": self.worst_module,
            "per_module": {k: num(v) for k, v in self.per_module.items()},
        }


def compare(a: Adapter, b: Adapter, *, rtol: float = 1e-5) -> Report:
    """Compare per-module dW of ``a`` (reference) and ``b``.

    The report is ``ok`` only if no module is missing/extra, no shape differs,
    no tensor was ignored or dropped on either side, and every relative error
    ``||dW_a - dW_b|| / ||dW_a||`` is within ``rtol``.
    """
    missing = sorted(a.modules.keys() - b.modules.keys())
    extra = sorted(b.modules.keys() - a.modules.keys())
    shape_mismatch: list[str] = []
    per_module: dict[str, float] = {}
    for name in sorted(a.modules.keys() & b.modules.keys()):
        ma, mb = a.modules[name], b.modules[name]
        if not (ma.consistent and mb.consistent) or ma.A.shape != mb.A.shape or ma.B.shape != mb.B.shape:
            shape_mismatch.append(f"{name}: {_shapes(ma)} vs {_shapes(mb)}")
            continue
        ref = delta_norm(ma)
        diff = diff_norm(ma, mb)
        if ref < _TINY:
            per_module[name] = 0.0 if diff < _TINY else math.inf
        else:
            per_module[name] = diff / ref
    issues = [f"a: {s}" for s in a.key_issues] + [f"b: {s}" for s in b.key_issues]
    worst_module = max(per_module, key=lambda n: per_module[n], default=None)
    worst = per_module[worst_module] if worst_module else 0.0
    ok = (
        len(a.modules) > 0
        and not (missing or extra or shape_mismatch or issues)
        and all(e <= rtol for e in per_module.values())
    )
    return Report(ok, rtol, len(a.modules), missing, extra, shape_mismatch, issues, per_module, worst, worst_module)
