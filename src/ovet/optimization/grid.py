"""Grid enumeration + Pareto-front analysis for Phase 5.

The optimisation space we sweep over is::

    {alpha} × {projection_removal} × {layer_set} × {instruct_proxy}

with each cell evaluated over ``reps`` repetitions to average out
generation noise. Aggregated cells (one per parameter combination) are
then ranked by:

1. **Constrained best** — maximise emotion fidelity subject to a
   speaker_sim hard threshold (the design's headline acceptance
   criterion).
2. **Pareto front** along (emotion_score, speaker_sim).

Both helpers are pure-logic and tested without GPU.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Sequence
from itertools import product
import math


# ---------------------------------------------------------------------
# Grid specification
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GridPoint:
    alpha: float
    projection_removal: bool
    layer_ids: tuple[int, ...]
    instruct: str | None

    @property
    def tag(self) -> str:
        ls = "-".join(str(x) for x in self.layer_ids) or "none"
        ip = (self.instruct or "none").replace(",", "_").replace(" ", "_")
        return f"a{self.alpha:.2f}_p{int(self.projection_removal)}_L{ls}_I{ip}"


@dataclass
class GridSpec:
    alphas:     list[float]
    projections: list[bool]
    layer_sets: list[tuple[int, ...]]
    instructs:  list[str | None]

    def enumerate(self) -> list[GridPoint]:
        out = []
        for a, p, L, i in product(self.alphas, self.projections, self.layer_sets, self.instructs):
            out.append(GridPoint(alpha=a, projection_removal=p,
                                 layer_ids=tuple(L), instruct=i))
        return out

    def __len__(self) -> int:
        return len(self.alphas) * len(self.projections) * len(self.layer_sets) * len(self.instructs)


# ---------------------------------------------------------------------
# Pareto-front computation
# ---------------------------------------------------------------------

@dataclass
class ParetoPoint:
    """A single objective-vector with arbitrary metadata."""
    objectives: tuple[float, ...]   # higher is better on every axis
    meta: dict = field(default_factory=dict)


def is_dominated(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """True iff ``b`` dominates ``a`` (b ≥ a on all axes, b > a on at least one).

    Both vectors must have the same length and follow the
    "higher is better" convention.
    """
    if len(a) != len(b):
        raise ValueError("objective length mismatch")
    ge_all = all(bi >= ai - 1e-12 for ai, bi in zip(a, b))
    gt_any = any(bi > ai + 1e-12 for ai, bi in zip(a, b))
    return ge_all and gt_any


def pareto_front(points: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """Return the non-dominated subset of ``points``."""
    pts = list(points)
    out = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            if is_dominated(p.objectives, q.objectives):
                dominated = True
                break
        if not dominated:
            out.append(p)
    return out


# ---------------------------------------------------------------------
# Constrained best
# ---------------------------------------------------------------------

@dataclass
class Constraint:
    metric: str
    op: str          # "<=" or ">="
    value: float

    def passes(self, metrics: dict) -> bool:
        v = metrics.get(self.metric)
        if v is None:
            return False
        if self.op == "<=":
            return v <= self.value
        if self.op == ">=":
            return v >= self.value
        raise ValueError(self.op)


def constrained_best(
    cells: list[dict],
    objective: str,
    objective_direction: str,         # "min" or "max"
    constraints: Iterable[Constraint] = (),
) -> dict | None:
    """Pick the cell that optimises ``objective`` subject to ``constraints``.

    Each cell is a dict containing at least ``"metrics": dict[str, float]``.
    Returns the winning cell or ``None`` if no cell satisfies the
    constraints.
    """
    valid = [c for c in cells if all(k.passes(c["metrics"]) for k in constraints)]
    if not valid:
        return None
    direction = -1 if objective_direction == "min" else 1
    return max(valid, key=lambda c: direction * c["metrics"].get(objective, -math.inf))


# ---------------------------------------------------------------------
# Helpers to build the higher-is-better objective vector
# ---------------------------------------------------------------------

def make_objectives(metrics: dict, axes: Sequence[tuple[str, str]]) -> tuple[float, ...]:
    """Build a higher-is-better objective vector from ``metrics``.

    ``axes`` is a sequence of ``(metric_name, direction)`` where direction
    is ``"max"`` (use as-is) or ``"min"`` (negate).
    """
    out = []
    for name, direction in axes:
        v = float(metrics.get(name, 0.0))
        out.append(v if direction == "max" else -v)
    return tuple(out)
