"""Pure-logic tests for Phase 5 grid + Pareto + constrained selection."""
from __future__ import annotations
import pytest

from ovet.optimization.grid import (
    GridSpec, GridPoint, ParetoPoint,
    is_dominated, pareto_front,
    Constraint, constrained_best, make_objectives,
)


# ---------------------------------------------------------------------
# GridSpec / enumerate
# ---------------------------------------------------------------------

def test_gridspec_size():
    spec = GridSpec(alphas=[0, 1], projections=[True, False],
                    layer_sets=[(8,), (8, 12)], instructs=[None, "high pitch"])
    assert len(spec) == 2 * 2 * 2 * 2 == 16
    assert len(spec.enumerate()) == 16


def test_gridpoint_tag_is_filename_safe():
    gp = GridPoint(alpha=0.5, projection_removal=True,
                   layer_ids=(8, 12), instruct="high pitch")
    tag = gp.tag
    assert "/" not in tag and " " not in tag
    assert "a0.50" in tag
    assert "p1"    in tag
    assert "L8-12" in tag


def test_gridpoint_tag_handles_none_instruct():
    gp = GridPoint(alpha=0.0, projection_removal=False,
                   layer_ids=(), instruct=None)
    assert "Inone" in gp.tag


# ---------------------------------------------------------------------
# is_dominated / pareto_front
# ---------------------------------------------------------------------

def test_is_dominated_basic():
    a = (1.0, 1.0)
    b = (2.0, 2.0)   # b dominates a
    c = (1.0, 2.0)   # c weakly dominates a (≥ all, > one)
    d = (0.0, 3.0)   # d does NOT dominate a (≥ on axis 1, < on axis 0)
    assert is_dominated(a, b) is True
    assert is_dominated(a, c) is True
    assert is_dominated(a, d) is False


def test_is_dominated_equal_is_not_dominated():
    a = (1.0, 1.0)
    b = (1.0, 1.0)
    assert is_dominated(a, b) is False
    assert is_dominated(b, a) is False


def test_pareto_front_keeps_extremes():
    pts = [
        ParetoPoint(objectives=(2.0, 1.0), meta={"id": "A"}),  # max axis 0
        ParetoPoint(objectives=(1.0, 2.0), meta={"id": "B"}),  # max axis 1
        ParetoPoint(objectives=(0.5, 0.5), meta={"id": "C"}),  # dominated by both
        ParetoPoint(objectives=(1.5, 1.5), meta={"id": "D"}),  # not dominated by A or B
    ]
    front = pareto_front(pts)
    ids = {p.meta["id"] for p in front}
    assert "A" in ids
    assert "B" in ids
    assert "D" in ids
    assert "C" not in ids


def test_pareto_front_drops_clearly_dominated():
    pts = [
        ParetoPoint(objectives=(5.0, 5.0), meta={"id": "best"}),
        ParetoPoint(objectives=(2.0, 2.0), meta={"id": "mid"}),
        ParetoPoint(objectives=(1.0, 1.0), meta={"id": "low"}),
    ]
    front = pareto_front(pts)
    assert {p.meta["id"] for p in front} == {"best"}


# ---------------------------------------------------------------------
# Constraint / constrained_best
# ---------------------------------------------------------------------

def _cell(tag, **metrics):
    return {"tag": tag, "metrics": metrics}


def test_constraint_passes_and_fails():
    c = Constraint("speaker_sim", ">=", 0.5)
    assert c.passes({"speaker_sim": 0.5}) is True
    assert c.passes({"speaker_sim": 0.7}) is True
    assert c.passes({"speaker_sim": 0.4}) is False
    assert c.passes({}) is False


def test_constrained_best_picks_min_under_constraint():
    cells = [
        _cell("A", vad_dist=0.10, speaker_sim=0.20),  # violates spk >= 0.30
        _cell("B", vad_dist=0.20, speaker_sim=0.40),  # passes
        _cell("C", vad_dist=0.30, speaker_sim=0.50),  # passes
    ]
    out = constrained_best(
        cells, objective="vad_dist", objective_direction="min",
        constraints=[Constraint("speaker_sim", ">=", 0.30)],
    )
    assert out["tag"] == "B"  # lower vad among passing


def test_constrained_best_returns_none_when_no_pass():
    cells = [
        _cell("A", vad_dist=0.10, speaker_sim=0.10),
        _cell("B", vad_dist=0.20, speaker_sim=0.20),
    ]
    out = constrained_best(
        cells, objective="vad_dist", objective_direction="min",
        constraints=[Constraint("speaker_sim", ">=", 0.50)],
    )
    assert out is None


def test_constrained_best_max_direction():
    cells = [
        _cell("A", e2v_cos=0.50, speaker_sim=0.40),
        _cell("B", e2v_cos=0.80, speaker_sim=0.60),
        _cell("C", e2v_cos=0.95, speaker_sim=0.10),  # violates
    ]
    out = constrained_best(
        cells, objective="e2v_cos", objective_direction="max",
        constraints=[Constraint("speaker_sim", ">=", 0.30)],
    )
    assert out["tag"] == "B"


# ---------------------------------------------------------------------
# make_objectives
# ---------------------------------------------------------------------

def test_make_objectives_signs():
    metrics = {"vad_dist": 0.20, "speaker_sim": 0.70}
    obj = make_objectives(metrics, [("vad_dist", "min"), ("speaker_sim", "max")])
    # min is negated so higher-is-better convention holds across the vector
    assert obj == (-0.20, 0.70)
