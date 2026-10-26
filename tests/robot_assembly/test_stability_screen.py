import numpy as np

import castle_reasoning.data.verification.stability as stability


def tower_plan(fixed=()):
    """Three stacked cubes on the floor; parking on the floor, then stacked."""
    plan = {
        "target_id": "T",
        "sizes": [[0.02] * 3] * 3,
        "target": [[0.4, 0.0, 0.01], [0.4, 0.0, 0.03], [0.4, 0.0, 0.05]],
        "parking": [[0.3, 0.2, 0.01], [0.3, 0.1, 0.01], [0.3, 0.1, 0.03]],
        "supports": [[0, -1], [1, 0], [2, 1]],
        "parking_supports": [[2, -1], [1, 2], [0, -1]],
        "removal_order": [2, 1, 0],
    }
    if fixed:
        plan["fixed_parts"] = list(fixed)
    return plan


def test_states_follow_the_removal_order_with_matching_supports():
    plan = tower_plan()
    plan["parking"] = [[0.3, 0.2, 0.01], [0.3, 0.1, 0.03], [0.3, 0.1, 0.01]]
    states = stability.configurations(plan)
    assert [s["kind"] for s in states] == ["target", "partial", "partial", "parked"]
    assert [s["last_removed"] for s in states] == [None, 2, 1, 0]
    np.testing.assert_allclose(states[0]["goals"], plan["target"])
    np.testing.assert_allclose(states[1]["goals"][2], plan["parking"][2])
    np.testing.assert_allclose(states[1]["goals"][:2], np.asarray(plan["target"])[:2])
    assert states[0]["supports"] == plan["supports"]
    # After removing 2, only supports among standing parts and parked parts remain.
    assert states[1]["supports"] == [[0, -1], [1, 0], [2, -1]]
    assert states[2]["supports"] == [[0, -1], [2, -1], [1, 2]]
    assert states[3]["supports"] == plan["parking_supports"]


def test_fixed_foundation_never_moves():
    states = stability.configurations(tower_plan(fixed=[0]))
    assert [s["removed"] for s in states] == [0, 1, 2]
    assert states[-1]["kind"] == "parked"
    np.testing.assert_allclose(states[-1]["goals"][0], tower_plan()["target"][0])


def test_initial_state_places_parts_upright_and_keeps_padding():
    home = np.arange(14 + 4 * 7, dtype=float)
    goals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    qpos = stability.initial_qpos(home, goals)
    poses = qpos[14:].reshape(-1, 7)
    np.testing.assert_allclose(poses[:3, :3], goals)
    np.testing.assert_allclose(poses[:3, 3:], [[1, 0, 0, 0]] * 3)
    np.testing.assert_allclose(poses[3], home[14 + 21 :])
    np.testing.assert_allclose(qpos[:14], home[:14])


def test_summary_reports_the_first_failing_state():
    plan = tower_plan()
    states = stability.configurations(plan)
    ok = {
        "position_error_m": 0.0,
        "orientation_error_deg": 0.0,
        "settled": True,
        "passed": True,
    }
    tipped = dict(ok, orientation_error_deg=7.0, passed=False)
    summary = stability.summarize(plan, states, [ok, tipped, ok, ok])
    assert summary["passed"] is False
    assert summary["first_failure_state"] == 1
    assert summary["first_failure_kind"] == "partial"
    assert summary["last_removed"] == 2
    assert summary["failure"] == ["orientation"]
    clean = stability.summarize(plan, states, [ok] * 4)
    assert clean["passed"] is True and "failure" not in clean


def test_only_parked_parts_get_horizontal_placement_error():
    plan = tower_plan()
    state = stability.configurations(plan)[2]
    assert state["parked"] == [1, 2]
    exact = state["goals"].copy()
    moved = stability.placed(state, np.random.default_rng(0), jitter=0.001)
    offset = moved["goals"] - exact
    np.testing.assert_array_equal(offset[0], 0)
    np.testing.assert_array_equal(offset[:, 2], 0)
    assert np.abs(offset).max() <= 0.001 and np.abs(offset[[1, 2], :2]).min() > 0
    np.testing.assert_array_equal(state["goals"], exact)
    assert moved["kind"] == state["kind"] and moved["supports"] == state["supports"]
    again = stability.placed(state, np.random.default_rng(0), jitter=0.001)
    np.testing.assert_array_equal(again["goals"], moved["goals"])


def window(passed=True, contacts=None, settled=None):
    result = {
        "position_error_m": 0.0,
        "orientation_error_deg": 0.0,
        "settled": passed if settled is None else settled,
        "passed": passed and contacts is not False,
    }
    if contacts is not None:
        result["contacts"] = {"passed": contacts, "failures": []}
    return result


def test_verdict_uses_the_settle_window_and_rejects_creeping_contacts():
    verdict = stability.verdict
    assert not verdict("partial", window(False), window())["passed"]
    # A partial structure must stay within the scene limits and come to rest;
    # momentary motion in the settle window is not checked by the collector.
    assert verdict("partial", window(settled=False), window())["passed"]
    for creep in (window(False), window(settled=False)):
        failed = verdict("partial", window(), creep)
        assert not failed["passed"] and failed["window"] == "hold"
    # The settle stage may wait out residual motion of the target only.
    assert verdict("target", window(False, True), window(True, True))["passed"]
    assert not verdict("parked", window(False, True), window(True, True))["passed"]
    # It never waits out a missing support.
    failed = verdict("target", window(True, False), window(True, True))
    assert not failed["passed"] and failed["window"] == "settle"
    # A support that opens after the settle window is creep.
    for kind in ("target", "parked"):
        creep = verdict(kind, window(True, True), window(True, False))
        assert not creep["passed"] and creep["window"] == "hold"
        assert creep["contacts"]["passed"] is False


def test_a_state_fails_only_if_every_replicate_fails():
    ok = {"passed": True, "position_error_m": 0.0}
    bad = {"passed": False, "position_error_m": 0.01, "window": "settle"}
    flaky = stability.reproduced([bad, ok])
    assert flaky["passed"] and flaky["failed_replicates"] == 1
    assert flaky["position_error_m"] == 0.0
    real = stability.reproduced([bad, dict(bad, position_error_m=0.02)])
    assert not real["passed"] and real["failed_replicates"] == 2
    assert real["position_error_m"] == 0.01 and real["window"] == "settle"
