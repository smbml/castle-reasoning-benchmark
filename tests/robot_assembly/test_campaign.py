
import json
import subprocess
import sys
from pathlib import Path

import pytest

import castle_reasoning.data.assembly.campaign as campaign
import castle_reasoning.data.assembly.records as assembly_records
import castle_reasoning.data.assembly.run_campaign as assembly_run_campaign
from castle_reasoning.runtime.diagnostics import io_timing


def plan_entry(target_id, parts, timestep=0.002, fingerprint=None):
    return {
        "path": f"/plans/{target_id}.json",
        "sha256": "0" * 64,
        "target_id": target_id,
        "fingerprint": fingerprint or f"fp-{target_id}",
        "parts": parts,
        "fixed_parts": [],
        "physics_timestep": timestep,
    }


def test_describe_plan_uses_the_plate_timestep_limit(tmp_path):
    plain = {
        "target_id": "A",
        "target_fingerprint": "a",
        "sizes": [[1, 1, 1]] * 3,
        "edge_access": {},
    }
    plate = dict(
        plain,
        target_id="B",
        target_fingerprint="b",
        fixed_parts=[0],
        edge_access={"max_physics_timestep_s": 0.001},
    )
    for value in (plain, plate):
        io_timing.write_json(tmp_path / f"{value['target_id']}.json", value)
    a = campaign.describe_plan(tmp_path / "A.json")
    b = campaign.describe_plan(tmp_path / "B.json")
    assert (a["physics_timestep"], a["parts"], a["fixed_parts"]) == (0.002, 3, [])
    assert (b["physics_timestep"], b["fixed_parts"]) == (0.001, [0])


def test_shards_split_by_timestep_and_size_and_stay_bounded():
    plans = [plan_entry(f"S{i:03d}", 7 + i % 20) for i in range(100)]
    plans += [plan_entry(f"P{i:03d}", 9 + i % 5, timestep=0.001) for i in range(10)]
    shards = campaign.make_shards(plans, max_plans=30, part_buckets=[12, 18])
    assert sorted(p["target_id"] for s in shards for p in s["plans"]) == sorted(
        p["target_id"] for p in plans
    )
    assert all(len(s["plans"]) <= 30 for s in shards)
    for shard in shards:
        assert {p["physics_timestep"] for p in shard["plans"]} == {
            shard["physics_timestep"]
        }
        assert shard["max_parts"] == max(p["parts"] for p in shard["plans"])
    small = [s for s in shards if s["shard_id"].startswith("2ms-b0")]
    assert small and all(s["max_parts"] <= 12 for s in small)
    assert [s["index"] for s in shards] == list(range(len(shards)))
    costs = [
        sum(p["parts"] for p in s["plans"]) * 0.002 / s["physics_timestep"]
        for s in shards
    ]
    assert costs == sorted(costs, reverse=True)


def test_small_buckets_join_the_next_smaller_bucket():
    plans = [plan_entry(f"S{i:03d}", 8) for i in range(20)]
    plans += [plan_entry("BIG1", 40), plan_entry("BIG2", 30)]
    shards = campaign.make_shards(
        plans, max_plans=48, part_buckets=[12, 26], min_plans=8
    )
    assert len(shards) == 1
    assert shards[0]["max_parts"] == 40 and len(shards[0]["plans"]) == 22


def test_buckets_merge_until_a_shard_fills_its_worlds(monkeypatch, tmp_path):
    plans = []
    for i, parts in enumerate([8] * 6 + [14] * 3 + [20] * 2):
        path = tmp_path / f"P{i}.json"
        io_timing.write_json(
            path,
            {
                "target_id": f"P{i}",
                "target_fingerprint": f"f{i}",
                "sizes": [[1, 1, 1]] * parts,
                "edge_access": {},
            },
        )
        plans.append(str(path))
    argv = ["campaign", "--plans", *plans, "--output", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", [*argv, "--worlds", "8", "--repeats", "2"])
    spec = assembly_run_campaign.manifest(assembly_run_campaign.parse())
    assert spec["min_plans_per_shard"] == 4
    assert [len(s["plans"]) for s in spec["shards"]] == [5, 6]
    monkeypatch.setattr(sys, "argv", [*argv, "--min-plans-per-shard", "1"])
    assert len(assembly_run_campaign.manifest(assembly_run_campaign.parse())["shards"]) == 3


def test_campaign_refuses_duplicate_targets():
    with pytest.raises(ValueError, match="ids must be unique"):
        campaign.make_shards(
            [plan_entry("A", 8), plan_entry("A", 9, fingerprint="x")],
            max_plans=4,
            min_plans=1,
            part_buckets=[],
        )
    with pytest.raises(ValueError, match="more than once"):
        campaign.make_shards(
            [plan_entry("A", 8, fingerprint="x"), plan_entry("B", 8, fingerprint="x")],
            max_plans=4,
            min_plans=1,
            part_buckets=[],
        )


def test_run_seeds_never_overlap():
    starts = {
        campaign.run_seed(0, shard, run) for shard in range(20) for run in range(5)
    }
    assert len(starts) == 100
    assert (
        min(b - a for a, b in zip(sorted(starts), sorted(starts)[1:]))
        == campaign.RUN_SEED_STRIDE
    )
    with pytest.raises(ValueError):
        campaign.run_seed(0, 0, campaign.RUNS_PER_SHARD)


def test_seeds_must_fit_the_engine(monkeypatch, tmp_path):
    assert campaign.seeds_fit(1_000_000_000, 7)
    assert not campaign.seeds_fit(1_000_000_000, 13)
    plans = write_plans(tmp_path, 1)
    with pytest.raises(SystemExit, match="above"):
        run_campaign(
            monkeypatch, tmp_path, plans, lambda *_: True, "--seed", "2000000000"
        )


def write_summary(output, records, completed=True):
    output.mkdir(parents=True)
    summary = {"attempts": records}
    if completed:
        summary["completed"] = True
    io_timing.write_json(output / "summary.json", summary)


def test_deficits_count_accepted_attempts_of_every_run(tmp_path):
    shard = {"plans": [plan_entry("A", 8), plan_entry("B", 8), plan_entry("C", 8)]}
    write_summary(
        tmp_path / "r0",
        [
            {"attempt": 0, "target_id": "A", "accepted": True, "slot": 0},
            {"attempt": 1, "target_id": "B", "accepted": False, "slot": 1},
            {"attempt": 2, "target_id": "A", "accepted": True, "slot": 0},
        ],
        completed=False,
    )
    write_summary(
        tmp_path / "r1", [{"attempt": 0, "target_id": "B", "accepted": True, "slot": 0}]
    )
    runs = [
        {"output": str(tmp_path / "r0")},
        {"output": str(tmp_path / "r1")},
        {"output": str(tmp_path / "never-started")},
    ]
    assert {
        k: [p["target_id"] for p in v]
        for k, v in campaign.deficits(shard, runs, 2).items()
    } == {2: ["C"], 1: ["B"]}


def fake_collector(accept):
    """Stand-in for the collector: admits the attempts `accept(target, repeat, seed)` admits."""

    def run(command, stdout, stderr, check):
        def value(flag):
            return command[command.index(flag) + 1]

        plans = command[command.index("--plans") + 1 : command.index("--output")]
        seed = int(value("--seed"))
        records = []
        for repeat in range(int(value("--attempts-per-target"))):
            for path in plans:
                target = json.loads(Path(path).read_text())["target_id"]
                accepted = accept(target, repeat, seed)
                records.append(
                    {
                        "attempt": len(records),
                        "target_id": target,
                        "repeat": repeat,
                        "slot": 0,
                        "status": "accepted" if accepted else "replay",
                        "accepted": accepted,
                        "action_targets": 100,
                    }
                )
        write_summary(Path(value("--output")), records)
        return subprocess.CompletedProcess(command, 0)

    return run


def write_plans(directory, count):
    paths = []
    for i in range(count):
        path = directory / f"T{i}.json"
        io_timing.write_json(
            path,
            {
                "target_id": f"T{i}",
                "target_fingerprint": f"f{i}",
                "sizes": [[1, 1, 1]] * (8 + i),
                "edge_access": {},
            },
        )
        paths.append(str(path))
    return paths


def read_runs(directory):
    return json.loads((directory / "runs.json").read_text())


def run_campaign(monkeypatch, tmp_path, plans, accept, *extra, launched=None):
    collector = fake_collector(accept)

    def run(command, **kwargs):
        if launched is not None:
            launched.append(Path(command[command.index("--output") + 1]))
        return collector(command, **kwargs)

    monkeypatch.setattr(assembly_run_campaign.subprocess, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign",
            "--plans",
            *plans,
            "--output",
            str(tmp_path / "out"),
            "--repeats",
            "2",
            *extra,
        ],
    )
    assembly_run_campaign.main()
    return json.loads((tmp_path / "out" / "index.json").read_text())


def test_campaign_tops_up_failed_targets_and_resumes(monkeypatch, tmp_path):
    plans = write_plans(tmp_path, 3)
    failed_once = set()

    def accept(target, repeat, seed):
        if target == "T1" and target not in failed_once:
            failed_once.add(target)
            return False
        return True

    index = run_campaign(monkeypatch, tmp_path, plans, accept)
    assert index["accepted_demonstrations"] == 6
    assert index["targets_complete"] == 3
    runs = read_runs(tmp_path / "out" / "shards" / "2ms-b0-000")
    topups = read_runs(tmp_path / "out" / "topups" / "topup-2ms")
    assert [
        (r["round"], r["attempts_per_target"], r["targets"], r["worlds"])
        for r in runs + topups
    ] == [
        (0, 2, ["T2", "T1", "T0"], 6),
        (1, 1, ["T1"], 1),
    ]
    assert len({r["seed"] for r in runs + topups}) == 2
    seeds = [d["seed"] for d in index["demonstrations"]]
    assert len(seeds) == len(set(seeds))
    # Rerunning the same command finds nothing missing and launches nothing.
    again = run_campaign(
        monkeypatch, tmp_path, plans, lambda *_: pytest.fail("relaunched")
    )
    assert again["accepted_demonstrations"] == 6


def failing(counts):
    """Fails each target's first `counts[target]` attempts, then accepts."""
    remaining = dict(counts)

    def accept(target, repeat, seed):
        if remaining.get(target, 0) > 0:
            remaining[target] -= 1
            return False
        return True

    return accept


def test_topups_pool_every_shard_of_a_timestep_after_all_first_rounds(
    monkeypatch, tmp_path
):
    plans = write_plans(tmp_path, 6)
    plate = tmp_path / "P.json"
    io_timing.write_json(
        plate,
        {
            "target_id": "P",
            "target_fingerprint": "fp",
            "sizes": [[1, 1, 1]] * 8,
            "edge_access": {"max_physics_timestep_s": 0.001},
        },
    )
    launched = []
    index = run_campaign(
        monkeypatch,
        tmp_path,
        [*plans, str(plate)],
        failing({"T5": 2, "T0": 1, "P": 1}),
        "--part-buckets",
        "10",
        "--min-plans-per-shard",
        "1",
        launched=launched,
    )
    out = tmp_path / "out"
    assert [path.relative_to(out).as_posix() for path in launched] == [
        "shards/2ms-b1-000/run00",
        "shards/2ms-b0-000/run00",
        "shards/1ms-b0-000/run00",
        "topups/topup-2ms/run00",
        "topups/topup-1ms/run00",
    ]
    # One run holds the short targets of both 2 ms shards, each with the
    # largest shortfall's attempts.
    pooled = read_runs(out / "topups" / "topup-2ms")
    assert [
        (r["round"], r["attempts_per_target"], r["targets"], r["worlds"])
        for r in pooled
    ] == [(1, 2, ["T5", "T0"], 4)]
    assert read_runs(out / "topups" / "topup-1ms")[0]["targets"] == ["P"]
    seeds = [
        r["seed"]
        for directory in launched
        for r in read_runs(directory.parent)
        if r["output"] == str(directory)
    ]
    assert len(set(seeds)) == 5
    assert index["targets_complete"] == 7
    # T0 was short by one and got two top-up attempts.
    assert index["accepted_demonstrations"] == 15
    groups = {
        (d["target_id"], d["run_group"], d["shard_id"])
        for d in index["demonstrations"]
        if d["target_id"] in ("T0", "P")
    }
    assert groups == {
        ("T0", "2ms-b0-000", "2ms-b0-000"),
        ("T0", "topup-2ms", "2ms-b0-000"),
        ("P", "1ms-b0-000", "1ms-b0-000"),
        ("P", "topup-1ms", "1ms-b0-000"),
    }


def test_resume_keeps_a_run_whose_collector_finished(monkeypatch, tmp_path):
    plans = write_plans(tmp_path, 2)
    run_campaign(monkeypatch, tmp_path, plans, lambda *_: True)
    shard = tmp_path / "out" / "shards" / "2ms-b0-000"
    runs = read_runs(shard)
    runs[0]["status"] = "running"
    io_timing.write_json(shard / "runs.json", runs)
    run_campaign(monkeypatch, tmp_path, plans, lambda *_: pytest.fail("relaunched"))
    assert read_runs(shard)[0]["status"] == "completed"
    # A first round the collector never finished is interrupted and runs again
    # for the targets it left short.
    summary = json.loads((shard / "run00" / "summary.json").read_text())
    summary.pop("completed")
    summary["attempts"] = summary["attempts"][:3]
    io_timing.write_json(shard / "run00" / "summary.json", summary)
    runs[0]["status"] = "running"
    io_timing.write_json(shard / "runs.json", runs)
    index = run_campaign(monkeypatch, tmp_path, plans, lambda *_: True)
    assert [
        (r["status"], r["targets"], r["attempts_per_target"]) for r in read_runs(shard)
    ] == [("interrupted", ["T1", "T0"], 2), ("completed", ["T0"], 1)]
    assert index["targets_complete"] == 2


def test_an_interrupted_topup_round_runs_again(monkeypatch, tmp_path):
    plans = write_plans(tmp_path, 2)
    run_campaign(monkeypatch, tmp_path, plans, failing({"T0": 2}), "--max-rounds", "2")
    group = tmp_path / "out" / "topups" / "topup-2ms"
    runs = read_runs(group)
    assert [(r["round"], r["targets"]) for r in runs] == [(1, ["T0"])]
    io_timing.write_json(group / "run00" / "summary.json", {"attempts": []})
    runs[0]["status"] = "running"
    io_timing.write_json(group / "runs.json", runs)
    index = run_campaign(
        monkeypatch, tmp_path, plans, lambda *_: True, "--max-rounds", "2"
    )
    assert [(r["round"], r["status"]) for r in read_runs(group)] == [
        (1, "interrupted"),
        (1, "completed"),
    ]
    assert index["targets_complete"] == 2


def test_campaign_refuses_a_changed_command(monkeypatch, tmp_path):
    plans = write_plans(tmp_path, 2)
    run_campaign(monkeypatch, tmp_path, plans, lambda *_: True)
    with pytest.raises(SystemExit, match="differs"):
        run_campaign(monkeypatch, tmp_path, plans, lambda *_: True, "--worlds", "7")


def test_collector_flags_follow_a_separator(monkeypatch, tmp_path):
    plans = write_plans(tmp_path, 1)
    run_campaign(
        monkeypatch, tmp_path, plans, lambda *_: True, "--", "--solver", "newton"
    )
    spec = json.loads((tmp_path / "out" / "campaign.json").read_text())
    assert spec["collector_flags"] == ["--solver", "newton"]


def screen_result(path, passed, **failure):
    return {
        "target_id": json.loads(Path(path).read_text())["target_id"],
        "plan_sha256": assembly_records.sha256(Path(path)),
        "passed": passed,
        **failure,
    }


def test_screen_excludes_failed_plans_by_content_hash(monkeypatch, tmp_path):
    plans = write_plans(tmp_path, 3)
    summary = tmp_path / "screen.json"
    failure = {
        "first_failure_state": 0,
        "first_failure_kind": "target",
        "failure_window": "hold",
        "failure": ["contacts"],
        "missing_supports": ["missing_support_3_1"],
    }
    io_timing.write_json(
        summary,
        {
            "results": [
                screen_result(plans[0], True),
                screen_result(plans[1], False, **failure),
                screen_result(plans[2], True),
            ]
        },
    )
    index = run_campaign(
        monkeypatch, tmp_path, plans, lambda *_: True, "--screen", str(summary)
    )
    assert index["accepted_demonstrations"] == 4
    assert {d["target_id"] for d in index["demonstrations"]} == {"T0", "T2"}
    spec = json.loads((tmp_path / "out" / "campaign.json").read_text())
    [excluded] = spec["screen"]["excluded"]
    assert excluded["target_id"] == "T1" and excluded["failure_window"] == "hold"
    assert excluded["missing_supports"] == ["missing_support_3_1"]


def test_screens_of_disjoint_plans_combine(monkeypatch, tmp_path):
    plans = write_plans(tmp_path, 3)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    io_timing.write_json(first, {"results": [screen_result(plans[0], True)]})
    io_timing.write_json(
        second,
        {"results": [screen_result(plans[1], True), screen_result(plans[2], False)]},
    )
    index = run_campaign(
        monkeypatch,
        tmp_path,
        plans,
        lambda *_: True,
        "--screen",
        str(first),
        str(second),
    )
    assert {d["target_id"] for d in index["demonstrations"]} == {"T0", "T1"}
    spec = json.loads((tmp_path / "out" / "campaign.json").read_text())
    assert [s["path"] for s in spec["screen"]["summaries"]] == [
        str(first.resolve()),
        str(second.resolve()),
    ]
    assert [e["target_id"] for e in spec["screen"]["excluded"]] == ["T2"]


def test_screens_that_disagree_on_a_plan_are_refused(tmp_path):
    [path] = write_plans(tmp_path, 1)
    described = [campaign.describe_plan(Path(path))]
    results = [screen_result(path, True), screen_result(path, False)]
    with pytest.raises(ValueError, match="disagree on T0"):
        campaign.apply_screen(described, results)
    kept, _ = campaign.apply_screen(described, results[:1] * 2)
    assert kept == described


def test_screen_must_cover_the_exact_plan_contents(tmp_path):
    [path] = write_plans(tmp_path, 1)
    described = [campaign.describe_plan(Path(path))]
    result = screen_result(path, True)
    io_timing.write_json(Path(path), {**json.loads(Path(path).read_text()), "edited": 1})
    with pytest.raises(ValueError, match="absent from the stability screen"):
        campaign.apply_screen([campaign.describe_plan(Path(path))], [result])
    kept, excluded = campaign.apply_screen(described, [result])
    assert kept == described and excluded == []
