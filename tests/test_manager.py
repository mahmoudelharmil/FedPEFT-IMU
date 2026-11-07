"""Tests of the pipeline manager (no GPU, no training): plan, queue, log parsing, settings."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fedpeft.manager import plan as plan_mod  # noqa: E402
from fedpeft.manager import progress, state  # noqa: E402


@pytest.fixture(scope="module")
def plan():
    return plan_mod.build_plan({})


def test_plan_covers_every_dataset_and_stage_order(plan):
    assert plan.datasets()[0] == "ucihar"
    assert set(plan.datasets()) == {"ucihar", "wisdm", "harbox", "capture24"}
    for ds in plan.datasets():
        stages = [j.stage for j in plan.select(datasets=[ds], only_missing=False)]
        order = [plan_mod.STAGES.index(s) for s in stages]
        assert order == sorted(order), f"{ds}: jobs out of stage order"
        assert stages[-1] == "report"


def test_plan_has_no_duplicate_outputs(plan):
    seen = {}
    for j in plan.jobs.values():
        if j.stage == "report":
            continue
        for o in j.outputs:
            assert o not in seen, f"{o} planned by {seen[o]} and {j.id}"
            seen[o] = j.id


def test_ablation_jobs_match_the_study(plan):
    abl = [j for j in plan.jobs.values() if j.dataset == "ucihar" and j.stage == "ablation"]
    ranks = sorted(j.tag for j in abl if j.group == "rank")
    assert ranks == sorted(["r2", "r4", "r16"] * 2)            # two encoders; r = 8 is the main V3 run
    # UCI HAR's LIMU-BERT already uses LoRA on all layers: the placement ablation is the W_q/W_v variant
    tags = {(j.encoder, j.tag) for j in abl if j.group == "switches"}
    assert ("limu_bert", "loraqv") in tags and ("limu_bert", "loraall") not in tags
    assert not any(e == "ssl_wearables" and t.startswith("lora") for e, t in tags)
    rank_job = next(j for j in abl if j.group == "rank")
    assert rank_job.script == "06_run_ablations.py" and any(a.startswith("ablation.ranks=[") for a in rank_job.args)


def test_trials_only_repeat_methods_of_the_main_split(plan):
    # Capture-24 never ran Ditto-FT on its main split, so its trials do not either
    cap = [j for j in plan.jobs.values() if j.dataset == "capture24" and j.stage == "trials"]
    assert cap and not any(j.method == "ditto_ft" for j in cap)
    assert {j.seed for j in cap} == {1, 2}


def test_job_command_and_rerun_flag(plan):
    j = next(j for j in plan.jobs.values() if j.stage == "federated")
    cmd = j.command("python", rerun=True)
    assert cmd[1:3] == ["-u", str(ROOT / "scripts" / "05_run_federated.py")] and "--rerun" in cmd
    assert "--rerun" not in j.command("python")


def test_queue_enqueue_dedup_move_and_start_from(tmp_path, monkeypatch):
    s = state.empty_state()
    jobs = [{"id": f"job{i}", "label": str(i)} for i in range(5)]
    assert state.enqueue(s, jobs) == 5
    assert state.enqueue(s, jobs[:2]) == 0                      # already waiting
    keys = [it["key"] for it in s["items"]]
    state.move(s, keys[3], "top")
    assert s["items"][0]["key"] == keys[3]
    n = state.start_from(s, keys[2])                             # skip what is above job2
    assert n == 3 and [it["status"] for it in s["items"]][:3] == ["skipped"] * 3
    assert state.counts(s)["pending"] == 2


def test_state_files_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path)
    monkeypatch.setattr(state, "QUEUE_FILE", tmp_path / "queue.json")
    monkeypatch.setattr(state, "LOCK_FILE", tmp_path / "queue.lock")
    state.update(lambda s: state.enqueue(s, [{"id": "a"}]))
    assert json.loads((tmp_path / "queue.json").read_text())["items"][0]["id"] == "a"
    assert not (tmp_path / "queue.lock").exists()


FL_LOG = """\
22:00:00 | data | Split seed 0 (chronological): 21 clients (5,000 train / 1,000 test windows)
22:00:05 | fl.limu_bert.v5_proposed | LIMU-BERT | V5 · FedPEFT-IMU (proposed): 21 clients, 4/round, 100 rounds, lr=0.002, r_max=8
22:00:25 | fl.limu_bert.v5_proposed | round  10  loss 0.641  F1 0.7210  acc 0.7426  worst10 0.498  up 0.15 MB  (1.9s/round)
22:00:45 | fl.limu_bert.v5_proposed | round  20  loss 0.462  F1 0.8278  acc 0.8354  worst10 0.588  up 0.12 MB  (1.9s/round)
"""


def test_progress_federated(tmp_path):
    log = tmp_path / "j.log"
    log.write_text(FL_LOG, encoding="utf-8")
    p = progress.parse(log, "05_run_federated.py", 100)
    assert p["phase"] == "train" and p["step"] == 20 and p["total"] == 100
    assert p["info"]["clients"] == 21 and len(p["curve"]) == 2 and p["curve"][-1]["f1"] == pytest.approx(0.8278)
    log.write_text(FL_LOG + "22:05:00 | fl.limu_bert.v5_proposed | round 100  loss 0.18  F1 0.9006  acc 0.9021  worst10 0.756  up 0.14 MB  (1.9s/round)\n"
                   "22:05:09 | fl.limu_bert.v5_proposed | after personalisation: F1 0.9145 (was 0.9006), 9s\n"
                   "22:06:30 | fl.limu_bert.v5_proposed | FINAL F1 0.9145 acc 0.9144 client-acc 0.9105 worst10 0.756 | "
                   "up 0.038 MB/client/round | held-out zero-shot 0.842 | 3.2 min\n", encoding="utf-8")
    p = progress.parse(log, "05_run_federated.py", 100)
    assert p["phase"] == "saved" and p["final"]["zero_shot"] == pytest.approx(0.842)


def test_progress_centralized_and_error(tmp_path):
    log = tmp_path / "c.log"
    log.write_text(
        "10:00:00 | central.limu_bert.scratch | limu_bert/scratch: 5,884 train windows, 734 validation windows, 10 trainable parameters\n"
        "10:00:02 | central.limu_bert.scratch | epoch  1  loss 0.6251  val-F1 0.8404  (2s, 2771 win/s)\n"
        "Traceback (most recent call last):\n  File \"x.py\", line 1\nRuntimeError: CUDA out of memory\n", encoding="utf-8")
    p = progress.parse(log, "04_run_centralized.py", 25)
    assert p["phase"] == "train" and p["step"] == 1 and "out of memory" in p["error"]
