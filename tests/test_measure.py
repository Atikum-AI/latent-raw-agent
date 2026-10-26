"""Tests for measure/. Run from the repo root: pytest tests/test_measure.py
The database test runs only when EVIDENCE_DATABASE_URL is set."""
import json
import os

import numpy as np
import OpenEXR
import pytest
from PIL import Image

from measure import metrics, vision, store
from measure.__main__ import main


def make_shot(tmp_path, name="shot01", sky_value=255, exr_gain=1.0):
    """A 40x60 frame: top half 'sky' (clipped white), bottom half 'car' (grey-blue)."""
    src = np.zeros((40, 60, 3), np.uint8)
    src[:20] = sky_value
    src[20:] = (90, 100, 140)
    d = tmp_path / name
    (d / "masks").mkdir(parents=True)
    Image.fromarray(src).save(d / "frame.png")
    sky = np.zeros((40, 60), np.uint8); sky[:20] = 255
    car = np.zeros((40, 60), np.uint8); car[20:] = 255
    Image.fromarray(sky).save(d / "masks" / "sky.png")
    Image.fromarray(car).save(d / "masks" / "car.png")
    Image.fromarray(sky).save(d / "synth.png")  # the model synthesized the sky's range
    exr = metrics.srgb_to_linear(src.astype(np.float32) / 255) * exr_gain
    exr[:20] *= 4.0  # expanded highlights
    with OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION},
                      {"RGB": exr.astype(np.float32)}) as f:
        f.write(str(d / "frame.exr"))
    return d


def test_item_rows(tmp_path):
    d = make_shot(tmp_path)
    src = metrics.load_8bit(d / "frame.png")
    items = metrics.load_item_masks(d / "masks", src.shape[:2])
    synth = metrics.load_mask(d / "synth.png", src.shape[:2])
    exr = metrics.load_exr(d / "frame.exr")[1]
    rows = {r["item"]: r for r in metrics.item_rows("shot01", 1, src, items, synth, exr)}
    assert rows["sky"]["clipped_pct"] == 100.0
    assert rows["car"]["clipped_pct"] == 0.0
    assert rows["sky"]["synthesized_pct"] == 100.0
    assert rows["car"]["synthesized_pct"] == 0.0
    assert rows["car"]["b_over_g"] > 1.0      # blue-ish car
    assert rows["sky"]["r_over_g"] == 1.0     # white sky is neutral
    assert rows["sky"]["hdr_peak_lum"] > 3.9  # 4x expansion survived the EXR round trip
    assert rows["car"]["evidence_id"] == "item:shot01:f1:car"


def test_balance_rows_one_stop(tmp_path):
    a = make_shot(tmp_path, "shot01")
    b = make_shot(tmp_path, "shot02", exr_gain=2.0)  # same scene, one stop brighter
    rows = []
    for d, shot in ((a, "shot01"), (b, "shot02")):
        src = metrics.load_8bit(d / "frame.png")
        items = metrics.load_item_masks(d / "masks", src.shape[:2])
        exr = metrics.load_exr(d / "frame.exr")[1]
        rows.append(metrics.item_rows(shot, 1, src, items, np.zeros(src.shape[:2], bool), exr))
    bal = {r["item"]: r for r in metrics.balance_rows(rows[0], rows[1], "before")}
    assert bal["car"]["lum_gap_stops"] == pytest.approx(1.0, abs=1e-3)
    assert bal["car"]["r_over_g_gap"] == pytest.approx(0.0, abs=1e-3)
    assert bal["car"]["cites"] == ["item:shot01:f1:car", "item:shot02:f1:car"]


def test_cli_shot_and_compare(tmp_path, capsys):
    a = make_shot(tmp_path, "shot01")
    b = make_shot(tmp_path, "shot02", exr_gain=2.0)
    for d, shot in ((a, "shot01"), (b, "shot02")):
        main(["shot", "--shot", shot, "--frame", "1", "--src", str(d / "frame.png"),
              "--items", str(d / "masks"), "--exr", str(d / "frame.exr"),
              "--synth", str(d / "synth.png"), "--out", str(tmp_path / f"{shot}.json")])
    saved = json.loads((tmp_path / "shot01.json").read_text())
    assert len(saved["item_metrics"]) == 2 and saved["frame_range"][0]["synthesized_pct"] == 50.0
    main(["compare", str(tmp_path / "shot01.json"), str(tmp_path / "shot02.json"),
          "--stage", "before", "--out", str(tmp_path / "bal.json")])
    assert len(json.loads((tmp_path / "bal.json").read_text())["balance_checks"]) == 2


def test_vision_live_then_sample_mode(tmp_path, monkeypatch):
    calls = []

    def fake_vision(question, image_path, step=None):
        calls.append(question)
        return "a dark grey car, lower half", {"id": "job_1", "answer_model": "super-vl"}

    monkeypatch.setattr(vision.nightshift, "vision", fake_vision)
    cache = tmp_path / "vision_cache.json"
    monkeypatch.setattr(vision, "SAMPLE_MODE", False)
    live = vision.check_frame("shot01", 1, "frame.jpg", keys=("contents",), cache_path=cache)
    monkeypatch.setattr(vision, "SAMPLE_MODE", True)
    replay = vision.check_frame("shot01", 1, "frame.jpg", keys=("contents",), cache_path=cache)
    assert len(calls) == 1                     # sample mode made no model call
    assert live == replay
    assert replay[0]["evidence_id"] == "vision:shot01:f1:contents"


def test_nightshift_waits_for_the_winner(monkeypatch, tmp_path):
    from measure import nightshift

    class R:
        def __init__(self, body): self.body, self.ok = body, True
        def json(self): return self.body

    monkeypatch.setattr(nightshift, "LOG", str(tmp_path / "ns.jsonl"))
    posted, artifact_calls = {}, []

    def post(url, headers, json, timeout):
        posted.update(json); posted["header_app"] = headers["meterless-app"]
        return R({"id": "job_9", "status": "queued"})

    def get(url, headers, timeout):
        if url.endswith("/artifacts"):
            artifact_calls.append(1)
            if len(artifact_calls) == 1:  # job says completed, winner not filled in yet
                return R({"data": [{"summary": "not yet decided", "model": "nemotron-super-vl"}]})
            return R({"data": [{"summary": "loser", "model": "other"},
                               {"summary": "brightest area is the sky, clipped",
                                "model": "nemotron-super-vl", "winner": True}]})
        return R({"id": "job_9", "status": "completed", "lanes_resolved": ["race:super-vl"]})

    monkeypatch.setattr(nightshift.requests, "post", post)
    monkeypatch.setattr(nightshift.requests, "get", get)
    img = tmp_path / "f.jpg"; Image.new("RGB", (4, 4)).save(img)
    answer, job = nightshift.vision("Where is the brightest area?", str(img), poll_every=0)
    assert answer == "brightest area is the sky, clipped"
    assert len(artifact_calls) == 2
    assert job["answer_model"] == "nemotron-super-vl"
    assert posted["attempt_budget"] == {"level": "high"}
    assert posted["app"] == posted["header_app"] == "latent-raw-vision"
    assert json.loads((tmp_path / "ns.jsonl").read_text().splitlines()[0])["kind"] == "vision"


def test_nightshift_errored_job_stops_at_once(monkeypatch, tmp_path):
    from measure import nightshift

    class R:
        def __init__(self, body): self.body, self.ok = body, True
        def json(self): return self.body

    monkeypatch.setattr(nightshift, "LOG", str(tmp_path / "ns.jsonl"))
    monkeypatch.setattr(nightshift.requests, "post",
                        lambda url, headers, json, timeout: R({"id": "job_e", "status": "errored",
                                                               "error_code": "fleet_cannot_vision"}))
    img = tmp_path / "f.jpg"; Image.new("RGB", (4, 4)).save(img)
    with pytest.raises(RuntimeError, match="errored"):
        nightshift.vision("q", str(img), poll_every=0, timeout=999)


@pytest.mark.skipif(not os.environ.get("EVIDENCE_DATABASE_URL"), reason="no database")
def test_postgres_upsert_and_view(tmp_path):
    a = make_shot(tmp_path, "shot01")
    main(["shot", "--shot", "shot01", "--frame", "1", "--src", str(a / "frame.png"),
          "--items", str(a / "masks"), "--exr", str(a / "frame.exr"), "--db"])
    main(["shot", "--shot", "shot01", "--frame", "1", "--src", str(a / "frame.png"),
          "--items", str(a / "masks"), "--exr", str(a / "frame.exr"), "--db"])  # twice: upsert
    with store.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT subject, (facts->>'clipped_pct')::float FROM shot_evidence "
                    "WHERE shot_id = 'shot01' AND kind = 'item' ORDER BY subject")
        assert cur.fetchall() == [("car", 0.0), ("sky", 100.0)]  # one row each, not two


def test_rolled_off_highlights_count_as_clipped(tmp_path):
    """AI video rolls highlights off below 254; 252 must still count as clipped."""
    d = make_shot(tmp_path, sky_value=252)
    src = metrics.load_8bit(d / "frame.png")
    items = metrics.load_item_masks(d / "masks", src.shape[:2])
    rows = {r["item"]: r for r in metrics.item_rows("shot01", 1, src, items, np.zeros(src.shape[:2], bool))}
    assert rows["sky"]["clipped_pct"] == 100.0
    assert rows["car"]["clipped_pct"] == 0.0
