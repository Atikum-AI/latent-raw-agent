"""Small Nightshift (Meterless) client for Latent Raw.

- chat():   text questions to Nemotron Lightning (app name "latent-raw")
- vision(): key-frame questions via the jobs path (app name "latent-raw-vision"),
            because images on /chat/completions are refused ("multimodal
            processing is not enabled", brief section 04)
- every call appends one JSON line with the meterless receipt to logs/nightshift.jsonl

Env: METERLESS_BASE_URL (default http://127.0.0.1:8088/v1), METERLESS_RUNTIME_TOKEN
"""
import base64, json, os, time
from datetime import datetime, timezone
import requests

BASE = os.environ.get("METERLESS_BASE_URL", "http://127.0.0.1:8088/v1").rstrip("/")
TOKEN = os.environ.get("METERLESS_RUNTIME_TOKEN", "")
LOG = os.environ.get("NIGHTSHIFT_LOG", "logs/nightshift.jsonl")

TEXT_APP = "latent-raw"            # locked to the fast lane in Nightshift's Apps panel
VISION_APP = "latent-raw-vision"   # locked to the deep lane (Super VL)


def _headers(app):
    return {"Authorization": f"Bearer {TOKEN}", "meterless-app": app,
            "meterless-scope": "user", "Content-Type": "application/json"}


def _receipt(body):
    if not isinstance(body, dict):
        return None
    return body.get("meterless.receipt") or (body.get("meterless") or {}).get("receipt")


def _log(kind, request, response, receipt, step=None):
    os.makedirs(os.path.dirname(LOG) or ".", exist_ok=True)
    line = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, "step": step,
            "request": request, "response": response, "receipt": receipt}
    with open(LOG, "a") as f:
        f.write(json.dumps(line, default=str) + "\n")


def chat(messages, model="meterless/fast", max_tokens=800, step=None):
    """Text-only call. max_tokens must stay >= 600 or Nemotron returns empty_response."""
    req = {"model": model, "messages": messages, "max_tokens": max(max_tokens, 600)}
    r = requests.post(f"{BASE}/chat/completions", headers=_headers(TEXT_APP), json=req, timeout=120)
    r.raise_for_status()
    body = r.json()
    text = body["choices"][0]["message"]["content"]
    _log("chat", req, text, _receipt(body), step)
    return text, _receipt(body)


def vision(question, image_path, poll_every=2.0, timeout=180, step=None):
    """Ask the vision lane about one frame through POST /v1/jobs. Returns (answer, job)."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    # attempt_budget gives the answer room; without it replies get cut off.
    # With only the header, the job ran as app "default", so name the app in the body too.
    req = {"kind": "best_of_n", "n": 1, "attempt_budget": {"level": "high"}, "app": VISION_APP,
           "messages": [{"role": "user", "content": question}],
           "attachments": [{"kind": "image", "name": os.path.basename(image_path),
                            "url": f"data:{mime};base64,{b64}"}]}
    h = _headers(VISION_APP)
    job = requests.post(f"{BASE}/jobs", headers=h, json=req, timeout=60).json()
    if "id" not in job:
        # A bad token or a refused job comes back without an id; show Nightshift's reason.
        raise RuntimeError(f"Nightshift refused the job: {json.dumps(job)[:300]}")
    job_id, deadline = job["id"], time.time() + timeout
    while job.get("status") != "completed":
        if job.get("status") in ("failed", "errored", "error", "cancelled") or time.time() > deadline:
            _log("vision", {"question": question, "image": image_path}, None, job, step)
            raise RuntimeError(f"job {job_id} ended as {job.get('status')}")
        time.sleep(poll_every)
        job = requests.get(f"{BASE}/jobs/{job_id}", headers=h, timeout=30).json()
    answer, model = _winning_answer(job_id, h, poll_every, max(deadline, time.time() + 60))
    _log("vision", {"question": question, "image": image_path, "job_id": job_id},
         answer, _receipt(job) or {"lanes_resolved": job.get("lanes_resolved"), "model": model}, step)
    job["answer_model"] = model
    return answer, job


def _winning_answer(job_id, h, poll_every, deadline):
    """Poll GET /jobs/<id>/artifacts until the winning attempt has its summary.

    A job reports "completed" a few seconds before the winner's summary is filled in,
    so reading straight away returns the "not yet decided" placeholder."""
    while True:
        r = requests.get(f"{BASE}/jobs/{job_id}/artifacts", headers=h, timeout=60)
        if r.ok:
            body = r.json()
            items = body.get("data", body) if isinstance(body, dict) else body
            items = items if isinstance(items, list) else [items]
            for a in items:
                if isinstance(a, dict) and a.get("winner") and a.get("summary"):
                    return a["summary"], a.get("model")
        if time.time() > deadline:
            raise RuntimeError(f"job {job_id} completed but no winning answer arrived in time")
        time.sleep(poll_every)


if __name__ == "__main__":
    print(chat([{"role": "user", "content": "Say hello in five words."}])[0])
