"""Small Nightshift (Meterless) client for Latent Raw.

- chat():   text questions to Nemotron Lightning (app name "latent-raw")
- vision(): key-frame questions via the jobs path (app name "latent-raw-vision"),
            because images on /chat/completions are refused ("multimodal
            processing is not enabled", brief section 04)
- every call appends one JSON line with the meterless receipt to logs/nightshift.jsonl

Env: METERLESS_BASE_URL (default http://127.0.0.1:8088/v1), METERLESS_RUNTIME_TOKEN
"""
import base64, io, json, os, re, time, uuid, zipfile
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
    job_id, deadline = job["id"], time.time() + timeout
    while job.get("status") != "completed":
        if job.get("status") in ("failed", "errored", "error", "cancelled") or time.time() > deadline:
            _log("vision", {"question": question, "image": image_path}, None, job, step)
            raise RuntimeError(f"job {job_id} ended as {job.get('status')}")
        time.sleep(poll_every)
        job = requests.get(f"{BASE}/jobs/{job_id}", headers=h, timeout=30).json()
    answer, model = _answer_from_artifacts(job_id, h)
    _log("vision", {"question": question, "image": image_path, "job_id": job_id},
         answer, _receipt(job) or {"lanes_resolved": job.get("lanes_resolved"), "model": model}, step)
    job["answer_model"] = model
    return answer, job


def _answer_from_artifacts(job_id, h):
    """GET /jobs/<id>/artifacts: each attempt has "summary" (the answer) and "model".
    Falls back to the zip export if the artifacts list is empty."""
    r = requests.get(f"{BASE}/jobs/{job_id}/artifacts", headers=h, timeout=60)
    if r.ok:
        body = r.json()
        items = body.get("data", body) if isinstance(body, dict) else body
        items = items if isinstance(items, list) else [items]
        attempts = [a for a in items if isinstance(a, dict) and a.get("summary")]
        if attempts:
            return attempts[-1]["summary"], attempts[-1].get("model")
    export = requests.get(f"{BASE}/jobs/{job_id}/export", headers=h, timeout=60)
    return _answer_from_export(export.content), None


def _answer_from_export(zip_bytes):
    """The export is a zip; the answer text lives in data.js. The exact layout is not
    documented, so look for a JSON object inside data.js and fall back to the raw file."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        name = next(n for n in z.namelist() if n.endswith("data.js"))
        raw = z.read(name).decode("utf-8", "replace")
    m = re.search(r"[\[{].*[\]}]", raw, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            texts = _find_texts(data)
            if texts:
                return texts[-1]
        except json.JSONDecodeError:
            pass
    return raw


def _find_texts(obj, out=None):
    """Collect assistant text from whatever nesting data.js uses."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        if obj.get("role") == "assistant" and isinstance(obj.get("content"), str):
            out.append(obj["content"])
        for k in ("text", "answer", "output"):
            if isinstance(obj.get(k), str):
                out.append(obj[k])
        for v in obj.values():
            _find_texts(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _find_texts(v, out)
    return out


if __name__ == "__main__":
    print(chat([{"role": "user", "content": "Say hello in five words."}])[0])
