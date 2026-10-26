"""scripts/fetch_samples.py against a fake GitHub API."""
import importlib.util
import io
import zipfile

import pytest

spec = importlib.util.spec_from_file_location("fetch_samples", "scripts/fetch_samples.py")
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)


class R:
    def __init__(self, body=None, content=b"", status=200):
        self.body, self.content, self.status_code = body, content, status
    def json(self): return self.body
    def raise_for_status(self): pass


def fake_github(monkeypatch, assets, files):
    def get(url, headers, timeout):
        if "/releases/tags/" in url:
            return R({"assets": [{"name": n, "browser_download_url": "dl/" + n, "size": 10} for n in assets]})
        return R(content=files[url[3:]])
    monkeypatch.setattr(fs.requests, "get", get)


def test_per_file_assets_and_zip(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("samples/clip02_lantern_table/balance.exr", b"B")
    files = {"clip01_car_street_sun__expand_hdr.exr": b"E", "more.zip": buf.getvalue(), "notes.txt": b"x"}
    fake_github(monkeypatch, list(files), files)
    assert fs.fetch(dest=tmp_path) == 2
    assert (tmp_path / "clip01_car_street_sun" / "expand_hdr.exr").read_bytes() == b"E"
    assert (tmp_path / "clip02_lantern_table" / "balance.exr").read_bytes() == b"B"
    assert fs.fetch(dest=tmp_path) == 1  # the per-file asset is skipped once you have it


def test_zip_cannot_escape_samples(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../evil.txt", b"x")
    fake_github(monkeypatch, ["bad.zip"], {"bad.zip": buf.getvalue()})
    with pytest.raises(ValueError):
        fs.fetch(dest=tmp_path / "samples")
