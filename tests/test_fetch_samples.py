"""scripts/fetch_samples.py against a fake Hugging Face API."""
import importlib.util

import pytest

spec = importlib.util.spec_from_file_location("fetch_samples", "scripts/fetch_samples.py")
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)


class R:
    def __init__(self, body=None, content=b"", links=None):
        self.body, self.content, self.links, self.status_code = body, content, links or {}, 200
    def json(self): return self.body
    def raise_for_status(self): pass


def fake_hf(monkeypatch, pages, files):
    """pages: list of tree listings (to test paging); files: path -> bytes."""
    def get(url, headers, timeout):
        if "/api/datasets/" in url:
            i = int(url.split("page=")[1]) if "page=" in url else 0
            links = {"next": {"url": f"{url.split('&page=')[0]}&page={i + 1}"}} if i + 1 < len(pages) else {}
            return R([{"type": "file", "path": p} for p in pages[i]], links=links)
        return R(content=files[url.split("/resolve/main/")[1]])
    monkeypatch.setattr(fs.requests, "get", get)


def test_downloads_clip_folders_across_pages(tmp_path, monkeypatch):
    files = {"README.md": b"r", ".gitattributes": b"g", "tools/finish.py": b"t",
             "clip01_car_street_sun/clip.json": b"{}",
             "clip01_car_street_sun/expand_hdr.exr": b"E",
             "samples/clip02_lantern_table/clip.json": b"{}",
             "samples/clip02_lantern_table/masks/lantern_flame.png": b"M"}
    fake_hf(monkeypatch, [["README.md", ".gitattributes", "tools/finish.py", "clip01_car_street_sun/clip.json",
                           "clip01_car_street_sun/expand_hdr.exr"],
                          ["samples/clip02_lantern_table/clip.json",
                           "samples/clip02_lantern_table/masks/lantern_flame.png"]], files)
    assert fs.fetch(dest=tmp_path) == 4
    assert not (tmp_path / "tools").exists()
    assert (tmp_path / "clip01_car_street_sun" / "expand_hdr.exr").read_bytes() == b"E"
    assert (tmp_path / "clip02_lantern_table" / "masks" / "lantern_flame.png").read_bytes() == b"M"
    assert not (tmp_path / "README.md").exists()
    assert fs.fetch(dest=tmp_path) == 0  # already have them


def test_path_cannot_escape_samples(tmp_path, monkeypatch):
    fake_hf(monkeypatch, [["clip/clip.json", "clip/../../evil.txt"]], {"clip/clip.json": b"{}", "clip/../../evil.txt": b"x"})
    with pytest.raises(ValueError):
        fs.fetch(dest=tmp_path / "samples")
