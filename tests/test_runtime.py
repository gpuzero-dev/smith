"""The heavy part of a GPU node's Python, fetched at start (runtime.py).

What these hold it to: a file arrives whole and checked whatever the source
does - drops a piece, ignores ranges, lacks the file, serves other bytes -
and the mirror is asked first and the manifest's own address after it.
"""

import hashlib
import http.server
import json
import os
import threading

import pytest

import runtime


class _Files(http.server.BaseHTTPRequestHandler):
    """A server of a directory that takes range requests, and can be told to
    misbehave: ``drops`` breaks the first answer for that many pieces,
    ``ranges`` False answers every range with the whole file."""

    root = ""
    ranges = True
    drops = 0
    asked: list = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        type(self).asked.append(self.path)
        path = os.path.join(self.root, self.path.lstrip("/").split("/")[-1])
        if not os.path.exists(path):
            self.send_error(404)
            return
        data = open(path, "rb").read()
        wanted = self.headers.get("Range")
        if not wanted or not self.ranges:
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        first, last = (int(n) for n in wanted.split("=")[1].split("-"))
        part = data[first:last + 1]
        self.send_response(206)
        self.send_header("Content-Range", "bytes %d-%d/%d" % (first, last, len(data)))
        self.send_header("Content-Length", str(len(part)))
        self.end_headers()
        if type(self).drops and first > 0:
            type(self).drops -= 1
            self.wfile.write(part[: len(part) // 2])
            self.close_connection = True
            return
        self.wfile.write(part)


@pytest.fixture
def serve(tmp_path, monkeypatch):
    """A server per directory: ``serve(name)`` gives its base address."""
    servers = []
    monkeypatch.setattr(runtime, "PIECE", 1000)
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)

    def start(name, **behaviour):
        root = tmp_path / name
        root.mkdir(exist_ok=True)
        handler = type("Handler", (_Files,), {"root": str(root), "asked": [], **behaviour})
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return "http://127.0.0.1:%d" % server.server_port, root, handler

    yield start
    for server in servers:
        server.shutdown()


def _wheel(root, name, size):
    data = os.urandom(size)
    (root / name).write_bytes(data)
    return {"name": name, "url": "", "sha256": hashlib.sha256(data).hexdigest()}


def test_files_arrive_in_pieces_and_checked(serve, tmp_path):
    base, root, _ = serve("index")
    entries = [_wheel(root, "torch-2.11.0-cp312-linux_x86_64.whl", 5500), _wheel(root, "small-1.0-py3-none-any.whl", 10)]
    for entry in entries:
        entry["url"] = base + "/" + entry["name"]
    done = runtime.fetch({"version": 1, "files": entries}, str(tmp_path / "out"))
    assert done["files"] == 2 and done["bytes"] == 5510
    for entry in entries:
        assert (tmp_path / "out" / entry["name"]).read_bytes() == (root / entry["name"]).read_bytes()


def test_a_dropped_piece_is_fetched_again_on_its_own(serve, tmp_path):
    base, root, handler = serve("index", drops=2)
    entry = _wheel(root, "big-1.0-py3-none-any.whl", 4500)
    entry["url"] = base + "/" + entry["name"]
    runtime.fetch({"version": 1, "files": [entry]}, str(tmp_path / "out"))
    assert (tmp_path / "out" / entry["name"]).read_bytes() == (root / entry["name"]).read_bytes()
    # Five pieces, two of them asked twice; never the whole file again.
    assert handler.drops == 0
    assert len([path for path in handler.asked if path.endswith(entry["name"])]) == 1 + 5 + 2


def test_a_source_without_ranges_is_fetched_whole(serve, tmp_path):
    base, root, _ = serve("plain", ranges=False)
    entry = _wheel(root, "plain-1.0-py3-none-any.whl", 3000)
    entry["url"] = base + "/" + entry["name"]
    runtime.fetch({"version": 1, "files": [entry]}, str(tmp_path / "out"))
    assert (tmp_path / "out" / entry["name"]).read_bytes() == (root / entry["name"]).read_bytes()


def test_the_mirror_first_and_the_index_for_what_it_lacks(serve, tmp_path):
    index, index_root, index_handler = serve("index")
    mirror, mirror_root, _ = serve("mirror")
    mirrored = _wheel(index_root, "torch-2.11.0-cp312-linux_x86_64.whl", 2000)
    (mirror_root / mirrored["name"]).write_bytes((index_root / mirrored["name"]).read_bytes())
    missing = _wheel(index_root, "new-1.0-py3-none-any.whl", 100)
    for entry in (mirrored, missing):
        entry["url"] = index + "/" + entry["name"]
    done = runtime.fetch({"version": 1, "files": [mirrored, missing]}, str(tmp_path / "out"), mirror=mirror)
    assert done["from"][mirrored["name"]].startswith(mirror)
    assert done["from"][missing["name"]].startswith(index)
    # The index was never asked for what the mirror had.
    assert not any(path.endswith(mirrored["name"]) for path in index_handler.asked)


def test_bytes_that_are_not_the_manifests_are_refused(serve, tmp_path):
    base, root, _ = serve("mirror")
    entry = _wheel(root, "torch-2.11.0-cp312-linux_x86_64.whl", 1500)
    entry["sha256"] = "0" * 64
    entry["url"] = base + "/" + entry["name"]
    with pytest.raises(OSError, match="could not be fetched"):
        runtime.fetch({"version": 1, "files": [entry]}, str(tmp_path / "out"))
    assert not (tmp_path / "out" / entry["name"]).exists()


def test_a_poisoned_mirror_falls_back_to_the_index(serve, tmp_path):
    index, index_root, _ = serve("index")
    mirror, mirror_root, _ = serve("mirror")
    entry = _wheel(index_root, "torch-2.11.0-cp312-linux_x86_64.whl", 1500)
    (mirror_root / entry["name"]).write_bytes(b"x" * 1500)
    entry["url"] = index + "/" + entry["name"]
    done = runtime.fetch({"version": 1, "files": [entry]}, str(tmp_path / "out"), mirror=mirror)
    assert done["from"][entry["name"]].startswith(index)


def test_lock_takes_each_wheel_with_its_hash_and_refuses_a_source_distribution(tmp_path):
    report = {"install": [
        {"download_info": {"url": "https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=ab",
                           "archive_info": {"hash": "sha256=ab", "hashes": {"sha256": "ab"}}}},
        {"download_info": {"url": "https://files.pythonhosted.org/x/sympy-1.14.0-py3-none-any.whl",
                           "archive_info": {"hash": "sha256=cd"}}},
    ]}
    (tmp_path / "report.json").write_text(json.dumps(report))
    manifest = runtime.lock(str(tmp_path / "report.json"), str(tmp_path / "manifest.json"))
    assert manifest["files"] == [
        {"name": "sympy-1.14.0-py3-none-any.whl", "url": "https://files.pythonhosted.org/x/sympy-1.14.0-py3-none-any.whl", "sha256": "cd"},
        {"name": "torch-2.11.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl",
         "url": "https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl", "sha256": "ab"},
    ]
    report["install"].append({"download_info": {"url": "https://x/flash_attn-2.8.tar.gz", "archive_info": {"hash": "sha256=ef"}}})
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(SystemExit, match="not a wheel"):
        runtime.lock(str(tmp_path / "report.json"), str(tmp_path / "manifest.json"))
