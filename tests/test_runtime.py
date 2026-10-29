"""The heavy part of a GPU node's Python, fetched at start (runtime.py).

What these hold it to: a file arrives whole and checked whatever the source
does - drops a piece, ignores ranges, lacks the file, serves other bytes,
lets a signed URL expire - and the backend's cache is asked first, the
mirror next and the manifest's own address last.
"""

import hashlib
import http.server
import json
import os
import threading
import types
import urllib.parse

import pytest

import runtime
import smith


class _Files(http.server.BaseHTTPRequestHandler):
    """A server of a directory that takes range requests, and can be told to
    misbehave: ``drops`` breaks the first answer for that many pieces,
    ``ranges`` False answers every range with the whole file, and
    ``lifetime`` answers 403 - as a bucket does for an expired presigned URL -
    to a ``?sig=`` that has already been used that many times."""

    root = ""
    ranges = True
    drops = 0
    lifetime = 0
    asked: list = []
    uses: dict = {}
    lock = threading.Lock()

    def log_message(self, *args):
        pass

    def do_GET(self):
        type(self).asked.append(self.path)
        url = urllib.parse.urlsplit(self.path)
        signature = urllib.parse.parse_qs(url.query).get("sig", [""])[0]
        if self.lifetime and signature:
            with self.lock:
                used = self.uses.get(signature, 0)
                self.uses[signature] = used + 1
            if used >= self.lifetime:
                self.send_error(403, "Request has expired")
                return
        path = os.path.join(self.root, urllib.parse.unquote(url.path.lstrip("/").split("/")[-1]))
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

    #: PUTs answered 403 before one is taken, as a bucket answers a presigned
    #: PUT that expired while it waited.
    expire_puts = 0
    put: list = []

    def do_PUT(self):
        """A bucket's presigned PUT: the URL says the size it was signed for
        (``?size=``), and a body without that exact ``Content-Length`` - or
        sent in chunks, as urllib does a file without one - is refused."""
        url = urllib.parse.urlsplit(self.path)
        signed = int(urllib.parse.parse_qs(url.query)["size"][0])
        length = self.headers.get("Content-Length")
        if self.headers.get("Transfer-Encoding") or length != str(signed):
            self.send_error(403, "SignatureDoesNotMatch")
            return
        data = self.rfile.read(signed)
        if type(self).expire_puts:
            type(self).expire_puts -= 1
            self.send_error(403, "Request has expired")
            return
        assert self.headers.get("Content-Type") == "application/octet-stream"
        name = urllib.parse.unquote(url.path.lstrip("/").split("/")[-1])
        with open(os.path.join(self.root, name), "wb") as out:
            out.write(data)
        type(self).put.append(name)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def serve(tmp_path, monkeypatch):
    """A server per directory: ``serve(name)`` gives its base address."""
    servers = []
    monkeypatch.setattr(runtime, "PIECE", 1000)
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)

    def start(name, **behaviour):
        root = tmp_path / name
        root.mkdir(exist_ok=True)
        handler = type("Handler", (_Files,), {"root": str(root), "asked": [], "uses": {}, "put": [], **behaviour})
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


class _Signer(http.server.BaseHTTPRequestHandler):
    """A backend's signing service: answers a ``get`` of ``runtime/<name>``
    with the file server's URL of that name and a new ``?sig=`` each time,
    as a presigned URL would be, or ``refuse`` as its status."""

    files = ""
    token = "the-token"
    refuse = 0
    signed: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        asked = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.headers.get("Authorization") != "Bearer " + self.token:
            self.send_error(401)
            return
        if self.refuse:
            self.send_error(self.refuse)
            return
        assert asked["op"] == "get" and asked["path"].startswith("runtime/")
        type(self).signed.append(asked["path"])
        name = asked["path"][len("runtime/"):]
        body = json.dumps({"url": "%s/%s?sig=%d" % (self.files, urllib.parse.quote(name), len(self.signed)),
                           "expires": 600}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def backend():
    """``backend(files, **behaviour)``: a signing service for the file
    server at ``files``; gives its address and its handler."""
    servers = []

    def start(files, **behaviour):
        handler = type("Signer", (_Signer,), {"files": files, "signed": [], **behaviour})
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return "http://127.0.0.1:%d/api/cache/global" % server.server_port, handler

    yield start
    for server in servers:
        server.shutdown()


def test_the_cache_first_then_the_mirror_then_the_index(serve, backend, tmp_path):
    index, index_root, index_handler = serve("index")
    bucket, bucket_root, _ = serve("bucket")
    mirror, mirror_root, _ = serve("mirror")
    cached = _wheel(index_root, "torch-2.11.0+cu128-cp312-linux_x86_64.whl", 2500)
    (bucket_root / cached["name"]).write_bytes((index_root / cached["name"]).read_bytes())
    mirrored = _wheel(index_root, "mirrored-1.0-py3-none-any.whl", 300)
    (mirror_root / mirrored["name"]).write_bytes((index_root / mirrored["name"]).read_bytes())
    missing = _wheel(index_root, "new-1.0-py3-none-any.whl", 100)
    for entry in (cached, mirrored, missing):
        entry["url"] = index + "/" + entry["name"]
    address, signer = backend(bucket)
    done = runtime.fetch({"version": 1, "files": [cached, mirrored, missing]}, str(tmp_path / "out"),
                         mirror=mirror, signer=runtime.Signer(address, "the-token"))
    assert done["kinds"] == {cached["name"]: "cache", mirrored["name"]: "mirror", missing["name"]: "index"}
    for entry in (cached, mirrored, missing):
        assert (tmp_path / "out" / entry["name"]).read_bytes() == (index_root / entry["name"]).read_bytes()
    # Signed once per file, when it is fetched; the signed URL is never what
    # the log names as the file's source.
    assert sorted(signer.signed) == sorted("runtime/" + entry["name"] for entry in (cached, mirrored, missing))
    assert "sig=" not in done["from"][cached["name"]]
    assert runtime._whence(done) == "1 from the cache, 1 from the mirror, 1 from the index"
    assert not any(path.endswith(cached["name"]) for path in index_handler.asked)


def test_a_cache_that_will_not_sign_falls_back_to_the_index(serve, backend, tmp_path):
    index, index_root, _ = serve("index")
    bucket, _, _ = serve("bucket")
    entry = _wheel(index_root, "torch-2.11.0-cp312-linux_x86_64.whl", 1500)
    entry["url"] = index + "/" + entry["name"]
    address, _ = backend(bucket, refuse=503)
    done = runtime.fetch({"version": 1, "files": [entry]}, str(tmp_path / "out"),
                         signer=runtime.Signer(address, "the-token"))
    assert done["kinds"] == {entry["name"]: "index"}


def test_an_expired_url_is_signed_again_and_the_piece_fetched(serve, backend, tmp_path):
    # Each signature serves three requests - the size and two pieces - and
    # then answers 403, as a presigned URL does once it expires mid-download.
    bucket, bucket_root, files = serve("bucket", lifetime=3)
    entry = _wheel(bucket_root, "torch-2.11.0-cp312-linux_x86_64.whl", 9500)
    entry["url"] = "http://127.0.0.1:9/unused/" + entry["name"]
    address, signer = backend(bucket)
    done = runtime.fetch({"version": 1, "files": [entry]}, str(tmp_path / "out"),
                         signer=runtime.Signer(address, "the-token"))
    assert done["kinds"] == {entry["name"]: "cache"}
    assert (tmp_path / "out" / entry["name"]).read_bytes() == (bucket_root / entry["name"]).read_bytes()
    # Some piece was refused as expired, and the file's URL was signed again
    # for it - but not once per refused piece past what RESIGNS allows.
    assert max(files.uses.values()) > files.lifetime
    assert 2 <= len(signer.signed) <= runtime.RESIGNS + 1


def test_a_url_the_bucket_always_refuses_is_signed_a_bounded_number_of_times(serve, backend, tmp_path, monkeypatch):
    index, index_root, _ = serve("index")
    bucket, bucket_root, _ = serve("bucket", lifetime=1)
    entry = _wheel(index_root, "torch-2.11.0-cp312-linux_x86_64.whl", 9500)
    (bucket_root / entry["name"]).write_bytes((index_root / entry["name"]).read_bytes())
    entry["url"] = index + "/" + entry["name"]
    monkeypatch.setattr(runtime, "RESIGNS", 2)
    address, signer = backend(bucket)
    done = runtime.fetch({"version": 1, "files": [entry]}, str(tmp_path / "out"),
                         signer=runtime.Signer(address, "the-token"))
    # Every signature serves one request: the cache gives the file up after
    # RESIGNS renewals, and the index serves it.
    assert done["kinds"] == {entry["name"]: "index"}
    assert len(signer.signed) == 1 + 2


def test_the_token_comes_from_the_environment_and_never_the_command_line(serve, backend, tmp_path, monkeypatch):
    bucket, bucket_root, _ = serve("bucket")
    entry = _wheel(bucket_root, "small-1.0-py3-none-any.whl", 10)
    entry["url"] = "http://127.0.0.1:9/unused/" + entry["name"]
    (tmp_path / "runtime.json").write_text(json.dumps({"version": 1, "files": [entry]}))
    address, signer = backend(bucket)
    arguments = ["fetch", "--manifest", str(tmp_path / "runtime.json"), "--dir", str(tmp_path / "out"),
                 "--signed", address]
    monkeypatch.delenv("SMITH_CACHE_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="SMITH_CACHE_TOKEN"):
        runtime.main(arguments)
    monkeypatch.setenv("SMITH_CACHE_TOKEN", "the-token")
    runtime.main(arguments)
    assert signer.signed == ["runtime/" + entry["name"]]

    # And smith puts it there: in the environment of the command it runs,
    # never in its arguments; without the backend's cache, no --signed at all.
    config = types.SimpleNamespace(python="python", runtime="/app/runtime.json", backend="http://backend",
                                   token="the-token", cache_service=True)
    command, env = smith.Agent.runtime_command(types.SimpleNamespace(config=config), "runtime.py")
    assert command[command.index("--signed") + 1] == "http://backend/api/cache/global"
    assert not any("the-token" in part for part in command)
    assert env["SMITH_CACHE_TOKEN"] == "the-token"
    config.cache_service = False
    command, env = smith.Agent.runtime_command(types.SimpleNamespace(config=config), "runtime.py")
    assert "--signed" not in command


# ─── filling the cache ────────────────────────────────────────────────


class _Builder(http.server.BaseHTTPRequestHandler):
    """A backend's builder service: ``list`` answers what the bucket's
    directory holds under ``runtime/``, ``put`` a URL of the bucket signed
    for that size; ``refuse`` answers every call with that status."""

    bucket = ""
    root = ""
    token = "builder-token"
    refuse = 0
    asked: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        asked = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.headers.get("Authorization") != "Bearer " + self.token:
            self.send_error(401)
            return
        if self.refuse:
            self.send_error(self.refuse)
            return
        type(self).asked.append((asked["op"], asked["path"]))
        if asked["op"] == "list":
            assert asked["path"] == "runtime/"
            answer = {"paths": sorted("runtime/" + name for name in os.listdir(self.root))}
        else:
            assert asked["op"] == "put" and asked["path"].startswith("runtime/")
            name = asked["path"][len("runtime/"):]
            answer = {"url": "%s/%s?sig=%d&size=%d" % (self.bucket, urllib.parse.quote(name), len(self.asked),
                                                       asked["size"]), "expires": 600}
        body = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def builder():
    """``builder(bucket, root, **behaviour)``: a builder service for the
    bucket at ``bucket`` whose files are in ``root``; its address and handler."""
    servers = []

    def start(bucket, root, **behaviour):
        handler = type("Builder", (_Builder,), {"bucket": bucket, "root": str(root), "asked": [], **behaviour})
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return "http://127.0.0.1:%d/api/cache-runtime/sign" % server.server_port, handler

    yield start
    for server in servers:
        server.shutdown()


def _lock(tmp_path, index, entries):
    for entry in entries:
        entry["url"] = index + "/" + entry["name"]
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({"version": 1, "files": entries}))
    return str(path)


def test_fill_uploads_what_the_cache_lacks_and_leaves_what_it_has(serve, builder, tmp_path):
    index, index_root, index_handler = serve("index")
    bucket, bucket_root, bucket_handler = serve("bucket")
    present = _wheel(index_root, "present-1.0-py3-none-any.whl", 300)
    (bucket_root / present["name"]).write_bytes((index_root / present["name"]).read_bytes())
    big = _wheel(index_root, "torch-2.11.0+cu128-cp312-linux_x86_64.whl", 5500)
    small = _wheel(index_root, "small-1.0-py3-none-any.whl", 10)
    manifest = json.loads(open(_lock(tmp_path, index, [present, big, small])).read())
    address, service = builder(bucket, bucket_root)
    work = tmp_path / "work"
    done = runtime.fill(manifest, runtime.Builder(address, "builder-token"), directory=str(work))
    assert done["present"] == [present["name"]]
    assert sorted(done["uploaded"]) == sorted([big["name"], small["name"]]) and not done["failed"]
    for entry in (present, big, small):
        assert (bucket_root / entry["name"]).read_bytes() == (index_root / entry["name"]).read_bytes()
    # Listed once, a put signed per missing wheel, the present one never
    # fetched or put; nothing is left on disk.
    assert service.asked == [("list", "runtime/"), ("put", "runtime/" + big["name"]), ("put", "runtime/" + small["name"])]
    assert not any(path.endswith(present["name"]) for path in index_handler.asked)
    assert sorted(bucket_handler.put) == sorted([big["name"], small["name"]])
    assert os.listdir(work) == []


def test_fill_never_uploads_bytes_that_are_not_the_manifests(serve, builder, tmp_path, monkeypatch):
    index, index_root, _ = serve("index")
    bucket, bucket_root, bucket_handler = serve("bucket")
    bad = _wheel(index_root, "torch-2.11.0-cp312-linux_x86_64.whl", 1500)
    bad["sha256"] = "0" * 64
    good = _wheel(index_root, "good-1.0-py3-none-any.whl", 100)
    manifest = _lock(tmp_path, index, [bad, good])
    address, service = builder(bucket, bucket_root)
    monkeypatch.setenv("SMITH_BUILDER_TOKEN", "builder-token")
    assert runtime.main(["fill", "--manifest", manifest, "--builder", address]) == 1
    # The other wheel still went up; the poisoned one was never even signed.
    assert bucket_handler.put == [good["name"]]
    assert ("put", "runtime/" + bad["name"]) not in service.asked
    assert not (bucket_root / bad["name"]).exists()


def test_an_expired_put_is_signed_again(serve, builder, tmp_path):
    index, index_root, _ = serve("index")
    bucket, bucket_root, bucket_handler = serve("bucket", expire_puts=1)
    entry = _wheel(index_root, "torch-2.11.0-cp312-linux_x86_64.whl", 2500)
    manifest = json.loads(open(_lock(tmp_path, index, [entry])).read())
    address, service = builder(bucket, bucket_root)
    done = runtime.fill(manifest, runtime.Builder(address, "builder-token"))
    assert done["uploaded"] == [entry["name"]]
    assert (bucket_root / entry["name"]).read_bytes() == (index_root / entry["name"]).read_bytes()
    assert service.asked.count(("put", "runtime/" + entry["name"])) == 2


def test_a_dry_run_says_what_is_missing_and_uploads_nothing(serve, builder, tmp_path, monkeypatch, capsys):
    index, index_root, index_handler = serve("index")
    bucket, bucket_root, bucket_handler = serve("bucket")
    entry = _wheel(index_root, "new-1.0-py3-none-any.whl", 100)
    manifest = _lock(tmp_path, index, [entry])
    address, service = builder(bucket, bucket_root)
    monkeypatch.setenv("SMITH_BUILDER_TOKEN", "builder-token")
    assert runtime.main(["fill", "--manifest", manifest, "--builder", address, "--dry-run"]) == 0
    assert "missing  " + entry["name"] in capsys.readouterr().out
    assert service.asked == [("list", "runtime/")]
    assert index_handler.asked == [] and bucket_handler.put == []


def test_the_builder_token_comes_from_the_environment_and_never_the_command_line(serve, builder, tmp_path, monkeypatch):
    index, index_root, _ = serve("index")
    bucket, bucket_root, bucket_handler = serve("bucket")
    entry = _wheel(index_root, "small-1.0-py3-none-any.whl", 10)
    manifest = _lock(tmp_path, index, [entry])
    address, _ = builder(bucket, bucket_root)
    arguments = ["fill", "--manifest", manifest, "--builder", address]
    monkeypatch.delenv("SMITH_BUILDER_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="SMITH_BUILDER_TOKEN"):
        runtime.main(arguments)
    # There is no argument to pass it as.
    with pytest.raises(SystemExit):
        runtime.main(arguments + ["--token", "builder-token"])
    monkeypatch.setenv("SMITH_BUILDER_TOKEN", "wrong")
    with pytest.raises(SystemExit, match="401"):
        runtime.main(arguments)
    monkeypatch.setenv("SMITH_BUILDER_TOKEN", "builder-token")
    assert runtime.main(arguments) == 0
    assert bucket_handler.put == [entry["name"]]


def test_a_backend_without_a_cache_is_nothing_to_fill(serve, builder, tmp_path, monkeypatch, capsys):
    index, index_root, index_handler = serve("index")
    bucket, bucket_root, _ = serve("bucket")
    entry = _wheel(index_root, "small-1.0-py3-none-any.whl", 10)
    manifest = _lock(tmp_path, index, [entry])
    address, _ = builder(bucket, bucket_root, refuse=503)
    monkeypatch.setenv("SMITH_BUILDER_TOKEN", "builder-token")
    assert runtime.main(["fill", "--manifest", manifest, "--builder", address]) == 0
    assert "keeps no cache" in capsys.readouterr().out
    assert index_handler.asked == []
