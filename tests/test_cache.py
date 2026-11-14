"""The cache of compiled wheels and kernels (GPU-181).

What these tests hold it to is the issue's "done when": a second machine with
the same key skips the compilation, and a machine with a different key loads
nothing of the first's. The store is a directory; the GPUs are given rather
than asked of a driver, so each test says which machine it is.
"""

import json
import os
import threading

import pytest

import cache as _cache
from cache import main


def _fill(directory, files):
    for name, data in files.items():
        path = os.path.join(directory, *name.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(data)


def _read(directory):
    return {name: open(os.path.join(directory, *name.split("/")), "rb").read() for name in _cache._local_files(directory)}


def test_a_second_machine_with_the_same_key_gets_what_the_first_built(tmp_path):
    store = _cache.LocalStore(str(tmp_path / "store"))
    key = _cache.compute_key("triton", gpus=["sm_89"])
    first, second = str(tmp_path / "first"), str(tmp_path / "second")
    _fill(first, {"ab/kernel.cubin": b"\x7fELF", "ab/kernel.json": b"{}"})

    sent = _cache.push(store, key, first)
    got = _cache.pull(store, _cache.compute_key("triton", gpus=["sm_89"]), second)

    assert (sent.hit, sent.files) == (False, 2)
    assert (got.hit, got.files) == (True, 2)
    assert _read(second) == _read(first)


def test_another_gpu_loads_nothing(tmp_path):
    store = _cache.LocalStore(str(tmp_path / "store"))
    first = str(tmp_path / "first")
    _fill(first, {"kernel.cubin": b"built for sm_89"})
    _cache.push(store, _cache.compute_key("triton", gpus=["sm_89"]), first)

    target = str(tmp_path / "other")
    got = _cache.pull(store, _cache.compute_key("triton", gpus=["sm_90"]), target)

    assert not got.hit
    assert _read(target) == {}


def test_a_stored_key_that_differs_is_a_miss_whatever_its_id_says(tmp_path):
    """The id only finds the key: a stored key that does not match field by
    field - a hash collision, or a key someone edited - loads nothing."""
    store = _cache.LocalStore(str(tmp_path / "store"))
    key = _cache.compute_key("wheels", gpus=["sm_89"])
    base = "wheels/%s" % key.id
    forged = dict(key.fields, torch="2.0.0+cu118")
    store.put(base + "/key.json", json.dumps(forged).encode())
    store.put(base + "/files/flash_attn.whl", b"for another torch")

    got = _cache.pull(store, key, str(tmp_path / "wheels"))

    assert not got.hit
    assert "torch" in got.reason
    with pytest.raises(ValueError):
        _cache.push(store, key, str(tmp_path / "wheels"))


def test_a_push_sends_only_what_the_store_lacks(tmp_path):
    store = _cache.LocalStore(str(tmp_path / "store"))
    key = _cache.compute_key("inductor", gpus=[])
    directory = str(tmp_path / "inductor")
    _fill(directory, {"a.py": b"a"})
    _cache.push(store, key, directory)
    _fill(directory, {"b.py": b"bb"})

    again = _cache.push(store, key, directory)

    assert (again.hit, again.files, again.bytes) == (True, 1, 2)


class DictStore:
    """A store that lists whatever names it was given, as a bucket can."""

    def __init__(self):
        self.objects = {}

    def get(self, path):
        return self.objects.get(path)

    def put(self, path, data):
        self.objects[path] = data

    def list(self, prefix):
        return sorted(path for path in self.objects if path.startswith(prefix))


@pytest.mark.parametrize("name", ["../../escape.whl", "/etc/escape.whl", "a//b.whl", "a\\..\\..\\escape.whl"])
def test_a_name_that_leaves_the_directory_is_refused(tmp_path, name):
    store = DictStore()
    key = _cache.compute_key("wheels", gpus=[])
    store.put("wheels/%s/key.json" % key.id, key.canonical().encode())
    store.put("wheels/%s/files/%s" % (key.id, name), b"x")

    with pytest.raises(ValueError):
        _cache.pull(store, key, str(tmp_path / "wheels" / "inner"))
    assert not (tmp_path / "escape.whl").exists()


def test_a_library_named_with_with_is_part_of_the_key():
    plain = _cache.compute_key("tilelang", gpus=["sm_89"])
    named = _cache.compute_key("tilelang", ["pyyaml"], gpus=["sm_89"])

    assert "lib:pyyaml" in named.fields
    assert plain.id != named.id


def test_the_command_line_pulls_what_it_pushed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(_cache, "gpu_capabilities", lambda: ["sm_89"])
    first, second, store = (str(tmp_path / name) for name in ("first", "second", "store"))
    _fill(first, {"w.whl": b"wheel"})

    assert main(["push", "--name", "wheels", "--dir", first, "--store", store]) == 0
    assert main(["pull", "--name", "wheels", "--dir", second, "--store", store]) == 0

    assert "hit" in capsys.readouterr().out.splitlines()[-1]
    assert _read(second) == {"w.whl": b"wheel"}


def test_a_bucket_without_its_keys_is_refused_by_name(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("SMITH_CACHE_ACCESS_KEY", raising=False)
    monkeypatch.delenv("SMITH_CACHE_SECRET_KEY", raising=False)

    code = main(["pull", "--name", "wheels", "--dir", str(tmp_path), "--store", "s3://gpuzero-cache"])

    assert code == 1
    assert "SMITH_CACHE_ACCESS_KEY" in capsys.readouterr().err


class SigningService:
    """A signing service and the bucket behind it, in one small HTTP server.

    ``POST /sign`` answers the protocol in ``cache.py``; its URLs point
    back at ``/objects/<path>``, which keeps the bytes. Writes under ``ro/``
    are refused, as a service refuses a part that machines may only read.
    """

    def __init__(self, token="t0ken"):
        import threading
        import urllib.parse
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.token = token
        self.objects = {}
        self.object_auth = []
        #: The size each put declared, by path: the bucket takes no other.
        self.declared = {}
        #: How many signatures to refuse with ``busy_status`` before signing
        #: again, as a proxy's rate limit does (GPU-385), and the
        #: ``Retry-After`` they carry, if any.
        self.busy = 0
        self.busy_status = 429
        self.retry_after = None
        #: When set, the first signature asked for each (op, path) under
        #: ``/files/`` is refused: a pull's burst, without leaving to chance
        #: which of its transfers the limit falls on.
        self.busy_once = False
        self.refused = set()
        #: Every operation asked for, in order, for a test to count.
        self.asked = []
        #: How long a signed URL lives, as the service says.
        self.expires = 600
        self.lock = threading.Lock()
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body=b"", kind="application/json", headers=()):
                self.send_response(status)
                self.send_header("Content-Type", kind)
                for name, value in headers:
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.headers.get("Authorization") != "Bearer " + service.token:
                    return self._send(401, b'{"detail": "no token"}')
                op, path = body["op"], body["path"]
                with service.lock:
                    service.asked.append(op)
                    busy = service.busy > 0
                    if busy:
                        service.busy -= 1
                    elif service.busy_once and "/files/" in path and (op, path) not in service.refused:
                        service.refused.add((op, path))
                        busy = True
                if busy:
                    extra = [("Retry-After", service.retry_after)] if service.retry_after else []
                    return self._send(service.busy_status, b"<html>Too Many Requests</html>", "text/html", extra)
                if op == "put" and path.startswith("ro/"):
                    return self._send(403, b'{"detail": "read-only"}')
                if op == "list":
                    paths = sorted(p for p in service.objects if p.startswith(path))
                    return self._send(200, json.dumps({"paths": paths}).encode())
                if op == "get_many":
                    urls = {
                        one: "%s/objects/%s" % (service.url, urllib.parse.quote(one))
                        for one in body["paths"]
                    }
                    return self._send(200, json.dumps({"urls": urls, "expires": service.expires}).encode())
                if op == "put":
                    if not isinstance(body.get("size"), int):
                        return self._send(403, b'{"detail": "a put says its size"}')
                    service.declared[path] = body["size"]
                url = "%s/objects/%s" % (service.url, urllib.parse.quote(path))
                return self._send(200, json.dumps({"url": url}).encode())

            def _object(self):
                service.object_auth.append(self.headers.get("Authorization"))
                return urllib.parse.unquote(self.path[len("/objects/"):])

            def do_GET(self):
                path = self._object()
                if path not in service.objects:
                    return self._send(404)
                return self._send(200, service.objects[path], "application/octet-stream")

            def do_PUT(self):
                path = self._object()
                data = self.rfile.read(int(self.headers["Content-Length"]))
                if len(data) != service.declared.get(path):
                    # As a URL signed with its length refuses another.
                    return self._send(403)
                service.objects[path] = data
                return self._send(200)

        class Server(ThreadingHTTPServer):
            # socketserver queues 5 connections by default, and a push opens
            # 16 at a time: on Linux the overflow came back to the client as
            # "Connection reset by peer" and failed the push (GPU-385).
            request_queue_size = 128

        self.server = Server(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def service():
    running = SigningService()
    yield running
    running.close()


def test_a_signed_store_pushes_and_pulls_without_a_bucket_key(tmp_path, monkeypatch, capsys, service):
    monkeypatch.setattr(_cache, "gpu_capabilities", lambda: ["sm_89"])
    monkeypatch.setenv("SMITH_CACHE_TOKEN", service.token)
    for name in ("SMITH_CACHE_ACCESS_KEY", "SMITH_CACHE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    first, second = str(tmp_path / "first"), str(tmp_path / "second")
    _fill(first, {"w.whl": b"wheel", "sub/k.cubin": b"kernel"})
    store = "sign+%s/sign" % service.url

    assert main(["push", "--name", "wheels", "--dir", first, "--store", store]) == 0
    assert main(["pull", "--name", "wheels", "--dir", second, "--store", store]) == 0

    assert "hit" in capsys.readouterr().out.splitlines()[-1]
    assert _read(second) == {"w.whl": b"wheel", "sub/k.cubin": b"kernel"}
    # The token goes to the service only: the bucket's URLs carry their own
    # signature, and a token sent there would be one more place it leaks.
    assert service.object_auth and not any(service.object_auth)


def test_a_signed_store_reports_a_missing_object_as_none(service):
    store = _cache.SignedStore(service.url + "/sign", service.token)
    assert store.get("nothing/here") is None


def test_a_refused_write_fails_the_push_and_says_why(tmp_path, monkeypatch, capsys, service):
    monkeypatch.setattr(_cache, "gpu_capabilities", lambda: ["sm_89"])
    monkeypatch.setenv("SMITH_CACHE_TOKEN", service.token)
    directory = str(tmp_path / "d")
    _fill(directory, {"w.whl": b"wheel"})

    code = main(["push", "--name", "wheels", "--dir", directory, "--store", "sign+%s/sign" % service.url])
    assert code == 0  # wheels/ is writable here

    store = _cache.SignedStore(service.url + "/sign", service.token)
    with pytest.raises(OSError, match="403"):
        store.put("ro/wheels/x", b"x")


@pytest.fixture
def waits(monkeypatch):
    """The waits of the signed store's backoff, recorded instead of slept."""
    waited = []
    monkeypatch.setattr(_cache, "_sleep", waited.append)
    return waited


def test_a_pull_waits_out_a_rate_limit_instead_of_dropping_the_cache(tmp_path, monkeypatch, service, waits):
    """The nginx in front of the service answered a pull's burst of
    signatures with 429, and the pull gave up on the first: the node compiled
    everything the cache held (GPU-385)."""
    monkeypatch.setattr(_cache, "gpu_capabilities", lambda: ["sm_89"])
    monkeypatch.setenv("SMITH_CACHE_TOKEN", service.token)
    first, second = str(tmp_path / "first"), str(tmp_path / "second")
    files = {"k%02d.cubin" % index: b"kernel %d" % index for index in range(40)}
    _fill(first, files)
    store = "sign+%s/sign" % service.url
    assert main(["push", "--name", "inductor", "--dir", first, "--store", store]) == 0

    service.busy_once = True
    assert main(["pull", "--name", "inductor", "--dir", second, "--store", store]) == 0

    assert _read(second) == files
    # The list of the files, then their signatures, all in one get_many since
    # GPU-421: each refused once.
    assert len(waits) == 2
    assert all(0 <= wait <= _cache.RETRY_LONGEST for wait in waits)


@pytest.mark.parametrize("status", [502, 503, 504])
def test_a_backend_between_two_deploys_is_asked_again(service, waits, status):
    service.objects["a/b"] = b"bytes"
    service.busy, service.busy_status = 2, status

    assert _cache.SignedStore(service.url + "/sign", service.token).get("a/b") == b"bytes"
    assert len(waits) == 2


def test_a_retry_after_is_the_wait(service, waits):
    service.objects["a/b"] = b"bytes"
    service.busy, service.retry_after = 1, "3"

    assert _cache.SignedStore(service.url + "/sign", service.token).get("a/b") == b"bytes"
    assert waits == [3.0]


def test_the_wait_doubles_up_to_the_longest():
    ceilings = [_cache.RETRY_FIRST * 2 ** attempt for attempt in range(10)]
    for attempt, ceiling in enumerate(ceilings):
        assert 0 <= _cache._retry_wait(attempt, None) <= min(ceiling, _cache.RETRY_LONGEST)
    # A service asking for an hour gets the longest wait, not the hour.
    assert _cache._retry_wait(0, "3600") == _cache.RETRY_LONGEST


def test_a_limit_that_does_not_lift_still_fails_and_says_so(service, waits):
    service.busy = _cache.SIGNED_ATTEMPTS

    with pytest.raises(OSError, match="refused get a/b: 429"):
        _cache.SignedStore(service.url + "/sign", service.token).get("a/b")
    assert len(waits) == _cache.SIGNED_ATTEMPTS - 1


def test_a_refusal_that_means_no_is_not_asked_again(service, waits):
    with pytest.raises(OSError, match="403"):
        _cache.SignedStore(service.url + "/sign", service.token).put("ro/x", b"x")
    assert waits == []


def test_a_signing_service_without_a_token_is_refused_by_name(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("SMITH_CACHE_TOKEN", raising=False)

    code = main(["pull", "--name", "wheels", "--dir", str(tmp_path), "--store", "sign+https://example.invalid/sign"])

    assert code == 1
    assert "SMITH_CACHE_TOKEN" in capsys.readouterr().err


class _Gate:
    """A store that lets a file through only once four are in flight: a pull
    or a push that moves one file at a time never gets past the first."""

    def __init__(self, inner):
        self.inner = inner
        self.barrier = threading.Barrier(4, timeout=10)

    def get(self, path):
        if "/files/" in path:
            self.barrier.wait()
        return self.inner.get(path)

    def put(self, path, data):
        if "/files/" in path:
            self.barrier.wait()
        self.inner.put(path, data)

    def list(self, prefix):
        return self.inner.list(prefix)


def test_files_move_several_at_a_time(tmp_path):
    """One at a time, every file waits out the store's round trips: 517
    kernels took four minutes to pull on a node (GPU-181)."""
    store = _cache.LocalStore(str(tmp_path / "store"))
    key = _cache.compute_key("triton", gpus=["sm_86"])
    first, second = str(tmp_path / "first"), str(tmp_path / "second")
    _fill(first, {"k%d/kernel.cubin" % n: b"%d" % n for n in range(8)})

    sent = _cache.push(_Gate(store), key, first)
    got = _cache.pull(_Gate(store), key, second)

    assert (sent.files, got.files) == (8, 8)
    assert _read(second) == _read(first)


def test_a_pull_of_many_files_asks_for_their_signatures_in_batches(tmp_path, monkeypatch, service):
    """GPU-421: one signature per file was a round trip to the service per
    file, 29 s for Triton's 705 on a node. The reads are signed a few hundred
    at a time, and each file is then fetched from the bucket directly."""
    monkeypatch.setattr(_cache, "gpu_capabilities", lambda: ["sm_89"])
    monkeypatch.setattr(_cache, "SIGN_AT_ONCE", 4)
    monkeypatch.setenv("SMITH_CACHE_TOKEN", service.token)
    first, second = str(tmp_path / "first"), str(tmp_path / "second")
    files = {"k%d.cubin" % index: b"kernel %d" % index for index in range(10)}
    _fill(first, files)
    store = "sign+%s/sign" % service.url
    assert main(["push", "--name", "triton", "--dir", first, "--store", store]) == 0

    service.asked.clear()
    assert main(["pull", "--name", "triton", "--dir", second, "--store", store]) == 0

    assert _read(second) == files
    assert service.asked.count("get_many") == 3  # 10 reads, 4 to a batch
    assert service.asked.count("get") == 1  # key.json alone


def test_a_read_signed_too_long_ago_is_signed_again_alone(service):
    """A URL signed in a batch waits for the transfers ahead of it; one that
    has aged past its life, less a margin, is not used."""
    service.objects["a/b"] = b"bytes"
    service.expires = 30  # under SIGNED_MARGIN: already too old to use
    store = _cache.SignedStore(service.url + "/sign", service.token)
    store.sign_reads("a/", ["a/b"])
    service.asked.clear()
    assert store.get("a/b") == b"bytes"
    assert service.asked == ["get"]
