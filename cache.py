"""cache: a directory of compiled things, shared between machines that would compile it the same way (GPU-181).

    python cache.py key  --name wheels
    python cache.py pull --name triton --dir <directory> --store <store> [--with triton]
    python cache.py push --name triton --dir <directory> --store <store> [--with triton]

A fresh machine spends its first minutes - for flash-attn, hours - compiling
what another machine like it already compiled: wheels without a binary on
PyPI, and the kernels Triton, Inductor and TileLang build on first use and
keep in a cache directory that dies with the machine. ``pull`` fills such a
directory from a store before the work starts, and ``push`` sends back what
the work added.

**smith's, not the run's (GPU-267).** Until Ravex 0.7.8 this was ``ravex
cache``. It moved here because it is the node's work - what a machine keeps
between jobs - and not the run's, which is Ravex's. The key, the layout and
the protocol did not change, so a store written by ``ravex cache`` is read as
it is. It is a file of its own, beside ``smith.py`` and standard library like
it, because smith runs it with **the job's interpreter**: torch's version and
the libraries named with ``--with`` are part of the key, and they are the ones
the training will use, not the ones smith runs with.

**The key is everything that changes the binary.** Not "the architecture":
the CPU, every GPU's compute capability, CUDA, torch, Python, and the version
of each library named with ``--with``. A key too wide hands a machine a kernel
built for another combination, which crashes at best and computes the wrong
numbers in silence at worst. So the whole key is stored beside the files, and
a pull **compares it**, field by field, with the one this machine computes:
the short id in the store's path only finds the candidate, it does not vouch
for it.

**Who may write a store is the store's business, not this file's.** A shared
store that the machines it serves can write is a way to run code on each
other's machines, so a common cache is written by one trusted builder and only
read everywhere else; a person's own cache, in their own bucket, is theirs.
Nothing here can tell the two apart, which is why nothing here decides it.

**A directory is a set of files, never rewritten.** Wheels and the kernel
caches are named by their content, so a name already in the store already
holds those bytes: a push sends only the names the store lacks, a pull only
the names the directory lacks, and two machines pushing the same key at once
write the same bytes twice at worst.

The layout under the store's root::

    <name>/<id>/key.json      the whole key, written by the first push
    <name>/<id>/files/<path>  one object per file of the directory

The store is a small interface, :class:`Store`: a directory, or a bucket
through ``moonclip.Bucket`` - saving files is Moonclip's, and the job's
environment has it, since Ravex needs it. A bucket's keys come from the
environment, never from the command line, where any user of the machine can
read them:

``SMITH_CACHE_ACCESS_KEY``, ``SMITH_CACHE_SECRET_KEY``
    The bucket's keys. Not the run's ``MOONCLIP_S3_*``: a cache is a bucket of
    its own, and a machine that may read it is not always one that may write.
``SMITH_CACHE_ENDPOINT``, ``SMITH_CACHE_REGION``, ``SMITH_CACHE_PATH_STYLE``
    Where the bucket is: for R2 the endpoint alone,
    ``https://<account>.r2.cloudflarestorage.com``.

**A store whose keys this machine does not hold**, ``sign+https://<service>``
(:class:`SignedStore`). A machine that runs somebody else's code cannot keep a
bucket's keys from that code, and a bucket key is worth the whole bucket. So
the machine holds only a token for a service, which signs one operation at a
time and decides which it allows; the bytes go between the machine and the
bucket, never through the service. The protocol, one ``POST`` to the address
with ``Authorization: Bearer <SMITH_CACHE_TOKEN>`` and a JSON body::

    {"op": "get",  "path": "<path>"}    ->  {"url": "<presigned GET>"}
    {"op": "put",  "path": "<path>", "size": <bytes>}  ->  {"url": "<presigned PUT>"}
    {"op": "list", "path": "<prefix>"}  ->  {"paths": ["<path>", ...]}
    {"op": "get_many", "path": "<prefix>", "paths": ["<path>", ...]}
                                        ->  {"urls": {"<path>": "<presigned GET>", ...}}

``get_many`` signs a pull's reads up to a thousand at a time (GPU-421): a
kernel cache is hundreds of small files, and a signature asked for each was a
round trip to the service per file - 705 of Triton's took 29 s on a node.

Paths are this module's, relative to the store's root; where that root is in
the bucket is the service's business. A signature refused with 429, 502, 503
or 504 is asked for again, a few times over some seconds (``RETRY_STATUSES``,
GPU-385): those say "not now", and one refusal would cost the whole cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import metadata
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, TypeVar

#: Characters of the key's hash in the store's path. Only for finding the key:
#: what decides is the comparison of the whole of it.
ID_CHARS = 16

#: Beside the files, the key they were built under.
KEY_FILE = "key.json"

#: Bumped when the layout or the key's fields change, so a store written by
#: one version is never read as another's. Still the layout ``ravex cache``
#: wrote: moving the code changed neither.
LAYOUT = 1

#: The name a file has while a pull writes it. Such files are never pushed:
#: one left by a pull that was killed is half a file under a name of its own.
#: ``.ravex-`` is what ``ravex cache`` named them, and a directory on a
#: machine's disk may still hold one.
PARTIAL_PREFIXES = (".smith-cache-", ".ravex-")


class Store(Protocol):
    """Where the cache lives: objects addressed by a relative path.

    Called from several threads at once (``TRANSFERS``), one path per call.
    """

    def get(self, path: str) -> Optional[bytes]:
        """The object's bytes, or None when there is none."""

    def put(self, path: str, data: bytes) -> None:
        """Write the whole object."""

    def list(self, prefix: str) -> List[str]:
        """Every path under ``prefix``, as given to :meth:`put`."""


class LocalStore:
    """A store on a disk: for tests, and for machines sharing a filesystem."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)

    def _path(self, path: str) -> str:
        return os.path.join(self.root, *path.split("/"))

    def get(self, path: str) -> Optional[bytes]:
        try:
            with open(self._path(path), "rb") as handle:
                return handle.read()
        except FileNotFoundError:
            return None

    def put(self, path: str, data: bytes) -> None:
        _write_atomic(self._path(path), data)

    def list(self, prefix: str) -> List[str]:
        base = self._path(prefix)
        found = []
        for folder, _dirs, files in os.walk(base):
            for name in files:
                full = os.path.join(folder, name)
                found.append(prefix.rstrip("/") + "/" + os.path.relpath(full, base).replace(os.sep, "/"))
        return sorted(found)


#: How long one request of a signed store may take: a wheel can be hundreds
#: of megabytes on a slow link, and a cache that gives up early only costs
#: the compilation it would have saved.
SIGNED_TIMEOUT = 600


#: The answers of a signing service that say "not now" rather than "no": a
#: proxy's limit (429) or a backend between two deploys (502, 503, 504). A
#: pull asks for one signature per file, 16 at a time, and gave up on the
#: first 429 of the nginx in front of the service, which threw away the whole
#: cache for a moment's excess (GPU-385). These are asked again.
RETRY_STATUSES = (429, 502, 503, 504)

#: How many times one signature is asked for in all, and the longest wait
#: between two: 0.5, 1, 2, 4 and 8 seconds at most, about 15 in all before
#: the pull gives up and the machine compiles.
SIGNED_ATTEMPTS = 6
RETRY_FIRST = 0.5
RETRY_LONGEST = 8.0

#: Replaced in tests, which should not wait out a backoff.
_sleep = time.sleep


def _retry_wait(attempt: int, retry_after: Optional[str]) -> float:
    """Seconds to wait before attempt ``attempt + 1``.

    A ``Retry-After`` in seconds is taken as the service's own answer, up to
    the longest wait; without one the wait doubles, with full jitter, so the
    16 transfers that were refused together do not all come back together.
    """
    if retry_after is not None:
        try:
            return min(max(float(retry_after), 0.0), RETRY_LONGEST)
        except ValueError:
            pass
    return random.uniform(0, min(RETRY_FIRST * (2 ** attempt), RETRY_LONGEST))


#: How many reads one ``get_many`` asks to sign: half the service's cap, so a
#: request stays well inside it (GPU-421).
SIGN_AT_ONCE = 500

#: How long before a signed URL runs out it is no longer used: one signed in
#: a batch waits for the transfers ahead of it, and on a slow link a URL can
#: age past its life before its turn. Then it is signed again, alone.
SIGNED_MARGIN = 60.0


class SignedStore:
    """A store whose every operation a service signs (see the module's
    docstring): no bucket key on this machine, only ``token`` for the service.
    """

    def __init__(self, address: str, token: str) -> None:
        self.address = address
        self.token = token
        #: Reads signed ahead by :meth:`sign_reads`: path -> (URL, when it is
        #: no longer to be used, by ``time.monotonic``).
        self._signed: Dict[str, Tuple[str, float]] = {}

    def _ask(self, op: str, path: str, size: Optional[int] = None,
             paths: Optional[List[str]] = None) -> Dict[str, object]:
        asked: Dict[str, object] = {"op": op, "path": path}
        if paths is not None:
            asked["paths"] = paths
        if size is not None:
            # A put says how large it is: the service may sign the URL with
            # that length, so the bucket takes no more, and refuse a file
            # past what it keeps.
            asked["size"] = size
        body = json.dumps(asked).encode("utf-8")
        request = urllib.request.Request(
            self.address,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.token,
                # Not urllib's default name, which a CDN in front of a service
                # may refuse as a bot's (GPU-193).
                "User-Agent": "smith",
            },
        )
        answer: object = None
        for attempt in range(SIGNED_ATTEMPTS):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    answer = json.loads(response.read().decode("utf-8") or "{}")
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                if exc.code in RETRY_STATUSES and attempt + 1 < SIGNED_ATTEMPTS:
                    _sleep(_retry_wait(attempt, exc.headers.get("Retry-After") if exc.headers else None))
                    continue
                raise OSError("the signing service refused %s %s: %d %s" % (op, path, exc.code, detail)) from None
        if not isinstance(answer, dict):
            raise ValueError("the signing service answered %s %s with %r" % (op, path, answer))
        return answer

    def _url(self, op: str, path: str, size: Optional[int] = None) -> str:
        url = self._ask(op, path, size).get("url")
        if not isinstance(url, str) or not url.startswith(("https://", "http://")):
            raise ValueError("the signing service gave no URL for %s %s" % (op, path))
        return url

    def sign_reads(self, prefix: str, paths: List[str]) -> None:
        """Sign the reads of ``paths``, all under ``prefix``, :data:`SIGN_AT_ONCE`
        to a request, for the :meth:`get` of each that follows (GPU-421)."""
        for start in range(0, len(paths), SIGN_AT_ONCE):
            batch = paths[start:start + SIGN_AT_ONCE]
            answer = self._ask("get_many", prefix, paths=batch)
            urls, expires = answer.get("urls"), answer.get("expires")
            if not isinstance(urls, dict) or set(urls) != set(batch) or not all(
                isinstance(url, str) and url.startswith(("https://", "http://")) for url in urls.values()
            ):
                raise ValueError("the signing service did not sign every read under %s" % prefix)
            usable = time.monotonic() + max(0.0, float(expires if isinstance(expires, (int, float)) else 0) - SIGNED_MARGIN)
            for path, url in urls.items():
                self._signed[path] = (url, usable)

    def _read_url(self, path: str) -> str:
        signed = self._signed.pop(path, None)
        if signed is not None and time.monotonic() < signed[1]:
            return signed[0]
        return self._url("get", path)

    def get(self, path: str) -> Optional[bytes]:
        request = urllib.request.Request(self._read_url(path), headers={"User-Agent": "smith"})
        try:
            with urllib.request.urlopen(request, timeout=SIGNED_TIMEOUT) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise OSError("get %s: %d" % (path, exc.code)) from None

    def put(self, path: str, data: bytes) -> None:
        request = urllib.request.Request(
            self._url("put", path, len(data)),
            data=data,
            method="PUT",
            # The presigned URL signs the host alone; a content type is
            # still sent, or urllib would send a form's.
            headers={"User-Agent": "smith", "Content-Type": "application/octet-stream"},
        )
        try:
            with urllib.request.urlopen(request, timeout=SIGNED_TIMEOUT):
                pass
        except urllib.error.HTTPError as exc:
            raise OSError("put %s: %d" % (path, exc.code)) from None

    def list(self, prefix: str) -> List[str]:
        paths = self._ask("list", prefix).get("paths")
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise ValueError("the signing service listed %s as %r" % (prefix, paths))
        return sorted(paths)


@dataclass(frozen=True)
class Key:
    """What a directory's files were built under, and what a pull must match."""

    fields: Dict[str, object]

    @property
    def id(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:ID_CHARS]

    def canonical(self) -> str:
        return json.dumps(self.fields, sort_keys=True, separators=(",", ":"))

    def differences(self, other: Dict[str, object]) -> List[str]:
        """The fields that differ from ``other``, by name; empty when they match."""
        names = sorted(set(self.fields) | set(other))
        return [name for name in names if self.fields.get(name) != other.get(name)]


def gpu_capabilities() -> List[str]:
    """Every GPU's compute capability, as ``sm_89``, from the driver.

    Asked of ``nvidia-smi`` rather than of torch: importing torch to read one
    number costs seconds and starts CUDA, before a job that may want to set up
    CUDA its own way. An empty list is a machine without an NVIDIA GPU, which
    is a key of its own, not an error.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return sorted({"sm_" + line.strip().replace(".", "") for line in out.splitlines() if line.strip()})


def _version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def compute_key(name: str, libraries: Sequence[str] = (), gpus: Optional[Sequence[str]] = None) -> Key:
    """This machine's key for the directory called ``name``.

    torch's version is read with its local part, ``2.11.0+cu128``, which is
    where a wheel says which CUDA it was built for: the CUDA a kernel links
    against is torch's, not whichever toolkit the machine has installed.
    """
    fields: Dict[str, object] = {
        "layout": LAYOUT,
        "name": name,
        "cpu": platform.machine().lower(),
        "os": sys.platform,
        "python": "cp%d%d" % sys.version_info[:2],
        "gpus": list(gpu_capabilities() if gpus is None else sorted(gpus)),
        "torch": _version("torch"),
    }
    for library in sorted(set(libraries)):
        fields["lib:" + library] = _version(library)
    return Key(fields)


def _write_atomic(path: str, data: bytes) -> None:
    """Write beside the target and rename, so a reader never sees half a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=PARTIAL_PREFIXES[0])
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        os.replace(temp, path)
    except BaseException:
        if os.path.exists(temp):
            os.unlink(temp)
        raise


def _local_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    found = []
    for folder, _dirs, files in os.walk(directory):
        for name in files:
            if name.startswith(PARTIAL_PREFIXES):
                continue
            found.append(os.path.relpath(os.path.join(folder, name), directory).replace(os.sep, "/"))
    return sorted(found)


def _inside(directory: str, relative: str) -> Optional[str]:
    """The file's path in ``directory``, or None for a name that would leave it.

    The names come from the store, which this machine did not write: a name
    with ``..`` or an absolute one would put a file anywhere a pull can reach.
    """
    if not relative or relative.startswith("/") or "\\" in relative:
        return None
    if any(part in ("", ".", "..") for part in relative.split("/")):
        return None
    root = os.path.realpath(directory)
    target = os.path.realpath(os.path.join(root, *relative.split("/")))
    return target if os.path.commonpath([root, target]) == root else None


#: How many files a pull or a push moves at once. A kernel cache is hundreds
#: of files of a few kilobytes each, and one at a time every file waited out
#: a round trip or two - a signature, then the bucket: 517 files took four
#: minutes on a node, longer than the compilation they saved (GPU-181). The
#: stores release the GIL while they wait, so threads are enough.
TRANSFERS = 16

_T = TypeVar("_T")


def _each(work: Callable[[_T], int], items: Sequence[_T]) -> List[int]:
    """``work`` on every item, ``TRANSFERS`` at a time; the first error is raised.

    Files are independent and each is written whole or not at all, so a
    transfer cut short by an error leaves nothing a later pull or push would
    trust: it just finds those names still missing.
    """
    if len(items) <= 1:
        return [work(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(TRANSFERS, len(items)), thread_name_prefix="smith-cache") as pool:
        return list(pool.map(work, items))


@dataclass
class Outcome:
    """What a pull or a push did, for the line the caller prints."""

    hit: bool
    files: int
    bytes: int
    reason: str = ""


def pull(store: Store, key: Key, directory: str) -> Outcome:
    """Fill ``directory`` with the files the store holds under ``key``.

    Loads nothing unless the stored key equals this one: a missing key, an
    unreadable one or one that differs in any field is a miss, and the reason
    says which.
    """
    base = "%s/%s" % (key.fields["name"], key.id)
    raw = store.get(base + "/" + KEY_FILE)
    if raw is None:
        return Outcome(False, 0, 0, "nothing stored for this key")
    try:
        stored = json.loads(raw)
    except ValueError:
        return Outcome(False, 0, 0, "the stored key is not JSON")
    if not isinstance(stored, dict):
        return Outcome(False, 0, 0, "the stored key is not an object")
    differ = key.differences(stored)
    if differ:
        return Outcome(False, 0, 0, "the stored key differs in " + ", ".join(differ))
    have = set(_local_files(directory))
    prefix = base + "/files/"
    wanted = []
    for path in store.list(prefix):
        relative = path[len(prefix):]
        target = _inside(directory, relative)
        if target is None:
            raise ValueError("the store names a file outside the directory: %r" % relative)
        if relative not in have:
            wanted.append((path, target))
    sign_reads = getattr(store, "sign_reads", None)
    if sign_reads is not None and wanted:
        # One request per few hundred reads rather than one per file (GPU-421).
        sign_reads(prefix, [path for path, _ in wanted])

    def fetch(item: Tuple[str, str]) -> int:
        path, target = item
        data = store.get(path)
        if data is None:
            return -1
        _write_atomic(target, data)
        return len(data)

    sizes = [got for got in _each(fetch, wanted) if got >= 0]
    return Outcome(True, len(sizes), sum(sizes))


def push(store: Store, key: Key, directory: str) -> Outcome:
    """Send the store the files of ``directory`` it does not hold under ``key``.

    The key goes first, so that the files are never there without it; a key
    already stored must be this one, or the push refuses rather than mix two
    builds under one id.
    """
    base = "%s/%s" % (key.fields["name"], key.id)
    raw = store.get(base + "/" + KEY_FILE)
    if raw is None:
        store.put(base + "/" + KEY_FILE, key.canonical().encode())
    else:
        try:
            stored = json.loads(raw)
        except ValueError:
            stored = None
        if not isinstance(stored, dict) or key.differences(stored):
            raise ValueError("%s holds another key; refusing to write over it" % base)
    prefix = base + "/files/"
    there = {path[len(prefix):] for path in store.list(prefix)}

    def send(relative: str) -> int:
        with open(os.path.join(directory, *relative.split("/")), "rb") as handle:
            data = handle.read()
        store.put(prefix + relative, data)
        return len(data)

    sizes = _each(send, [relative for relative in _local_files(directory) if relative not in there])
    return Outcome(raw is not None, len(sizes), sum(sizes))


def open_store(address: str, environ: Optional[Dict[str, str]] = None) -> Store:
    """The store at ``address``: ``s3://<bucket>/<prefix>``,
    ``sign+https://<service>``, ``file://<path>`` or a plain path.

    A bucket without both keys is refused here rather than at its first
    request, where the service's answer would be a signature error that does
    not say which variable is missing. Both are always passed to Moonclip,
    which would otherwise look for keys of its own in the environment: the
    run's bucket is not the cache's.
    """
    if address.startswith("s3://"):
        env = os.environ if environ is None else environ
        bucket, _, prefix = address[len("s3://"):].partition("/")
        access, secret = env.get("SMITH_CACHE_ACCESS_KEY"), env.get("SMITH_CACHE_SECRET_KEY")
        if not bucket:
            raise ValueError("%r names no bucket" % address)
        if not (access and secret):
            raise ValueError("a bucket needs SMITH_CACHE_ACCESS_KEY and SMITH_CACHE_SECRET_KEY")
        try:
            from moonclip import Bucket
        except ImportError:
            raise ValueError("a bucket needs moonclip with its Bucket, and this interpreter has none") from None

        style = env.get("SMITH_CACHE_PATH_STYLE")
        return Bucket(
            bucket,
            prefix=prefix.strip("/"),
            endpoint=env.get("SMITH_CACHE_ENDPOINT") or None,
            region=env.get("SMITH_CACHE_REGION") or None,
            path_style=None if style is None else style.lower() in ("1", "true", "yes"),
            access_key=access,
            secret_key=secret,
        )
    if address.startswith(("sign+https://", "sign+http://")):
        env = os.environ if environ is None else environ
        token = env.get("SMITH_CACHE_TOKEN")
        if not token:
            raise ValueError("a signing service needs SMITH_CACHE_TOKEN")
        return SignedStore(address[len("sign+"):], token)
    if address.startswith("file://"):
        address = address[len("file://"):]
    return LocalStore(address)


def describe(outcome: Outcome, action: str, key: Key) -> str:
    if action == "pull" and not outcome.hit:
        return "cache %s %s: miss (%s)" % (key.fields["name"], key.id, outcome.reason)
    verb = "got" if action == "pull" else "sent"
    return "cache %s %s: %s, %s %d file(s), %d bytes" % (
        key.fields["name"], key.id, "hit" if outcome.hit else "new", verb, outcome.files, outcome.bytes
    )


def libraries(values: Iterable[str]) -> List[str]:
    """``--with`` values, each one or several comma-separated names."""
    return sorted({name.strip() for value in values for name in value.split(",") if name.strip()})


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cache.py",
        description="fill a directory of compiled wheels or kernels from a store, or send it back",
    )
    parser.add_argument("action", choices=("key", "pull", "push"))
    parser.add_argument("--name", required=True, help="which directory: wheels, triton, inductor, tilelang...")
    parser.add_argument("--dir", default=None, help="the directory to fill or send (pull, push)")
    parser.add_argument("--store", default=None,
                        help="where the cache lives: s3://bucket/prefix, sign+https://service, or a directory (pull, push)")
    parser.add_argument("--with", dest="libraries", action="append", default=[],
                        help="a library whose version is part of the key; repeat, or comma-separate")
    args = parser.parse_args(argv)
    if args.action != "key" and not (args.dir and args.store):
        parser.error("%s needs --dir and --store" % args.action)

    key = compute_key(args.name, libraries(args.libraries))
    if args.action == "key":
        print(json.dumps({"id": key.id, **key.fields}, indent=2, sort_keys=True))
        return 0
    try:
        store = open_store(args.store)
        run = pull if args.action == "pull" else push
        outcome = run(store, key, args.dir)
    except (ValueError, OSError) as exc:
        # A cache is never worth failing the work for: the caller compiles
        # what it would have downloaded, and the line says why.
        print("cache %s: %s" % (args.action, exc), file=sys.stderr)
        return 1
    print(describe(outcome, args.action, key))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
