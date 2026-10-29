"""runtime: the heavy part of a GPU node's Python, downloaded when the node starts instead of carried in its image.

    python runtime.py lock    --report <pip report> --out <manifest>
    python runtime.py fetch   --manifest <manifest> --dir <directory> [--signed <address>] [--mirror <url>]
    python runtime.py install --manifest <manifest> [--signed <address>] [--mirror <url>] [--python <interpreter>]

**Why not in the image.** torch with its CUDA libraries is about 7 GB, and in
the image it was layers of about a gigabyte each. A rented machine pulls the
image with its own docker, from the registry, before smith can say anything:
on 2026-10-29 two marketplace hosts in a row spent more than ten minutes on
it, a layer dropping halfway and starting again from nothing
("Retrying in 5 seconds"), and never started. Nothing here can make another
host's docker resume a layer. So the image keeps what is small and stays the
same - Python, smith, the libraries smith and the runs need to start, the
compiler - and the rest arrives here, where every download is ours to make:
in pieces, several at once, each piece retried on its own.

**The manifest is the lock.** One JSON file, made when the image is built
(``lock``, from ``pip install --dry-run --report``) and carried in it: for
each wheel its file name, the address it was resolved from, and its sha256;
the size is asked of the source. The image decides exactly what is installed; a node only decides
where to fetch it from. A wheel whose bytes do not hash to the manifest's is
refused, whatever served it.

**Where from.** Each file from the first of these that has it, and from the
next one when a source fails or lacks it:

1. The backend's cache, when the node uses it (``SMITH_CACHE=backend``, and
   smith passes ``--signed <address>``, the signing service of the cache's
   read-only ``global`` scope). Each wheel is at ``runtime/<file name>``
   there, and the node holds no key to the bucket: it asks the service, with
   the token in ``SMITH_CACHE_TOKEN`` - from the environment, never the
   command line, where any user of the machine can read it - for a URL that
   reads that one file, the protocol of ``cache.py``'s ``SignedStore``. Who
   fills the cache is the backend's business: this public repository's CI
   holds no bucket key.
2. A mirror, if one is given - ``--mirror`` or ``SMITH_RUNTIME_URL``, a base
   address under which each wheel is at its own file name - for whoever runs
   smith without such a backend and hosts a plain one.
3. The address the manifest resolved it from.

A wheel's name and content are fixed by whoever published it, so one copy in
the cache or the mirror serves every image that names it, and filling either
is uploading the names it lacks.

**A signed URL expires** (the service says after how long; minutes), and a
slow host may still be fetching a gigabyte wheel when it does. So a file's URL
is signed when the file is first asked for, not all of them up front, and a
piece the bucket refuses with 403 - or 400, which some stores answer for an
expired signature - has its file's URL signed again and is asked again, a
bounded number of times (``RESIGNS``), instead of failing the file.

**How.** Every file is split into pieces of ``PIECE`` bytes, and ``TRANSFERS``
pieces of all the files are fetched at once, each with an HTTP range request,
each retried on its own. A file whose source ignores ranges is fetched whole
instead. Then each file is hashed, and the whole set is installed with uv
(pip where uv is missing), ``--no-deps`` and from these files only: the
manifest already is the resolution, and nothing else is asked of any index.

Only the standard library, like smith beside it: this runs before the heavy
part exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

#: Bytes per range request. Small enough that a dropped connection costs
#: little, large enough that a 3 GB set is a few hundred requests.
PIECE = 32 * 1024 * 1024

#: Range requests in flight at once, across all the files.
TRANSFERS = 16

#: Tries per piece before its source is given up for that file.
TRIES = 5

#: Times a file's signed URL is signed again after the bucket refused it as
#: expired. Two or three cover the slowest host that is worth keeping; the
#: bound is for a service whose URLs the bucket always refuses, which would
#: otherwise be signed again for ever.
RESIGNS = 5

TIMEOUT = 60

#: Some CDNs refuse urllib's own User-Agent.
AGENT = "smith-runtime/1"

#: Where each wheel is in the backend's cache, by its file name.
CACHE_PREFIX = "runtime/"


def say(message: str) -> None:
    print("[runtime] " + message, flush=True)


# ─── the manifest ─────────────────────────────────────────────────────


def lock(report_path: str, out_path: str) -> Dict[str, Any]:
    """The manifest from a pip installation report (``pip install --dry-run
    --ignore-installed --report``): one entry per wheel it resolved."""
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    files = []
    for item in report.get("install", []):
        info = item.get("download_info") or {}
        url = info.get("url") or ""
        hashes = (info.get("archive_info") or {}).get("hashes") or {}
        digest = hashes.get("sha256") or (info.get("archive_info") or {}).get("hash", "").partition("=")[2]
        name = urllib.parse.unquote(url.rsplit("/", 1)[-1].split("#", 1)[0])
        if not name.endswith(".whl"):
            raise SystemExit("not a wheel, which a node would have to build: %s" % url)
        if not digest:
            raise SystemExit("no sha256 for %s: the index gave none, and an unchecked file is not installed" % name)
        files.append({"name": name, "url": url.split("#", 1)[0], "sha256": digest})
    if not files:
        raise SystemExit("the report resolved nothing")
    manifest = {"version": 1, "files": sorted(files, key=lambda entry: entry["name"])}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
        handle.write("\n")
    return manifest


def read_manifest(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("version") != 1 or not manifest.get("files"):
        raise SystemExit("%s is not a runtime manifest this smith reads" % path)
    return manifest


def manifest_id(manifest: Dict[str, Any]) -> str:
    """What a set is, for knowing it is already installed: its hashes."""
    joined = "\n".join(entry["sha256"] for entry in manifest["files"])
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


# ─── fetching ─────────────────────────────────────────────────────────


class Unranged(Exception):
    """The source answered a range request with the whole file."""


class Signer:
    """The backend's signing service for its cache (``cache.py``'s
    ``SignedStore`` speaks the same protocol): one ``POST`` per file, which
    answers a URL that reads it straight from the bucket. Only ``get`` is
    ever asked: a node reads the runtime and never writes it.

    Its own few lines rather than an import of ``cache.py``: the image's
    build runs this file alone, before anything beside it is copied."""

    def __init__(self, address: str, token: str) -> None:
        self.address = address
        self.token = token

    def sign(self, path: str) -> str:
        request = urllib.request.Request(
            self.address,
            data=json.dumps({"op": "get", "path": path}).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.token, "User-Agent": AGENT},
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                answer = json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            # A service without the file, or without a cache at all, is a
            # source that lacks it: the next one is asked, as for a mirror.
            raise OSError("the cache refused to sign %s: %d" % (path, exc.code)) from None
        except ValueError:
            raise OSError("the cache answered %s with something not JSON" % path) from None
        url = answer.get("url") if isinstance(answer, dict) else None
        if not isinstance(url, str) or not url.startswith(("https://", "http://")):
            raise OSError("the cache gave no URL for %s" % path)
        return url


class Plain:
    """A source at a fixed address: a mirror, or the index."""

    def __init__(self, address: str, kind: str) -> None:
        self.address = address
        self.kind = kind

    def url(self) -> str:
        return self.address

    def renew(self, stale: str, error: urllib.error.HTTPError) -> bool:
        return False

    def __str__(self) -> str:
        return self.address


class Signed:
    """A file in the backend's cache, at a URL signed when it is first asked
    for and signed again when the bucket says it expired. Shared by all the
    pieces of its file, which run in different threads: the first piece to
    find the URL expired signs it again, and the others, refused with the
    same stale URL, take the new one instead of each signing their own."""

    kind = "cache"

    def __init__(self, signer: Signer, path: str) -> None:
        self.signer = signer
        self.path = path
        self._url: Optional[str] = None
        self._renewals = 0
        self._lock = threading.Lock()

    def url(self) -> str:
        with self._lock:
            if self._url is None:
                self._url = self.signer.sign(self.path)
            return self._url

    def renew(self, stale: str, error: urllib.error.HTTPError) -> bool:
        """Whether to ask again, at a URL signed anew: for a refusal that
        an expired signature gives, while this file has renewals left."""
        if error.code not in (400, 403):
            return False
        with self._lock:
            if self._url != stale:
                return True
            if self._renewals >= RESIGNS:
                return False
            self._renewals += 1
            self._url = self.signer.sign(self.path)
            return True

    def __str__(self) -> str:
        # Never the signed URL itself, which reads the file until it expires.
        return "the cache's " + self.path


Source = Any  # Plain or Signed


def _open(source: Source, first: Optional[int] = None, last: Optional[int] = None):
    """The answer of ``source`` for a range, or the whole file; a signed URL
    the bucket refused is signed again here, so no caller has to know."""
    while True:
        url = source.url()
        request = urllib.request.Request(url, headers={"User-Agent": AGENT})
        if first is not None:
            request.add_header("Range", "bytes=%d-%d" % (first, last))
        try:
            return urllib.request.urlopen(request, timeout=TIMEOUT)
        except urllib.error.HTTPError as exc:
            exc.close()
            if not source.renew(url, exc):
                raise


def _size(source: Source) -> int:
    """The file's size, from the first byte's range answer: ``Content-Range``
    says the whole size, and a HEAD is one more thing a CDN may refuse."""
    with _open(source, 0, 0) as response:
        if response.status != 206:
            raise Unranged(str(source))
        total = response.headers.get("Content-Range", "").rpartition("/")[2]
        if not total.isdigit():
            raise Unranged(str(source))
        return int(total)


def _piece(source: Source, path: str, first: int, last: int) -> None:
    """Bytes ``first`` to ``last`` of ``source`` into ``path``, at their place."""
    error: Optional[BaseException] = None
    for attempt in range(TRIES):
        try:
            with _open(source, first, last) as response:
                if response.status != 206:
                    raise Unranged(str(source))
                with open(path, "r+b") as out:
                    out.seek(first)
                    remaining = last - first + 1
                    while remaining:
                        chunk = response.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            raise OSError("the connection closed %d bytes short" % remaining)
                        out.write(chunk)
                        remaining -= len(chunk)
            return
        except Unranged:
            raise
        except (OSError, urllib.error.URLError) as exc:
            error = exc
            time.sleep(min(2 ** attempt, 10))
    raise OSError("bytes %d-%d of %s: %s" % (first, last, source, error))


def _whole(source: Source, path: str) -> None:
    """A source without ranges: the file in one request, retried whole."""
    error: Optional[BaseException] = None
    for attempt in range(TRIES):
        try:
            with _open(source) as response, open(path, "wb") as out:
                shutil.copyfileobj(response, out, 1024 * 1024)
            return
        except (OSError, urllib.error.URLError) as exc:
            error = exc
            time.sleep(min(2 ** attempt, 10))
    raise OSError("%s: %s" % (source, error))


def _hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sources(entry: Dict[str, Any], mirror: Optional[str], signer: Optional[Signer]) -> List[Source]:
    sources: List[Source] = []
    if signer is not None:
        sources.append(Signed(signer, CACHE_PREFIX + entry["name"]))
    if mirror:
        sources.append(Plain(mirror.rstrip("/") + "/" + urllib.parse.quote(entry["name"]), "mirror"))
    sources.append(Plain(entry["url"], "index"))
    return sources


def fetch(
    manifest: Dict[str, Any], directory: str, mirror: Optional[str] = None, signer: Optional[Signer] = None
) -> Dict[str, Any]:
    """Every file of the manifest into ``directory``, each checked against
    its sha256. A file already there with the right hash is kept. Returns
    what it did, for the log: bytes, seconds, and which files came from
    where - ``from`` the source of each, ``kinds`` whether that was the
    cache, the mirror or the index."""
    os.makedirs(directory, exist_ok=True)
    started = time.monotonic()
    wanted = []
    for entry in manifest["files"]:
        path = os.path.join(directory, entry["name"])
        if os.path.exists(path) and _hash(path) == entry["sha256"]:
            continue
        wanted.append(entry)
    fetched = 0
    origin: Dict[str, str] = {}
    kinds: Dict[str, str] = {}

    # One pool for everything: the pieces of every file go into it together,
    # so a 1 GB wheel and forty small ones fill the same TRANSFERS
    # connections, never more. A file whose source failed - or served bytes
    # that do not hash to the manifest's - goes round again from its next
    # source.
    # A cache's URL is signed in _plan, so only for a file still wanted and
    # only when it is about to be fetched.
    pending = [(entry, _sources(entry, mirror, signer)) for entry in wanted]
    with ThreadPoolExecutor(max_workers=TRANSFERS, thread_name_prefix="smith-runtime") as pool:
        while pending:
            plans = list(zip(pending, pool.map(lambda item: _plan(item[1][0]), pending)))
            work: Dict[str, List[Any]] = {}
            for (entry, sources), size in plans:
                path = os.path.join(directory, entry["name"])
                if isinstance(size, BaseException):
                    work[entry["name"]] = [size]
                elif size is None:
                    work[entry["name"]] = [pool.submit(_whole, sources[0], path)]
                else:
                    with open(path, "wb") as out:
                        out.truncate(size)
                    work[entry["name"]] = [
                        pool.submit(_piece, sources[0], path, first, min(first + PIECE, size) - 1)
                        for first in range(0, size, PIECE)
                    ]
            again = []
            for (entry, sources), _ in plans:
                path = os.path.join(directory, entry["name"])
                try:
                    for future in work[entry["name"]]:
                        if isinstance(future, BaseException):
                            raise future
                        future.result()
                    if _hash(path) != entry["sha256"]:
                        raise OSError("its sha256 is not the manifest's")
                    fetched += os.path.getsize(path)
                    origin[entry["name"]] = str(sources[0])
                    kinds[entry["name"]] = sources[0].kind
                except (OSError, urllib.error.URLError, Unranged) as exc:
                    if os.path.exists(path):
                        os.remove(path)
                    if len(sources) > 1:
                        say("%s from %s failed (%s); trying %s" % (entry["name"], sources[0], exc, sources[1]))
                        again.append((entry, sources[1:]))
                    else:
                        raise OSError("%s could not be fetched: %s" % (entry["name"], exc)) from None
            pending = again
    seconds = max(time.monotonic() - started, 1e-6)
    return {"files": len(wanted), "bytes": fetched, "seconds": seconds, "from": origin, "kinds": kinds}


def _plan(source: Source) -> Any:
    """How a source serves a file: its size when it takes ranges, None when
    it only sends it whole, or the error that says it cannot serve it - a
    cache that would not sign it among them."""
    try:
        return _size(source)
    except Unranged:
        return None
    except (OSError, urllib.error.URLError) as exc:
        return exc


# ─── installing ───────────────────────────────────────────────────────


def _marker(python: str) -> str:
    """Where an installed set is remembered: beside the interpreter's own
    packages, so a new container from the image - an empty disk - has none."""
    purelib = subprocess.run(
        [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return os.path.join(purelib, ".smith-runtime")


def install(
    manifest_path: str, mirror: Optional[str] = None, python: Optional[str] = None, signer: Optional[Signer] = None
) -> bool:
    """Fetch and install the manifest's set for ``python``, unless it is
    already there. Returns whether anything was installed."""
    manifest = read_manifest(manifest_path)
    python = python or sys.executable
    marker = _marker(python)
    wanted = manifest_id(manifest)
    if os.path.exists(marker) and open(marker, encoding="utf-8").read().strip() == wanted:
        say("set %s already installed" % wanted)
        return False
    directory = tempfile.mkdtemp(prefix="smith-runtime-")
    try:
        done = fetch(manifest, directory, mirror, signer)
        megabytes = done["bytes"] / 1e6
        say("fetched %d files, %.0f MB in %.1f s (%.0f MB/s): %s" % (
            done["files"], megabytes, done["seconds"], megabytes / done["seconds"], _whence(done)))
        paths = [os.path.join(directory, entry["name"]) for entry in manifest["files"]]
        started = time.monotonic()
        uv = shutil.which("uv")
        if uv:
            command = [uv, "pip", "install", "--python", python, "--no-deps", "--no-index", *paths]
        else:
            command = [python, "-m", "pip", "install", "--no-deps", "--no-index", *paths]
        subprocess.run(command, check=True)
        say("installed in %.1f s" % (time.monotonic() - started))
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write(wanted + "\n")
    return True


def _whence(done: Dict[str, Any]) -> str:
    """How many files each kind of source gave, for the log: a cache that
    gave nothing is the first thing to see when a node fetched slowly."""
    counts = {kind: 0 for kind in ("cache", "mirror", "index")}
    for kind in done["kinds"].values():
        counts[kind] += 1
    return "%d from the cache, %d from the mirror, %d from the index" % (
        counts["cache"], counts["mirror"], counts["index"])


def _signer(address: Optional[str]) -> Optional[Signer]:
    """The cache's signing service at ``address``, with the token from the
    environment: on the command line any user of the machine could read it."""
    if not address:
        return None
    token = os.environ.get("SMITH_CACHE_TOKEN")
    if not token:
        raise SystemExit("--signed needs the service's token in SMITH_CACHE_TOKEN")
    return Signer(address, token)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="The heavy part of a GPU node's Python, fetched at start.")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("lock", help="a manifest from a pip installation report")
    one.add_argument("--report", required=True)
    one.add_argument("--out", required=True)
    for name, text in (("fetch", "the manifest's files into a directory"), ("install", "fetch and install the manifest's set")):
        one = sub.add_parser(name, help=text)
        one.add_argument("--manifest", required=True)
        if name == "fetch":
            one.add_argument("--dir", required=True)
        else:
            one.add_argument("--python", default=None)
        # Only when asked for, never from the environment alone: whether a
        # node uses the backend's cache is smith's decision (SMITH_CACHE).
        one.add_argument("--signed", default=None, help="the signing service of the cache's runtime/ wheels")
        one.add_argument("--mirror", default=os.environ.get("SMITH_RUNTIME_URL") or None)
    args = parser.parse_args(argv)
    if args.command == "lock":
        manifest = lock(args.report, args.out)
        say("%d wheels locked in %s" % (len(manifest["files"]), args.out))
    elif args.command == "fetch":
        done = fetch(read_manifest(args.manifest), args.dir, args.mirror, _signer(args.signed))
        say("fetched %d files, %.0f MB in %.1f s: %s" % (
            done["files"], done["bytes"] / 1e6, done["seconds"], _whence(done)))
    else:
        install(args.manifest, args.mirror, args.python, _signer(args.signed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
