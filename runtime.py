"""runtime: the heavy part of a GPU node's Python, downloaded when the node starts instead of carried in its image.

    python runtime.py lock    --report <pip report> --out <manifest>
    python runtime.py fetch   --manifest <manifest> --dir <directory> [--mirror <url>]
    python runtime.py install --manifest <manifest> [--mirror <url>] [--python <interpreter>]

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

**Where from.** A mirror if one is given - ``--mirror`` or
``SMITH_RUNTIME_URL``, a base address under which each wheel is at its own
file name: a public bucket close to every provider - and otherwise, or for a
file the mirror does not have, the address the manifest resolved it from.
A wheel's name and content are fixed by whoever published it, so one copy in
the mirror serves every image that names it, and filling the mirror is
uploading the names it lacks.

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

TIMEOUT = 60

#: Some CDNs refuse urllib's own User-Agent.
AGENT = "smith-runtime/1"


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


def _open(url: str, first: Optional[int] = None, last: Optional[int] = None):
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    if first is not None:
        request.add_header("Range", "bytes=%d-%d" % (first, last))
    return urllib.request.urlopen(request, timeout=TIMEOUT)


def _size(url: str) -> int:
    """The file's size, from the first byte's range answer: ``Content-Range``
    says the whole size, and a HEAD is one more thing a CDN may refuse."""
    with _open(url, 0, 0) as response:
        if response.status != 206:
            raise Unranged(url)
        total = response.headers.get("Content-Range", "").rpartition("/")[2]
        if not total.isdigit():
            raise Unranged(url)
        return int(total)


def _piece(url: str, path: str, first: int, last: int) -> None:
    """Bytes ``first`` to ``last`` of ``url`` into ``path``, at their place."""
    error: Optional[BaseException] = None
    for attempt in range(TRIES):
        try:
            with _open(url, first, last) as response:
                if response.status != 206:
                    raise Unranged(url)
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
    raise OSError("bytes %d-%d of %s: %s" % (first, last, url, error))


def _whole(url: str, path: str) -> None:
    """A source without ranges: the file in one request, retried whole."""
    error: Optional[BaseException] = None
    for attempt in range(TRIES):
        try:
            with _open(url) as response, open(path, "wb") as out:
                shutil.copyfileobj(response, out, 1024 * 1024)
            return
        except (OSError, urllib.error.URLError) as exc:
            error = exc
            time.sleep(min(2 ** attempt, 10))
    raise OSError("%s: %s" % (url, error))


def _hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sources(entry: Dict[str, Any], mirror: Optional[str]) -> List[str]:
    sources = []
    if mirror:
        sources.append(mirror.rstrip("/") + "/" + urllib.parse.quote(entry["name"]))
    sources.append(entry["url"])
    return sources


def fetch(manifest: Dict[str, Any], directory: str, mirror: Optional[str] = None) -> Dict[str, Any]:
    """Every file of the manifest into ``directory``, each checked against
    its sha256. A file already there with the right hash is kept. Returns
    what it did, for the log: bytes, seconds, and which files came from where."""
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

    # One pool for everything: the pieces of every file go into it together,
    # so a 1 GB wheel and forty small ones fill the same TRANSFERS
    # connections, never more. A file whose source failed - or served bytes
    # that do not hash to the manifest's - goes round again from its next
    # source.
    pending = [(entry, _sources(entry, mirror)) for entry in wanted]
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
                    origin[entry["name"]] = sources[0]
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
    return {"files": len(wanted), "bytes": fetched, "seconds": seconds, "from": origin}


def _plan(url: str) -> Any:
    """How a source serves a file: its size when it takes ranges, None when
    it only sends it whole, or the error that says it cannot serve it."""
    try:
        return _size(url)
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


def install(manifest_path: str, mirror: Optional[str] = None, python: Optional[str] = None) -> bool:
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
        done = fetch(manifest, directory, mirror)
        megabytes = done["bytes"] / 1e6
        mirrored = sum(1 for url in done["from"].values() if mirror and url.startswith(mirror.rstrip("/")))
        say("fetched %d files, %.0f MB in %.1f s (%.0f MB/s), %d from the mirror" % (
            done["files"], megabytes, done["seconds"], megabytes / done["seconds"], mirrored))
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="The heavy part of a GPU node's Python, fetched at start.")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("lock", help="a manifest from a pip installation report")
    one.add_argument("--report", required=True)
    one.add_argument("--out", required=True)
    one = sub.add_parser("fetch", help="the manifest's files into a directory")
    one.add_argument("--manifest", required=True)
    one.add_argument("--dir", required=True)
    one.add_argument("--mirror", default=os.environ.get("SMITH_RUNTIME_URL") or None)
    one = sub.add_parser("install", help="fetch and install the manifest's set")
    one.add_argument("--manifest", required=True)
    one.add_argument("--mirror", default=os.environ.get("SMITH_RUNTIME_URL") or None)
    one.add_argument("--python", default=None)
    args = parser.parse_args(argv)
    if args.command == "lock":
        manifest = lock(args.report, args.out)
        say("%d wheels locked in %s" % (len(manifest["files"]), args.out))
    elif args.command == "fetch":
        done = fetch(read_manifest(args.manifest), args.dir, args.mirror)
        say("fetched %d files, %.0f MB in %.1f s" % (done["files"], done["bytes"] / 1e6, done["seconds"]))
    else:
        install(args.manifest, args.mirror, args.python)
    return 0


if __name__ == "__main__":
    sys.exit(main())
