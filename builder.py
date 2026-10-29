"""builder: compiles, from PyPI's source, a wheel nodes had to compile, for the cache every node reads.

    python builder.py            # one build from the backend's queue, if there is one
    python builder.py --pending  # only says whether there is one, without taking it

Run by ``.github/workflows/cache.yml`` (its ``wheels`` job) inside the GPU
image, which is what the nodes run: the wheel is built against the image's Python and torch, and those
are part of the key a node looks it up by.

**Why a builder and not the nodes.** smith reports, after installing a job's
dependencies, the wheels uv had to build from source there (``SMITH_CACHE =
backend``). A node runs its owner's code, so nothing a node made can go where
every other node reads: a wheel from a node is whatever that code wanted it to
be. So the node reports a name - package, version, its ``cache.py key
--name wheels`` - and this compiles the package again from the sdist PyPI
publishes, checked against the SHA-256 PyPI publishes for it.

**What it does, for one build the backend hands it**:

1. checks that this machine's key, with the build's GPUs in place of its own
   (it has none), is the build's key field by field: a wheel built against
   another torch or Python would be stored where nodes with that one look;
2. downloads the sdist and checks its hash;
3. compiles it with ``pip wheel --no-build-isolation``, so against the image's
   own torch rather than whichever one a build environment would download, for
   the key's GPUs only (``TORCH_CUDA_ARCH_LIST``, ``FLASH_ATTN_CUDA_ARCHS``),
   as many compilations at once as the memory holds: one nvcc of flash-attn
   takes 6.5 GB (GPU-202);
4. puts the wheel and the key in the cache with ``cache.py``'s own push,
   through URLs the backend signs for this build alone;
5. tells the backend how it ended.

``SMITH_BACKEND`` is the backend, ``SMITH_BUILDER_TOKEN`` its builder token -
not an agent's: it may write what no node may. Standard library and
``cache.py`` beside it only.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.error
import urllib.request
from typing import Any, Deque, Dict, List, Optional

#: The memory one compilation takes, for how many run at once: one nvcc of
#: flash-attn peaked at 6.5 GB, and two in 7 GB were killed (GPU-202).
GB_PER_JOB = 7

#: The tail of the build's output the backend keeps when it fails.
TAIL = 4000


def ask(path: str, body: Optional[Dict[str, Any]] = None) -> Any:
    request = urllib.request.Request(
        os.environ["SMITH_BACKEND"].rstrip("/") + path,
        data=json.dumps(body or {}).encode(),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ["SMITH_BUILDER_TOKEN"],
            "User-Agent": "smith-builder",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode() or "null")


def sdist(package: str, version: str, folder: str) -> str:
    """The sdist PyPI publishes for ``package`` ``version``, downloaded into
    ``folder`` and checked against the hash PyPI gives for it."""
    url = "https://pypi.org/pypi/%s/%s/json" % (package, version)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "smith-builder"}), timeout=60) as response:
        release = json.loads(response.read().decode())
    files = [f for f in release.get("urls", []) if f.get("packagetype") == "sdist" and not f.get("yanked")]
    if len(files) != 1:
        raise RuntimeError("PyPI has %d sdists of %s %s, not one" % (len(files), package, version))
    file = files[0]
    expected = file["digests"]["sha256"]
    name = os.path.basename(file["filename"])
    path = os.path.join(folder, name)
    digest = hashlib.sha256()
    with urllib.request.urlopen(urllib.request.Request(file["url"], headers={"User-Agent": "smith-builder"}), timeout=600) as response, \
            open(path, "wb") as out:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError("%s is not what PyPI says: sha256 %s, not %s" % (name, digest.hexdigest(), expected))
    print("[builder] %s, sha256 %s as PyPI says" % (name, expected), flush=True)
    return path


def build_requires(path: str) -> List[str]:
    """What the sdist's ``pyproject.toml`` asks to be built with, torch left
    out: the image's is the one the wheel must be built against."""
    if not path.endswith((".tar.gz", ".tgz")):
        return []
    with tarfile.open(path) as archive:
        member = next((m for m in archive.getmembers() if m.name.count("/") == 1 and m.name.endswith("/pyproject.toml")), None)
        if member is None:
            return []
        handle = archive.extractfile(member)
        project = tomllib.loads(handle.read().decode()) if handle else {}
    requires = project.get("build-system", {}).get("requires", [])
    return [r for r in requires if re.split(r"[\s<>=!~;\[]", r.strip(), maxsplit=1)[0].lower() != "torch"]


def arch_env(gpus: List[str]) -> Dict[str, str]:
    """Compile for the key's GPUs alone: ``sm_86`` is ``8.6`` to torch's
    extensions and ``86`` to flash-attn's own list."""
    torch_list, flash = [], []
    for gpu in gpus:
        number = re.match(r"sm_(\d+)([a-z]?)$", gpu)
        if not number:
            raise RuntimeError("%r is not a GPU this builder knows" % gpu)
        digits, suffix = number.groups()
        torch_list.append("%s.%s%s" % (digits[:-1], digits[-1], suffix))
        flash.append(digits)
    memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    jobs = max(1, min(os.cpu_count() or 1, int(memory // GB_PER_JOB)))
    return {
        "TORCH_CUDA_ARCH_LIST": ";".join(torch_list),
        "FLASH_ATTN_CUDA_ARCHS": ";".join(flash),
        # flash-attn otherwise downloads a wheel from its own releases: bytes
        # that are not the source this checked.
        "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
        "MAX_JOBS": str(jobs),
    }


def build(item: Dict[str, Any], work: str) -> str:
    """The wheel of ``item``, in ``work/out``; its file name."""
    import cache as _cache

    key = item["key"]
    mine = _cache.compute_key("wheels", gpus=key["gpus"])
    differ = mine.differences(key)
    if differ:
        raise RuntimeError("this image builds for another key: it differs in %s (%s, not %s)" % (
            ", ".join(differ), {n: mine.fields.get(n) for n in differ}, {n: key.get(n) for n in differ}))
    source = sdist(item["package"], item["version"], work)
    env = dict(os.environ, **arch_env(key["gpus"]))
    env.pop("SMITH_BUILDER_TOKEN", None)
    print("[builder] %s" % {k: env[k] for k in ("TORCH_CUDA_ARCH_LIST", "FLASH_ATTN_CUDA_ARCHS", "MAX_JOBS")}, flush=True)
    constraints = os.path.join(work, "constraints.txt")
    with open(constraints, "w", encoding="utf-8") as handle:
        # A build requirement that would bring another torch is refused
        # rather than allowed to replace the one the key names.
        handle.write("torch==%s\n" % key["torch"] if key["torch"] else "")
    requires = ["setuptools", "wheel", "ninja", "packaging", "psutil"] + build_requires(source)
    run([sys.executable, "-m", "pip", "install", "-q", "-c", constraints] + requires, env)
    out = os.path.join(work, "out")
    run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w", out, source], env)
    wheels = [name for name in os.listdir(out) if name.endswith(".whl")]
    if len(wheels) != 1:
        raise RuntimeError("the build made %d wheels, not one: %s" % (len(wheels), wheels))
    return wheels[0]


def run(command: List[str], env: Dict[str, str]) -> None:
    """Run ``command``, its output passed on as it comes: a compilation of
    hours is otherwise silent until it ends. Its tail is the error."""
    print("[builder] " + " ".join(command), flush=True)
    tail: Deque[str] = collections.deque(maxlen=60)
    with subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, errors="replace") as process:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            tail.append(line)
    if process.returncode != 0:
        raise RuntimeError("%s exited %d:\n%s" % (" ".join(command[1:4]), process.returncode, "".join(tail)[-TAIL:]))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pending", action="store_true", help="only say whether there is a build to take")
    args = parser.parse_args(argv)
    if not os.environ.get("SMITH_BACKEND") or not os.environ.get("SMITH_BUILDER_TOKEN"):
        print("[builder] no SMITH_BACKEND or SMITH_BUILDER_TOKEN: nothing to build from", file=sys.stderr)
        if args.pending:
            print("pending=0")
        return 0
    if args.pending:
        queued = int(ask("/api/cache-builds/pending").get("queued") or 0)
        print("[builder] %d queued" % queued, file=sys.stderr)
        print("pending=%d" % min(queued, 1))
        return 0
    item = ask("/api/cache-builds/claim").get("build")
    if not item:
        print("[builder] nothing queued")
        return 0
    print("[builder] build %d: %s %s for %s" % (item["id"], item["package"], item["version"], item["key"]), flush=True)
    with tempfile.TemporaryDirectory() as work:
        try:
            wheel = build(item, work)
            import cache as _cache

            store = _cache.SignedStore("%s/api/cache-builds/%d/sign" % (os.environ["SMITH_BACKEND"].rstrip("/"), item["id"]),
                                       os.environ["SMITH_BUILDER_TOKEN"])
            outcome = _cache.push(store, _cache.Key(dict(item["key"])), os.path.join(work, "out"))
            print("[builder] " + _cache.describe(outcome, "push", _cache.Key(dict(item["key"]))), flush=True)
        except Exception as exc:  # what failed goes to the backend, whatever it was
            ask("/api/cache-builds/%d/done" % item["id"], {"ok": False, "message": str(exc)[-TAIL:]})
            print("[builder] failed: %s" % exc, file=sys.stderr)
            return 1
    print(json.dumps(ask("/api/cache-builds/%d/done" % item["id"], {"ok": True, "wheel": wheel})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
