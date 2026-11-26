"""The environment a job trains in, made from its template (GPU-523).

A job says what it trains with - ``{python, torch, ravex, moonclip}``, the
exact versions - and this makes that environment on the node, once: a
Python of that version, PyTorch from the image's lock for it (or the CPU
build from PyTorch's index), then Ravex and Moonclip and what Ravex reads
the machine with. The image carries none of it but uv and the locks, so a
release of Ravex or Moonclip needs no new image.

**One Python per template.** uv installs that version into the template's own
folder, and the packages go into its own ``site-packages``, not a virtual
environment: a repository's requirements are then a virtual environment on
top of it with ``--system-site-packages``, as they were on top of the image's
Python, so they see torch and Ravex and add to them.

**Made once, known by a marker.** A folder half made - a node lost halfway,
a failed install - has no marker and is made again from nothing. Two jobs
making the same template at the same moment on one machine would race;
smith runs one job at a time (shortcut: no lock, add one if an agent ever
runs two jobs at once).

**Ravex's scripts.** A script of the agent's own named ``ravex:<path>`` is a
file of that Ravex release's source - its examples and benches, which the
wheel does not carry - fetched once per release from its tag.

Standard library only, like smith; run as a script with smith's own Python:
``environment.py ensure --template '<json>' --root <dir> --out <file>``
writes ``{"python": ...}`` there, ``tree`` the release's folder.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from typing import Any, Dict, List, Optional

#: A version as the backend sends one: digits, dots, and a local part.
_VERSION = re.compile(r"[0-9][0-9A-Za-z.+_-]{0,31}")
_PYTHON = re.compile(r"3\.[0-9]{1,2}")
RAVEX_SOURCE = "https://github.com/gpuzero-dev/ravex/archive/refs/tags/v%s.tar.gz"
#: Where the CPU build of PyTorch is published; PyPI's own is CUDA's.
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def say(message: str) -> None:
    print("[smith] " + message, flush=True)


def checked(template: Dict[str, Any]) -> Dict[str, str]:
    """The template's four versions, each held to what a version can be: they
    become folder names and arguments to uv."""
    out = {}
    for key in ("python", "torch", "ravex", "moonclip"):
        value = str(template.get(key) or "")
        pattern = _PYTHON if key == "python" else _VERSION
        if not pattern.fullmatch(value):
            raise ValueError("the job's %s version %r is not one" % (key, value))
        out[key] = value
    return out


def folder(root: str, template: Dict[str, str]) -> str:
    name = "py{python}-torch{torch}-ravex{ravex}-moonclip{moonclip}".format(**template)
    return os.path.join(root, re.sub(r"[^A-Za-z0-9_.-]", "_", name))


def lock_for(locks: Optional[str], python: str) -> Optional[str]:
    """The image's lock of PyTorch for this Python, ``cp312.json`` and the like."""
    if not locks:
        return None
    path = os.path.join(locks, "cp%s.json" % python.replace(".", ""))
    return path if os.path.isfile(path) else None


def _run(command: List[str], what: str, env: Optional[Dict[str, str]] = None) -> str:
    done = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, env=env)
    if done.returncode != 0:
        said = [line for line in (done.stderr or done.stdout or "").splitlines() if line.strip()]
        raise RuntimeError("could not %s: %s" % (what, said[-1] if said else "no answer"))
    return done.stdout


def ensure(
    template: Dict[str, Any],
    root: str,
    locks: Optional[str] = None,
    runtime_script: Optional[str] = None,
    signed: Optional[str] = None,
    gpu: bool = False,
) -> str:
    """The Python of the template's environment, made first if it is not
    there. ``locks`` is the image's folder of PyTorch locks, one per Python;
    without one PyTorch's CPU build is installed from its index. ``signed``
    is the backend's cache of the locks' wheels, passed to ``runtime.py``."""
    versions = checked(template)
    home = folder(root, versions)
    marker = os.path.join(home, ".smith-ready")
    if os.path.isfile(marker):
        with open(marker, encoding="utf-8") as handle:
            python = json.load(handle)["python"]
        say("libraries ready: Python %(python)s, PyTorch %(torch)s, Ravex %(ravex)s, Moonclip %(moonclip)s" % versions)
        return python
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is not on this machine; it makes the job's environment")
    lock = lock_for(locks, versions["python"])
    if locks and lock is None:
        raise RuntimeError("this machine has no PyTorch for Python %s" % versions["python"])
    began = time.monotonic()
    shutil.rmtree(home, ignore_errors=True)
    os.makedirs(home)
    # A Python of its own, so what this template installs is nobody else's.
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=os.path.join(home, "python"))
    say("making Python %s" % versions["python"])
    _run([uv, "python", "install", versions["python"]], "install Python %s" % versions["python"], env)
    python = _run([uv, "python", "find", "--managed-python", versions["python"]],
                  "find Python %s" % versions["python"], env).strip().splitlines()[-1]
    # Its own packages may be written: uv's Pythons are not marked as the
    # system's, but say so rather than rely on it.
    install = [uv, "pip", "install", "--python", python, "--break-system-packages"]
    if lock:
        say("installing PyTorch %s from the image's lock" % versions["torch"])
        command = [sys.executable, runtime_script or os.path.join(os.path.dirname(__file__), "runtime.py"),
                   "install", "--manifest", lock, "--python", python]
        if signed:
            command += ["--signed", signed]
        if subprocess.run(command, stdin=subprocess.DEVNULL).returncode != 0:
            raise RuntimeError("could not install PyTorch %s" % versions["torch"])
    else:
        # The CPU build by its own local version, which only PyTorch's CPU
        # index has: asked as plain `torch==2.11.0` with that index beside
        # PyPI, uv took PyPI's CUDA build.
        cpu = versions["torch"].split("+", 1)[0] + "+cpu"
        say("installing PyTorch %s" % cpu)
        _run(install + ["torch==" + cpu, "--extra-index-url", CPU_INDEX], "install PyTorch")
    say("installing Ravex %(ravex)s and Moonclip %(moonclip)s" % versions)
    # What Ravex reads the machine with: CPU and RAM, and the GPUs where there are.
    extra = ["psutil"] + (["nvidia-ml-py"] if gpu else [])
    _run(install + ["ravex==" + versions["ravex"], "moonclip==" + versions["moonclip"]] + extra,
         "install Ravex %(ravex)s and Moonclip %(moonclip)s" % versions)
    with open(marker, "w", encoding="utf-8") as handle:
        json.dump({"python": python, "template": versions}, handle)
    say("libraries made in %.1f s" % (time.monotonic() - began))
    return python


def release_tree(root: str, ravex: str) -> str:
    """The source of that Ravex release - its examples and benches - fetched
    once from its tag. Only ``examples/`` and ``bench/`` are kept, and only
    regular files inside them: a tarball's links and absolute paths are not
    followed out of the folder."""
    if not _VERSION.fullmatch(ravex):
        raise ValueError("%r is not a Ravex version" % ravex)
    home = os.path.join(root, "_ravex", ravex)
    marker = os.path.join(home, ".smith-ready")
    if os.path.isfile(marker):
        return home
    shutil.rmtree(home, ignore_errors=True)
    os.makedirs(home)
    say("fetching Ravex %s's examples and benches" % ravex)
    request = urllib.request.Request(RAVEX_SOURCE % ravex, headers={"User-Agent": "smith"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            parts = member.name.split("/", 1)
            if len(parts) != 2 or not member.isfile():
                continue
            inside = parts[1]
            if not inside.startswith(("examples/", "bench/")) or ".." in inside.split("/"):
                continue
            target = os.path.join(home, *inside.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            source = archive.extractfile(member)
            assert source is not None
            with open(target, "wb") as handle:
                shutil.copyfileobj(source, handle)
    open(marker, "w").close()
    return home


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="The environment a job's template asks for.")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("ensure", help="make it if it is not there; print its Python")
    one.add_argument("--template", required=True)
    one.add_argument("--root", required=True)
    one.add_argument("--locks", default=None)
    one.add_argument("--signed", default=None)
    one.add_argument("--gpu", action="store_true")
    one.add_argument("--out", required=True, help="where to write {python}")
    one = sub.add_parser("tree", help="fetch a Ravex release's examples and benches; write where")
    one.add_argument("--ravex", required=True)
    one.add_argument("--root", required=True)
    one.add_argument("--out", required=True, help="where to write {tree}")
    args = parser.parse_args(argv)
    try:
        if args.command == "ensure":
            template = json.loads(args.template)
            python = ensure(template, args.root, args.locks, signed=args.signed, gpu=args.gpu)
            # Its folder's name, which a repository's environment on top of it is named after.
            answer = {"python": python, "flavour": os.path.basename(folder(args.root, checked(template)))}
        else:
            answer = {"tree": release_tree(args.root, args.ravex)}
    except (RuntimeError, ValueError, OSError) as exc:
        say(str(exc))
        return 2
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(answer, handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
