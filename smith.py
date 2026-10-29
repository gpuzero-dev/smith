"""smith: an agent for training nodes. Takes jobs from a backend's queue and runs them here.

    python smith.py --config smith.toml

The backend never starts a training: it may be a container with no torch and
no GPU, or have no process to start at all. It keeps a queue; smith runs where
the training environment actually is - a laptop's venv, a container, a rented
GPU node - takes a job, starts it, streams its output and says how it went.

**It runs scripts it was given, never commands it was sent.** A job names a
script from the ``[scripts]`` table of this agent's own configuration, and its
parameters become ``--name value`` flags whose names the backend has already
held to what a flag can be. An agent that ran a command line taken from an API
would be a remote shell on every training machine.

**What a job becomes.** A process running the script with

- ``RAVEX_STORAGE_PATH``: a new store under ``runs`` for a fresh run or a
  fork, the run's own store for a resume;
- ``RAVEX_RUN_ID`` and ``RAVEX_NAME`` for a fresh run or a fork, so the run
  the backend shows has the name it was given and an id this agent knows before
  Ravex starts;
- ``RAVEX_FORK_FROM`` and ``RAVEX_FORK_STEP`` for a fork, which is a fresh
  run of its own starting from another run's checkpoint;
- ``RAVEX_METRICS_ENDPOINT``: the backend, so Ravex sends the run there while
  it trains, and ``RAVEX_METRICS_TOKEN`` with the agent's token when
  the backend asks for one.

**Code from a repository.** An agent started with ``code = true`` (or
``SMITH_CODE=1``) also takes jobs that carry ``code``: a repository over
HTTPS, a commit as a full SHA, and the name of a manifest file in it. The job
names a script from the manifest's ``[scripts]`` table - the same shape as
this file's - instead of one of the agent's own. Before it starts, the job's
process fetches that commit alone into ``runs/_code/<repository>/<commit>``,
makes a virtual environment beside it when the repository has a
``requirements.txt`` or a ``pyproject.toml`` (on top of this Python's own
packages, so torch and Ravex are there already; with ``uv`` when it is
installed) and then runs the script with the job's parameters as flags. A
commit already fetched is used again. Every step is in the job's output, with
how long it took (GPU-219). It
is still never a command: a repository, a commit and a name, each checked.
It is opt-in because it runs whatever the repository holds: on a machine of
the person whose repository it is, or one rented for them, that is the point;
on a machine shared with others, leave it off. A private repository's token
comes with the job and reaches git through its environment, never its
command line, and never the training process.

**One run on several nodes.** A job that is one node's part of a run trained
by several - Ravex's outer loop - says so in ``outer``: ``member``, the name of
this node's own store inside the run's (two nodes writing one store would
overwrite each other), which a resume of the run resumes too; ``token``, the run's ``RAVEX_JOB_TOKEN``, which keeps
anyone else out of its rendezvous; and on the node that hosts it, ``serve``:
the port of a ``ravex rendezvous`` this agent starts beside the script and
keeps serving after it, for the nodes still training, until it takes its next
job. Where the rendezvous is and what the run's settings are come
in the job's ``config``, like any other run's. Nothing in ``outer`` is a
command: the agent runs Ravex's own server, on a port.

**Where a run's store lives.** On a disk of this machine under ``runs``, or -
with a ``[storage]`` table - in a bucket, under ``<prefix>/<run id>``, with
the local directory only as Moonclip's staging copy. A rented node
is gone once it stops; a store in the bucket is what lets "Resume" and "Fork"
work on another one. The run reports ``s3://bucket/prefix`` as its address,
and a resume or fork on any agent with the same bucket starts from there. The
bucket's keys come from the environment, ``MOONCLIP_S3_ACCESS_KEY`` and
``MOONCLIP_S3_SECRET_KEY``, never from this file; the jobs inherit them, and
Moonclip, which writes the store, reads them there (GPU-267).

**The token.** ``SMITH_TOKEN`` in the environment (or ``token`` in the
configuration, for a machine where that is how secrets arrive): sent as a
bearer token on every call, and handed to the runs and to ``ravex ship``.

**On a rented pod** one image serves every machine, so what differs between
them comes from the environment and wins over the file: ``SMITH_BACKEND``,
``SMITH_NAME``, ``SMITH_HARDWARE``, ``SMITH_GPUS``, and ``SMITH_STORAGE_TYPE``,
``_BUCKET``, ``_PREFIX``, ``_ENDPOINT``, ``_REGION``, ``_PATH_STYLE`` for the bucket.

Its output goes to ``runs/_logs/job-<id>.log`` - under torchrun, each rank's to
``job-<id>.rank<n>.log`` beside it - and from there to the backend every half
second, numbered line by line, so whoever watches the backend sees it while it
runs: a traceback from a job that died at import included, and a progress bar
redrawn with ``\r`` as the one line it is on a terminal. A secret of the job
printed by the script is replaced by its name before it leaves the machine.

**Stopping.** A job cancelled on the backend is noticed at the next report, and
the process is interrupted the way Ctrl-C would: Ravex sees a
``KeyboardInterrupt``, writes its final checkpoint and records the run as
``interrupted``. One that has not exited a minute later is killed.

**What one machine compiled, the next one does not (GPU-181).** The wheels
uv builds from source - flash-attn takes from minutes to hours - and the
kernels Triton, Inductor and TileLang compile on first use go to directories
under ``runs/_cache``, so the next job on this machine finds them. A job that
brings its owner's own bucket also takes them from ``_cache`` there before it
starts and sends back what it added: ``cache.py``, beside this file and run
with the job's own interpreter, decides whether what is stored was built for a
machine like this one, and loads nothing otherwise. It was ``ravex cache``
until the cache, which is the node's work and not the run's, moved here
(GPU-267). Never with this agent's bucket: on a rented
node that is the platform's, shared by everybody, and a cache that every node
may write is a way to run code on the others' machines. A cache that fails
is said in the job's output and costs nothing but the compilation.

With ``cache = "backend"`` (or ``SMITH_CACHE=backend``) the cache is the
backend's instead, and this machine holds no key to it (GPU-200): ``ravex
cache`` asks the backend, with this agent's token, to sign each operation
(``POST /api/cache/{scope}``). It pulls from ``global``, which machines may
only read, then from ``workspace``, the job's owner's own, and pushes only
there. The token is this agent's own, which the run already has for its
metrics (``RAVEX_METRICS_TOKEN``) and could read here anyway: the training
runs as the same user in the same container, and the token is in this
agent's environment. What keeps the cache safe is not who holds the token
but what the backend lets a workspace do with it - read ``global``, write
only its own, up to its quota - so a script, or a library it imports, can
reach no further than its owner's own cache.

The ``global`` part is filled by the backend's builder (``builder.py``,
GPU-202), never by a node. After the install this machine reports, by name,
the wheels uv still had to compile; the builder compiles them again from
PyPI's source for this machine's ``cache.py key --name wheels``, and the
next machine like this one pulls them as ``wheels`` and hands them to uv with
``--find-links``, where a wheel wins over the sdist of the same version.

The kernels go back while the job runs too, not only when it ends: five
minutes in and then every fifteen, the files that have stopped changing. A
node the provider takes back never reaches the end, and before this lost
everything it had compiled. The push after the job goes on after it, a
request and an upload per file, and every heartbeat says so (``pushing``)
until it is over: a backend that gives back idle machines would otherwise
count this one idle from the job's end and take it halfway through.

**What a run could not send.** A run that ends while the backend is down does
not wait for it - Ravex keeps the unsent metrics on disk for the next
execution on that store, and a finished run never gets one. This agent
outlives the run and sits beside its store, so after every job it runs
``ravex ship`` on it, again every half minute until the backend has it all.
Without that, the first end-to-end test left a run showing ``running`` forever
with four fifths of its points missing.

Only the standard library, so it runs in whatever environment the training
does without adding to it. Python 3.11 or newer, for ``tomllib``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: Seconds between two polls of the queue, and between two heartbeats.
POLL_SECONDS = 2.0

#: Seconds between two reports on a running job. Also how long a cancel on
#: the backend takes to reach the process, at most.
REPORT_SECONDS = 4.0

#: How long a job that was asked to stop has before it is killed.
STOP_GRACE_SECONDS = 60.0

#: Seconds between two attempts at sending what a finished run left unsent.
SHIP_RETRY_SECONDS = 30.0

#: Most lines of output one post carries, and most bytes read per poll.
LOG_BATCH = 500
LOG_READ = 1 << 20

#: Seconds between two reads of the jobs' output. Shorter than a poll: a
#: ``print`` on the node should be on the page within a second or two, and
#: a read with nothing new costs no call (GPU-178).
LOG_SECONDS = 0.5

TIMEOUT = 10.0

#: The image this agent runs in, as the build that made it named it
#: (``SMITH_IMAGE``, set in Dockerfile.gpu); empty outside one. Written at the
#: top of every job's output, because a node that started from an older copy
#: of a moving tag otherwise looks exactly like one that did not.
IMAGE = os.environ.get("SMITH_IMAGE", "").strip()

#: The cache of compiled wheels and kernels (``cache.py``), run as a script
#: with whichever interpreter's key it is: the job's, or this agent's for the
#: wheels its environment is built from. By path, because the job's
#: interpreter cannot import it - it is not installed anywhere.
CACHE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache.py")

#: The compiled directories a job keeps, by name in the cache: the variable
#: that points the tool at it, and the library whose version is part of what
#: the files were built for. ``uv`` is the wheels it built from source, which
#: `uv cache prune --ci` leaves alone while dropping what it downloaded.
CACHES = (
    ("uv", "UV_CACHE_DIR", "uv"),
    ("triton", "TRITON_CACHE_DIR", "triton"),
    ("inductor", "TORCHINDUCTOR_CACHE_DIR", "triton"),
    ("tilelang", "TILELANG_CACHE_DIR", "tilelang"),
)

#: The kernels a job compiled go back to its cache: one process for all of
#: them, in the job's own interpreter, so the key is the training's. After the
#: job, every file. While it runs (``settled`` set), only the files that have
#: not changed for that many seconds and are not a library's temporary,
#: linked into a snapshot beside the directory and pushed from there: a file
#: half-written when the push read it would be stored under its final name,
#: and every later push would skip it as already there.
_PUSH_KERNELS = r"""
import json, os, shutil, subprocess, sys, time
spec = json.loads(sys.argv[1])
settled = spec.get("settled")
for name, directory, library in spec["caches"]:
    source = directory
    if settled:
        source = directory.rstrip("/\\") + ".push"
        shutil.rmtree(source, ignore_errors=True)
        cutoff = time.time() - settled
        for folder, _dirs, files in os.walk(directory):
            for file in files:
                path = os.path.join(folder, file)
                try:
                    if ".tmp" in file or os.path.getmtime(path) > cutoff:
                        continue
                    target = os.path.join(source, os.path.relpath(path, directory))
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    try:
                        os.link(path, target)
                    except OSError:
                        shutil.copy2(path, target)
                except OSError:
                    continue
    subprocess.run([spec["python"], spec["cache_script"], "push", "--name", name,
                    "--dir", source, "--store", spec["store"], "--with", library],
                   stdin=subprocess.DEVNULL)
    if source != directory:
        shutil.rmtree(source, ignore_errors=True)
"""

#: While a job runs, its kernels go to its cache this long after it started
#: and then this often, not only when it ends: a node the provider takes back
#: never reaches the end, and everything it compiled was lost with it. The
#: first push comes once the first steps have compiled most of what the
#: training needs; later ones pick up what eval, a recompile or a checkpoint
#: compiled since. Each sends only the names the cache lacks.
KERNEL_PUSH_FIRST_SECONDS = 300.0
KERNEL_PUSH_EVERY_SECONDS = 900.0

#: How long a kernel file must have stayed unchanged to be pushed mid-run.
KERNEL_SETTLED_SECONDS = 120.0

#: Runs the script the way ``python script.py`` would, with one difference: a
#: stop request arrives as ``KeyboardInterrupt``. On Windows the only signal
#: one process can send another's group is CTRL_BREAK, whose default is to
#: end the process on the spot - no final checkpoint, and a status left saying
#: "running". Turned into an exception, it unwinds through Ravex like Ctrl-C.
#:
#: Once it has unwound - Ravex's final checkpoint written on the way out - it
#: ends here with one line, not the stack trace Python would print for it: a
#: stop asked for is not a crash, and a page of traceback in the job's output
#: read like one. The job's state does not come from this exit; the agent
#: knows a stop was asked for.
#:
#: Under torchrun, each rank writes to a file of its own (``SMITH_RANK_LOG``,
#: with the local rank in it) instead of the job's: four processes printing
#: into one file cut each other's lines in half, and which rank said what was
#: lost. At the descriptors, so what CUDA, NCCL or a C extension prints is in
#: it too, not only what goes through ``print`` (GPU-178).
_LAUNCHER = """
import os, runpy, signal, sys

def _stop(*_):
    raise KeyboardInterrupt

for _name in ("SIGBREAK", "SIGTERM"):
    if hasattr(signal, _name):
        signal.signal(getattr(signal, _name), _stop)

_ranked = os.environ.pop("SMITH_RANK_LOG", "")
if _ranked and os.environ.get("LOCAL_RANK", "").isdigit():
    _fd = os.open(_ranked % int(os.environ["LOCAL_RANK"]),
                  os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0), 0o644)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(_fd, 1)
    os.dup2(_fd, 2)
    os.close(_fd)

_path = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(_path)))
try:
    runpy.run_path(_path, run_name="__main__")
except KeyboardInterrupt:
    print("[smith] stopped on request", file=sys.stderr, flush=True)
    sys.exit(130)
"""


#: Whether the GPUs are there, asked of the interpreter that runs the
#: training, the way the training will: torch sees CUDA, as many GPUs as the
#: machine has, and a small sum on each comes back. A machine rented as a GPU
#: whose container cannot reach it would otherwise train on the CPU - slowly,
#: and paid for as a GPU - and say so only in a warning nobody reads (GPU-188).
#: The last line it prints is what is wrong, or what it found.
_GPU_CHECK = r"""
import sys
want = int(sys.argv[1])
try:
    import torch
except Exception as exc:
    print("torch cannot be imported: %s" % exc)
    sys.exit(1)
if not torch.cuda.is_available():
    print("torch finds no usable CUDA device on this machine")
    sys.exit(1)
have = torch.cuda.device_count()
if have < want:
    print("torch sees %d GPU(s), and this machine is meant to have %d" % (have, want))
    sys.exit(1)
try:
    for index in range(want):
        x = torch.ones(1024, device="cuda:%d" % index)
        (x * 2).sum().item()
except Exception as exc:
    print("GPU %d does not compute: %s" % (index, exc))
    sys.exit(1)
print("%d GPU(s): %s" % (want, ", ".join(torch.cuda.get_device_name(i) for i in range(want))))
"""

#: How long the check may take: loading torch and CUDA on a cold machine is
#: tens of seconds, never minutes.
GPU_CHECK_SECONDS = 180.0

#: How long fetching and installing the runtime may take (``runtime.py``):
#: about 3.5 GB of wheels, a minute at 500 Mb/s and ten at 50. Past that the
#: machine is too slow to be worth its price, and says so.
RUNTIME_SECONDS = 15 * 60.0


#: What a job's code may be: a repository over HTTPS, a commit as a full SHA,
#: a manifest and a script named plainly. Checked before anything runs, so a
#: value cannot become an option of git or a path out of the checkout.
_REPO = re.compile(r"^https://[A-Za-z0-9.-]+(?::\d+)?/[A-Za-z0-9_.~/-]+$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_RELATIVE = re.compile(r"^(?!/)(?!.*\.\.)[A-Za-z0-9_./-]{1,200}$")
_SCRIPT = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
#: What a parameter's name may be: it becomes a flag, ``--name``. Checked with
#: ``fullmatch`` like the others, since ``$`` alone also matches before a
#: final newline.
_PARAM = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}")

#: A job with ``code`` runs this first, as its own process: fetch the commit,
#: make its environment, find the script in the manifest, then become the
#: training. In the job's process and not in this agent, so a clone and an
#: install of several minutes neither stop the heartbeats nor escape the
#: job's output, and a stop reaches them like it reaches the training.
_PREPARE = r"""
import base64, json, os, shutil, signal, subprocess, sys, time, tomllib, urllib.request

# Every step says how long it took, from this process's start: the job's log
# is the only place the time of a node's start can be read from, and its lines
# carry no time of their own (GPU-219).
began = time.monotonic()
spec = json.loads(os.environ.pop("SMITH_PREPARE"))
token = os.environ.pop("SMITH_GIT_TOKEN", "")
# The cache's credentials go to `cache.py` and not into the training's
# environment, where a library that reports the environment would carry them
# off. A boundary they are not: the training runs as this process's user and
# can read them all the same, and the backend's token is in its environment
# for the metrics anyway (GPU-200).
cache_env = {name: os.environ.pop(name) for name in ("SMITH_CACHE_ACCESS_KEY", "SMITH_CACHE_SECRET_KEY",
                                                       "SMITH_CACHE_ENDPOINT", "SMITH_CACHE_REGION",
                                                       "SMITH_CACHE_PATH_STYLE", "SMITH_CACHE_TOKEN")
             if name in os.environ}


def say(message):
    print("[smith] " + message, flush=True)


def took(what, since):
    say("%s in %.1f s" % (what, time.monotonic() - since))


def fail(message):
    say(message)
    sys.exit(2)


def run(command, what, **kwargs):
    if subprocess.run(command, stdin=subprocess.DEVNULL, **kwargs).returncode != 0:
        fail("could not " + what)


# Pull or push some of the job's caches; a failure is said and passed. A pull
# goes through every store in order, each adding what the directory lacks.
def cache(action, python, names):
    stores = spec["cache_reads"] if action == "pull" else [spec["cache_store"]]
    for store in [store for store in stores if store]:
        for name, directory, library in spec["caches"]:
            if name not in names:
                continue
            since = time.monotonic()
            done = subprocess.run([python, spec["cache_script"], action, "--name", name, "--dir", directory,
                                   "--store", store, "--with", library],
                                  stdin=subprocess.DEVNULL, env=dict(os.environ, **cache_env))
            if done.returncode != 0:
                say("the %s cache was not %s after %.1f s; what it would have held is compiled here" % (
                    name, "used" if action == "pull" else "sent", time.monotonic() - since))
            else:
                took("%s the %s cache" % ("pulled" if action == "pull" else "pushed", name), since)


# The wheels the backend's builder compiled for this machine's key, from the
# cache's global part, which only that builder writes: no --with, since what a
# wheel was built against is the interpreter, torch and the GPUs, not the uv
# that installs it.
def wheels_pull():
    if not spec["wheels_store"]:
        return
    since = time.monotonic()
    done = subprocess.run([sys.executable, spec["cache_script"], "pull", "--name", "wheels",
                           "--dir", spec["wheels"], "--store", spec["wheels_store"]],
                          stdin=subprocess.DEVNULL, env=dict(os.environ, **cache_env))
    if done.returncode != 0:
        say("the builder's wheels were not used after %.1f s; what they would have been is compiled here"
            % (time.monotonic() - since))
    else:
        took("pulled the builder's wheels", since)


# What uv still had to compile here, by name, for the backend's builder to
# compile from PyPI's source for the next machine like this one. Only names:
# nothing this machine made goes where every machine reads. The wheels uv
# built are the ones `uv cache prune --ci` left in its sdists bucket.
def wheels_report():
    if not spec["wheels_report"]:
        return
    have = set(os.listdir(spec["wheels"])) if os.path.isdir(spec["wheels"]) else set()
    built = {}
    for folder, _dirs, files in os.walk(os.environ.get("UV_CACHE_DIR", "")):
        if os.sep + "sdists-" not in folder:
            continue
        for file in files:
            parts = file.split("-")
            if file.endswith(".whl") and file not in have and len(parts) >= 5:
                built[(parts[0], parts[1])] = None
    if not built:
        return
    try:
        key = json.loads(subprocess.run([sys.executable, spec["cache_script"], "key", "--name", "wheels"],
                                        stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True).stdout)
        body = json.dumps({"key": key, "wheels": [{"name": n, "version": v} for n, v in sorted(built)]}).encode()
        request = urllib.request.Request(spec["wheels_report"], data=body, method="POST", headers={
            "Content-Type": "application/json", "User-Agent": "smith",
            "Authorization": "Bearer " + cache_env.get("SMITH_CACHE_TOKEN", "")})
        with urllib.request.urlopen(request, timeout=30) as response:
            answer = json.loads(response.read().decode() or "{}")
        for wheel in answer.get("queued", []):
            say("%s %s was compiled here; the builder will compile it for the next machine like this one" % (
                wheel["name"], wheel["version"]))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        say("the compiled wheels were not reported: %s" % exc)


tree = spec["tree"]
ready = os.path.join(tree, ".smith-ready")
short = spec["commit"][:12]
if os.path.isfile(ready):
    with open(ready, encoding="utf-8") as handle:
        python = json.load(handle)["python"]
    say("%s at %s, fetched before" % (spec["repo"], short))
else:
    # From nothing each time: a fetch cut short leaves no marker.
    shutil.rmtree(tree, ignore_errors=True)
    os.makedirs(tree)
    say("fetching %s at %s" % (spec["repo"], short))
    since = time.monotonic()
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    if token:
        # In git's environment, where other users of the machine cannot read
        # it, rather than in the address or on the command line.
        basic = base64.b64encode(("x-access-token:" + token).encode()).decode()
        env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.extraHeader",
                   GIT_CONFIG_VALUE_0="Authorization: Basic " + basic)
    run(["git", "init", "-q", tree], "make a repository in " + tree, env=env)
    run(["git", "-C", tree, "fetch", "-q", "--depth", "1", spec["repo"], spec["commit"]],
        "fetch commit %s; is the repository right, and is a private one given its token?" % short, env=env)
    run(["git", "-C", tree, "checkout", "-q", "--detach", "FETCH_HEAD"], "check out " + short, env=env)
    took("fetched", since)
    python = sys.executable
    # Before the install, with this interpreter: the wheels are built against
    # what it has, which is what the new environment starts from.
    cache("pull", python, ("uv",))
    wheels_pull()
    wants = "requirements.txt" if os.path.isfile(os.path.join(tree, "requirements.txt")) else (
        "pyproject.toml" if os.path.isfile(os.path.join(tree, "pyproject.toml")) else None)
    if wants:
        venv = os.path.join(tree, ".smith-venv")
        run([sys.executable, "-m", "venv", "--system-site-packages", venv], "make a virtual environment")
        python = os.path.join(venv, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv, "bin", "python")
        what = ["-r", "requirements.txt"] if wants == "requirements.txt" else ["."]
        say("installing what %s asks for" % wants)
        # A wheel there wins over the sdist of the same version on PyPI, so
        # what the builder compiled is installed instead of compiled again.
        find_links = ["--find-links", spec["wheels"]] if os.path.isdir(spec["wheels"]) and os.listdir(spec["wheels"]) else []
        since = time.monotonic()
        uv = shutil.which("uv")
        if uv:
            run([uv, "pip", "install", "--python", python] + find_links + what, "install the dependencies", cwd=tree)
            took("installed", since)
            if spec["cache_store"]:
                # What it downloaded is out, what it built stays: the cache is
                # then only the wheels nobody publishes for this machine.
                subprocess.run([uv, "cache", "prune", "--ci", "-q"], stdin=subprocess.DEVNULL)
                cache("push", sys.executable, ("uv",))
                wheels_report()
        else:
            run([python, "-m", "pip", "install", "--disable-pip-version-check", "-q"] + find_links + what,
                "install the dependencies", cwd=tree)
            took("installed", since)
    with open(ready, "w", encoding="utf-8") as handle:
        json.dump({"python": python}, handle)

# The kernels with the interpreter that will compile them: a torch the
# repository installed in its environment is a key of its own.
cache("pull", python, ("triton", "inductor", "tilelang"))

manifest = os.path.join(tree, spec["manifest"])
if not os.path.isfile(manifest):
    fail("%s has no %s, which names the scripts that may be run" % (short, spec["manifest"]))
with open(manifest, "rb") as handle:
    scripts = tomllib.load(handle).get("scripts") or {}
entry = scripts.get(spec["script"])
if entry is None:
    fail("%s declares no script %r; it has %s" % (spec["manifest"], spec["script"], ", ".join(sorted(scripts)) or "none"))
path = entry.get("path") if isinstance(entry, dict) else entry
root = os.path.realpath(tree)
script = os.path.realpath(os.path.join(root, str(path or "")))
if not path or os.path.commonpath([script, root]) != root or not os.path.isfile(script):
    fail("scripts.%s in %s is not a file of the repository" % (spec["script"], spec["manifest"]))

if spec["gpus"] > 1:
    command = [python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(spec["gpus"]),
               spec["launcher_file"], script]
else:
    command = [python, "-c", spec["launcher"], script]
command += spec["flags"]
say("running %s, %.1f s after the job started" % (spec["script"], time.monotonic() - began))
os.chdir(root)
if os.name == "nt":
    # No exec on Windows: wait for the training, and leave a stop to it.
    for name in ("SIGBREAK", "SIGINT"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.SIG_IGN)
    sys.exit(subprocess.run(command).returncode)
os.execv(command[0], command)
"""


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


# ─── configuration ──────────────────────────────────────────────────


@dataclass
class Config:
    backend: str
    name: str
    hardware: str
    runs: str
    python: str
    scripts: Dict[str, str] = field(default_factory=dict)
    #: Per script, the parameters that change the model itself - its shape,
    #: what its weights mean. A checkpoint cannot continue with a different
    #: one, so the backend refuses to "save" or fork with it changed and
    #: offers a run from scratch instead. Only the script knows which they are.
    model_params: Dict[str, List[str]] = field(default_factory=dict)
    #: Per script, the parameters a job starts from, with their values: what
    #: the Launch page fills in when the script is picked. Only the script
    #: knows which flags it takes, and a form still holding another script's
    #: would start a job that dies on its first line (GPU-172).
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: The backend's bearer token for agents and runs; None when it asks none.
    token: Optional[str] = None
    #: The bucket runs keep their stores in; None keeps them on this disk.
    storage: Optional["Bucket"] = None
    #: GPUs on this machine. More than one, and a job runs one process per GPU
    #: under torchrun: a node with four GPUs uses all four.
    gpus: int = 1
    #: Whether it takes jobs that bring a repository's commit to run (see
    #: "Code from a repository" above). Off unless asked for.
    code: bool = False
    #: Whether it checks at start that its GPUs work (``_GPU_CHECK``), and
    #: takes no job if they do not. On by default for hardware named
    #: ``gpu...``; ``check_gpus`` / ``SMITH_CHECK_GPUS`` say otherwise.
    check_gpus: bool = False
    #: Whether a job's compiled dependencies are kept in the backend's cache,
    #: through URLs it signs (see "What one machine compiled" above). Off
    #: unless asked for: a backend without one would refuse every pull.
    cache_service: bool = False
    #: The manifest of the heavy part of this Python - torch, its CUDA
    #: libraries - which the image does not carry and ``runtime.py`` fetches
    #: and installs before the first job (``runtime`` in the file); None where
    #: the environment already has it all.
    runtime: Optional[str] = None

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
        here = os.path.dirname(os.path.abspath(path))

        def resolve(value: str) -> str:
            return os.path.normpath(os.path.join(here, os.path.expanduser(value)))

        scripts: Dict[str, str] = {}
        model_params: Dict[str, List[str]] = {}
        params: Dict[str, Dict[str, Any]] = {}
        for key, entry in (raw.get("scripts") or {}).items():
            script = entry.get("path") if isinstance(entry, dict) else entry
            if not script:
                raise SystemExit("scripts.%s has no path" % key)
            resolved = resolve(script)
            if not os.path.isfile(resolved):
                raise SystemExit("scripts.%s: %s does not exist" % (key, resolved))
            scripts[key] = resolved
            model_params[key] = [str(name) for name in (entry.get("model") or [])] if isinstance(entry, dict) else []
            given = entry.get("params") if isinstance(entry, dict) else None
            if given is not None and not isinstance(given, dict):
                raise SystemExit("scripts.%s.params is not a table" % key)
            params[key] = dict(given or {})
        # The environment wins over the file for what differs between machines
        # running the same image: a rented pod is configured with variables,
        # and its image carries one file for all of them.
        env = os.environ.get
        code = str(env("SMITH_CODE") or raw.get("code", False)).lower() in ("true", "1", "yes")
        if not scripts and not code:
            raise SystemExit("%s names no scripts; there would be nothing to run" % path)
        table = dict(raw.get("storage") or {})
        for key in ("type", "bucket", "prefix", "endpoint", "region", "path_style"):
            if env("SMITH_STORAGE_" + key.upper()):
                table[key] = env("SMITH_STORAGE_" + key.upper())
        storage = Bucket.from_table(table, path)
        return cls(
            backend=str(env("SMITH_BACKEND") or raw.get("backend", "http://127.0.0.1:8600")).rstrip("/"),
            name=str(env("SMITH_NAME") or raw.get("name") or socket.gethostname()),
            hardware=str(env("SMITH_HARDWARE") or raw.get("hardware", "local-cpu")),
            gpus=max(1, int(env("SMITH_GPUS") or raw.get("gpus", 1))),
            runs=resolve(raw.get("runs", "runs")),
            python=raw.get("python") or sys.executable,
            scripts=scripts,
            model_params=model_params,
            params=params,
            token=os.environ.get("SMITH_TOKEN") or raw.get("token") or None,
            storage=storage,
            code=code,
            check_gpus=str(
                env("SMITH_CHECK_GPUS")
                or raw.get("check_gpus", str(env("SMITH_HARDWARE") or raw.get("hardware", "local-cpu")).startswith("gpu"))
            ).lower() in ("true", "1", "yes"),
            cache_service=str(env("SMITH_CACHE") or raw.get("cache", "")).lower() == "backend",
            runtime=resolve(raw["runtime"]) if raw.get("runtime") else None,
        )


@dataclass
class Bucket:
    """Where the runs' stores go when they do not stay on this machine."""

    type: str
    bucket: str
    prefix: str = ""
    endpoint: Optional[str] = None
    region: Optional[str] = None
    path_style: bool = False
    #: The bucket's own keys, when a backend handed one with a job. Unset,
    #: the run takes this agent's MOONCLIP_S3_* from the environment. Out of
    #: the repr, so that printing a bucket never prints them.
    access_key: Optional[str] = field(default=None, repr=False)
    secret_key: Optional[str] = field(default=None, repr=False)
    #: With temporary keys, the token they sign nothing without: a backend
    #: may hand ones scoped to this bucket, or to its prefix, for a time.
    session_token: Optional[str] = field(default=None, repr=False)

    @classmethod
    def from_job(cls, table: Dict[str, Any]) -> "Bucket":
        """The bucket a backend handed with a job, keys included: that job's
        store goes there instead of this agent's bucket."""
        kind = str(table.get("type") or "s3")
        if kind not in ("s3", "r2") or not table.get("bucket"):
            raise RuntimeError("the job's storage is not a bucket this agent can use")
        if not (table.get("access_key") and table.get("secret_key")):
            raise RuntimeError("the job's storage came without its keys")
        return cls(
            type=kind,
            bucket=str(table["bucket"]),
            prefix=str(table.get("prefix") or "").strip("/"),
            endpoint=table.get("endpoint"),
            region=table.get("region"),
            path_style=bool(table.get("path_style")),
            access_key=str(table["access_key"]),
            secret_key=str(table["secret_key"]),
            session_token=str(table["session_token"]) if table.get("session_token") else None,
        )

    @classmethod
    def from_table(cls, table: Any, path: str) -> Optional["Bucket"]:
        if not table:
            return None
        kind = str(table.get("type", "r2"))
        if kind not in ("s3", "r2"):
            raise SystemExit("storage.type in %s is %r; a bucket is s3 or r2" % (path, kind))
        if not table.get("bucket"):
            raise SystemExit("storage in %s names no bucket" % path)
        if kind == "r2" and not table.get("endpoint"):
            raise SystemExit("storage in %s is r2 and has no endpoint" % path)
        for name in ("MOONCLIP_S3_ACCESS_KEY", "MOONCLIP_S3_SECRET_KEY"):
            # Said now, not by the first job to fail on it.
            if not os.environ.get(name):
                raise SystemExit("storage in %s is a bucket, but %s is not set" % (path, name))
        return cls(
            type=kind,
            bucket=str(table["bucket"]),
            prefix=str(table.get("prefix", "")).strip("/"),
            endpoint=table.get("endpoint"),
            region=table.get("region"),
            # From TOML a bool, from the environment a string: "false" is not true.
            path_style=str(table.get("path_style", False)).lower() in ("true", "1", "yes"),
        )

    def prefix_of(self, run_id: str) -> str:
        return "%s/%s" % (self.prefix, run_id) if self.prefix else run_id

    def prefix_in(self, uri: str) -> Optional[str]:
        """The prefix of ``uri`` if it is a store in this bucket, else None."""
        head = "s3://%s/" % self.bucket
        return uri[len(head):].strip("/") or None if uri.startswith(head) else None

    def environment(self, prefix: str) -> Dict[str, str]:
        """What a run needs to keep a store under ``prefix`` in this bucket:
        where it is for Ravex, the keys for Moonclip, which reads them itself."""
        env = {
            "RAVEX_STORAGE_TYPE": self.type,
            "RAVEX_STORAGE_BUCKET": self.bucket,
            "RAVEX_STORAGE_PREFIX": prefix,
            "RAVEX_STORAGE_PATH_STYLE": "true" if self.path_style else "false",
        }
        if self.endpoint:
            env["RAVEX_STORAGE_ENDPOINT"] = str(self.endpoint)
        if self.region:
            env["RAVEX_STORAGE_REGION"] = str(self.region)
        if self.access_key and self.secret_key:
            env["MOONCLIP_S3_ACCESS_KEY"] = self.access_key
            env["MOONCLIP_S3_SECRET_KEY"] = self.secret_key
            if self.session_token:
                env["MOONCLIP_S3_SESSION_TOKEN"] = self.session_token
        return env


# ─── the backend ────────────────────────────────────────────────────


class Unreachable(Exception):
    pass


class Backend:
    def __init__(self, base: str, token: Optional[str] = None) -> None:
        self.base = base
        self._down_since: Optional[float] = None
        # A name of its own: urllib's default, Python-urllib/3.x, is one that
        # a CDN in front of a backend may refuse outright as a bot (Cloudflare
        # answers 403 with error 1010), and every call would fail before it
        # reached the backend.
        self._headers = {"Content-Type": "application/json", "User-Agent": "smith"}
        if token:
            self._headers["Authorization"] = "Bearer " + token

    def call(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(self.base + path, data=data, method=method, headers=self._headers)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                answer = json.loads(response.read().decode("utf-8") or "null")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError("%s %s: %d %s" % (method, path, exc.code, detail)) from None
        except (urllib.error.URLError, OSError) as exc:
            if self._down_since is None:
                self._down_since = time.monotonic()
                log("backend %s unreachable (%s); jobs already running carry on" % (self.base, exc))
            raise Unreachable(str(exc)) from None
        if self._down_since is not None:
            log("backend reachable again")
            self._down_since = None
        return answer


# ─── a job, running ─────────────────────────────────────────────────


def new_run_id() -> str:
    """The same shape Ravex gives a run it names itself."""
    return "r%s-%s" % (time.strftime("%y%m%d%H%M", time.gmtime()), uuid.uuid4().hex[:6])


def flags(params: Dict[str, Any]) -> List[str]:
    """A job's parameters as the flags its script is started with. A name that
    is not a flag's is refused rather than passed on: a backend's word is not
    enough for what lands on a command line."""
    out: List[str] = []
    for key, value in params.items():
        if not isinstance(key, str) or not _PARAM.fullmatch(key):
            raise RuntimeError("%r is not a parameter's name" % (key,))
        if value is True:
            out.append("--" + key)
        elif value is False or value is None:
            continue
        else:
            out.extend(["--" + key, str(value)])
    return out


def redact(text: str, secrets: Dict[str, str]) -> str:
    """``text`` with each secret's value replaced by its name, the longest
    first so a value that contains another is not half replaced."""
    for name, value in sorted(secrets.items(), key=lambda item: -len(item[1])):
        if value and value in text:
            text = text.replace(value, "[secret %s]" % name)
    return text


def job_secrets(job: Dict[str, Any]) -> Dict[str, str]:
    """The secrets a backend gave this job: names and values for its
    process's environment. A name smith sets itself, or one the libraries
    under the run read their settings from, is dropped rather than let a
    secret replace the run's own settings: a ``MOONCLIP_S3_*`` secret would
    send the run's checkpoints to another bucket."""
    given = job.get("secrets")
    if not isinstance(given, dict):
        return {}
    return {
        str(name): str(value)
        for name, value in given.items()
        if not str(name).startswith(("RAVEX_", "SMITH_", "MOONCLIP_"))
    }


class Screen:
    """The line a terminal would be showing, as text is written to it.

    ``\\r`` takes the cursor back to the start of the line and what follows
    overwrites what was there, the way a progress bar redraws itself; ``\\n``
    ends the line. Without this a tqdm bar was a thousand lines, one per
    redraw, and the one that mattered - the last - was buried (GPU-178).
    Escape sequences other than colours are not interpreted: a line redrawn
    with cursor movements comes out as the text it wrote.
    """

    def __init__(self, text: str = "", cursor: int = 0) -> None:
        self.text = text
        self.cursor = cursor

    def copy(self) -> "Screen":
        return Screen(self.text, self.cursor)

    def write(self, data: str) -> List[str]:
        """Write ``data``; the lines it ended, in order."""
        ended: List[str] = []
        for piece in re.split(r"(\r|\n)", data):
            if piece == "\n":
                ended.append(self.text)
                self.text, self.cursor = "", 0
            elif piece == "\r":
                self.cursor = 0
            elif piece:
                self.text = self.text[: self.cursor] + piece + self.text[self.cursor + len(piece) :]
                self.cursor += len(piece)
                if len(self.text) > LINE_KEPT:
                    # A "line" of megabytes - a bar with no \r, a dump - is
                    # cut here rather than held whole in memory; the backend
                    # keeps less than this of it anyway.
                    self.text = self.text[:LINE_KEPT]
                    self.cursor = min(self.cursor, LINE_KEPT)
        return ended


#: Most characters of one line this agent holds. The backend keeps 4000.
LINE_KEPT = 8000


def _whole(data: bytes) -> int:
    """Where ``data`` ends if a UTF-8 character cut by the read is left out."""
    lead = len(data) - 1
    while lead > len(data) - 4 and lead > 0 and data[lead] & 0xC0 == 0x80:
        lead -= 1
    if lead < 0 or data[lead] < 0xC0:
        return len(data)
    size = 2 if data[lead] < 0xE0 else 3 if data[lead] < 0xF0 else 4
    return len(data) if len(data) - lead >= size else lead


class LogTail:
    """One stream of a job's output - its process, or one rank's - followed
    and sent to the backend.

    Lines are numbered from the start of the stream, and nothing moves until
    the backend has taken a whole read: a batch lost to a timeout is sent
    again with the same numbers, which the backend keeps once. The line still
    being written - no newline yet, typically a progress bar redrawing itself
    with ``\\r`` - goes as ``partial``, each time it changes, so the page
    shows the bar moving instead of nothing until it ends; it replaces the
    previous one and is never numbered. At the end of the job it is the last
    line.

    ``rank`` is -1 for the job's own process, and the rank's number for the
    stream of one process of several under torchrun (see ``rank_log``).
    """

    def __init__(self, job_id: int, path: str, secrets: Optional[Dict[str, str]] = None, rank: int = -1) -> None:
        self.job_id = job_id
        self.path = path
        self.rank = rank
        #: Values that must not leave this machine in a log line.
        self.secrets = secrets or {}
        self.offset = 0
        self.line = 0
        #: The line being written, as of ``offset``.
        self.screen = Screen()
        #: The line in progress as the backend last took it; None for none.
        self.partial: Optional[str] = None
        #: The process has exited: nothing more will be written.
        self.done = False

    def send(self, backend: "Backend") -> bool:
        """Send what is new. True once everything written so far has arrived."""
        try:
            with open(self.path, "rb") as handle:
                handle.seek(self.offset)
                data = handle.read(LOG_READ)
        except OSError:
            return self.done
        full = len(data) == LOG_READ
        # Consumed up to the last \n or \r: both are single bytes that never
        # sit inside a UTF-8 character, so what is left over decodes on its
        # own next time. A read full of neither is taken whole, cut back to
        # the start of a character.
        end = max(data.rfind(b"\n"), data.rfind(b"\r")) + 1
        if end == 0 and full:
            end = _whole(data)
        if self.done and not full:
            end = len(data)
        screen = self.screen.copy()
        lines = screen.write(data[:end].decode("utf-8", "replace"))
        rest = screen.copy()
        rest.write(data[end : _whole(data) if end < len(data) else end].decode("utf-8", "replace"))
        partial: Optional[str] = rest.text or None
        if self.done and not full:
            if partial is not None:
                lines.append(partial)
            partial = None
            screen = Screen()
        if self.secrets:
            lines = [redact(line, self.secrets) for line in lines]
            partial = redact(partial, self.secrets) if partial is not None else None
        if not lines and partial == self.partial:
            self.offset += end
            self.screen = screen
            return not full
        for first in range(0, max(len(lines), 1), LOG_BATCH):
            last = first + LOG_BATCH >= len(lines)
            backend.call(
                "POST",
                "/api/jobs/%d/logs" % self.job_id,
                {
                    "start": self.line + first,
                    "lines": lines[first : first + LOG_BATCH],
                    "rank": self.rank,
                    # Only with the last batch: until then it is not the line
                    # after the ones sent.
                    "partial": partial if last else None,
                },
            )
        self.offset += end
        self.line += len(lines)
        self.screen = screen
        self.partial = partial
        return not full


class Running:
    """A job this agent started, and the process carrying it out."""

    def __init__(self, job: Dict[str, Any], config: Config) -> None:
        self.job = job
        self.id = int(job["id"])
        self.stop_requested_at: Optional[float] = None
        self.last_report = 0.0
        #: When its kernels next go to its cache while it runs, and the push
        #: under way, so that two never overlap.
        self.next_push = time.monotonic() + KERNEL_PUSH_FIRST_SECONDS
        self.mid_push: Optional[subprocess.Popen] = None

        #: The repository's commit this job runs, checked; None for one of
        #: this agent's own scripts.
        self.code = self._checked_code(job, config) if job.get("code") else None
        script = config.scripts[job["script"]] if self.code is None else ""
        env = dict(os.environ)
        # The secrets a backend gave this job: its HF_TOKEN, its W&B key. In
        # the training process's environment only - never this agent's own,
        # which outlives the job - and set first, so that nothing smith sets
        # below can be replaced by one.
        self.secrets = job_secrets(job)
        env.update(self.secrets)
        env["PYTHONUNBUFFERED"] = "1"
        outer = job.get("outer") or {}
        #: This node's store inside the run's, on a run trained by several.
        self.member = str(outer.get("member") or "")
        if self.member and not self.member.replace("-", "").replace("_", "").isalnum():
            raise RuntimeError("member %r is not a name a store can have" % self.member)
        token = str(outer.get("token") or "")
        if token:
            env["RAVEX_JOB_TOKEN"] = token
        #: The rendezvous this agent serves for the run, if it hosts it.
        self.rendezvous: Optional[subprocess.Popen] = None
        # The bucket this job's store goes to: the one the backend handed with
        # it - the job's owner's own - or else this agent's.
        bucket = Bucket.from_job(job["storage"]) if job.get("storage") else config.storage
        #: Every value the job's output must not carry back: its secrets, and
        #: its bucket's keys when they came with it.
        self.hidden = dict(self.secrets)
        if token:
            self.hidden["RAVEX_JOB_TOKEN"] = token
        if bucket is not None and bucket.access_key and bucket.secret_key:
            self.hidden["STORAGE_ACCESS_KEY"] = bucket.access_key
            self.hidden["STORAGE_SECRET_KEY"] = bucket.secret_key
            if bucket.session_token:
                self.hidden["STORAGE_SESSION_TOKEN"] = bucket.session_token
        if self.code is not None and self.code.get("token"):
            self.hidden["GIT_TOKEN"] = str(self.code["token"])
        #: The compiled directories, on this machine for every job.
        self.caches = [
            (name, os.path.join(config.runs, "_cache", name), library) for name, _variable, library in CACHES
        ]
        for (_name, directory, _library), (_same, variable, _lib) in zip(self.caches, CACHES):
            env[variable] = directory
        #: Where they are shared, for a job that runs a repository: the
        #: stores pulled from, in order, the one pushed to, and what ``ravex
        #: cache`` needs to reach them - the backend's signing service, the
        #: job's owner's own bucket, or nowhere.
        self.cache_reads: List[str] = []
        self.cache_store = ""
        self.cache_env: Dict[str, str] = {}
        #: The wheels the backend's builder compiled for machines like this
        #: one, handed to uv as ``--find-links``, and where to report the ones
        #: this machine still had to compile so the builder does them next.
        self.wheels = os.path.join(config.runs, "_cache", "wheels")
        self.wheels_store = ""
        self.wheels_report = ""
        own = Bucket.from_job(job["storage"]) if job.get("storage") else None
        if self.code is not None and config.cache_service and config.token:
            signed = "sign+%s/api/cache/" % config.backend
            self.cache_reads = [signed + "global", signed + "workspace"]
            self.cache_store = signed + "workspace"
            self.wheels_store = signed + "global"
            self.wheels_report = "%s/api/cache-builds" % config.backend
            self.cache_env = {"SMITH_CACHE_TOKEN": config.token}
            env.update(self.cache_env)
        # Not with temporary keys: a backend hands them scoped to the runs'
        # prefix, and they would not reach _cache.
        elif self.code is not None and own is not None and own.access_key and own.secret_key and not own.session_token:
            self.cache_store = "s3://%s/%s" % (own.bucket, own.prefix_of("_cache"))
            self.cache_reads = [self.cache_store]
            self.cache_env = {"SMITH_CACHE_ACCESS_KEY": own.access_key, "SMITH_CACHE_SECRET_KEY": own.secret_key}
            if own.endpoint:
                self.cache_env["SMITH_CACHE_ENDPOINT"] = str(own.endpoint)
            if own.region:
                self.cache_env["SMITH_CACHE_REGION"] = str(own.region)
            self.cache_env["SMITH_CACHE_PATH_STYLE"] = "true" if own.path_style else "false"
            env.update(self.cache_env)
        env["RAVEX_METRICS_ENDPOINT"] = config.backend
        if config.token:
            env["RAVEX_METRICS_TOKEN"] = config.token
        source = ""
        if job["mode"] in ("resume", "update", "fork"):
            source = job.get("source_store_uri") or ""
            if source.startswith("s3://"):
                if bucket is None or bucket.prefix_in(source) is None:
                    raise RuntimeError(
                        "the store of run %s is %s, and this agent %s"
                        % (
                            job.get("source_run_id"),
                            source,
                            "keeps no bucket" if bucket is None else "keeps its runs in s3://%s" % bucket.bucket,
                        )
                    )
            elif not os.path.isdir(source):
                raise RuntimeError(
                    "the store of run %s is %r, which is not a directory on this machine"
                    % (job.get("source_run_id"), source)
                )
        if job["mode"] in ("resume", "update"):
            # The same run carrying on, from its newest checkpoint: the store
            # is the run's own, and Ravex resumes from what is in it.
            self.run_id = str(job["source_run_id"])
            # A node's part of a run on several resumes its own store, inside
            # the run's, as it wrote it: the same member it was.
            mine = [self.member] if self.member else []
            if bucket is not None and source.startswith("s3://"):
                # Back in the bucket under the same prefix. The staging copy is
                # this machine's; when it is empty - a different node from the
                # one that trained the run - Ravex brings the store down first.
                store = os.path.join(config.runs, self.run_id, *mine)
                env.update(bucket.environment("/".join([bucket.prefix_in(source) or self.run_id] + mine)))
            else:
                store = os.path.join(source, *mine)
            # It must carry on: with no checkpoint in the store, here or in
            # the bucket, the run stops instead of training from step 0 under
            # the same name. On 2026-10-08 a resume on a new node found none,
            # started over and overwrote the run's identity in the bucket.
            env["RAVEX_REQUIRE_RESUME"] = "1"
            if job["mode"] == "update":
                # Parameters somebody changed on a run: the script's
                # new ones win over the checkpoint's, or a new learning rate
                # would be read and then replaced by the old one.
                env["RAVEX_KEEP_HYPERPARAMETERS"] = "1"
        elif job["mode"] in ("fresh", "fork"):
            # The id the backend chose when the job was queued, so it could
            # point at the run before the run existed. A backend
            # from before that has none, and the agent makes one up.
            self.run_id = str(job.get("run_ref") or new_run_id())
            where = "%s/%s" % (self.run_id, self.member) if self.member else self.run_id
            store = os.path.join(config.runs, *where.split("/"))
            if bucket is not None:
                env.update(bucket.environment(bucket.prefix_of(where)))
            env["RAVEX_RUN_ID"] = self.run_id
            env["RAVEX_NAME"] = str(job["name"])
            if job["mode"] == "fork":
                # A run of its own, in a new store, starting from the parent's
                # checkpoint at that step. Ravex reads the parent's
                # store once - from the bucket, when that is where it is - and
                # pins the step there when it can.
                env["RAVEX_FORK_FROM"] = source
                if job.get("from_step") is not None:
                    env["RAVEX_FORK_STEP"] = str(job["from_step"])
        else:
            raise RuntimeError("this agent does not know how to start a %r job" % job["mode"])
        env["RAVEX_STORAGE_PATH"] = store
        self.store = store
        if job.get("config"):
            # Settings chosen for the run. RAVEX_OVERRIDE and not one
            # RAVEX_<NAME> each: those lose to the script's own decorator,
            # and a changed checkpoint_every would be read and ignored.
            env["RAVEX_OVERRIDE"] = json.dumps(job["config"])

        logs = os.path.join(config.runs, "_logs")
        os.makedirs(logs, exist_ok=True)
        self.log_path = os.path.join(logs, "job-%d.log" % self.id)
        self._log = open(self.log_path, "ab")
        if IMAGE:
            self._log.write(("[smith] image %s\n" % IMAGE).encode("utf-8"))
            self._log.flush()
        # Not `tail`: that is the method that reads the last lines for a
        # failure's message.
        self.output = LogTail(self.id, self.log_path, self.hidden)
        #: Every stream of this job's output: the process's, and one per rank
        #: when torchrun runs several (see ``_LAUNCHER``).
        self.outputs = [self.output]
        if config.gpus > 1:
            ranked = os.path.join(logs, "job-%d.rank%%d.log" % self.id)
            env["SMITH_RANK_LOG"] = ranked
            self.outputs += [LogTail(self.id, ranked % rank, self.hidden, rank) for rank in range(config.gpus)]
        serve = outer.get("serve")
        if serve:
            port = int(serve)
            if not 1024 <= port <= 65535:
                raise RuntimeError("port %d is not one a rendezvous can listen on" % port)
            # Before the script, which dials it. Its output beside the job's;
            # its token from the environment, never its command line.
            self._rendezvous_log = open(os.path.join(logs, "job-%d.rendezvous.log" % self.id), "ab")
            self.rendezvous = subprocess.Popen(
                [config.python, "-m", "ravex._cli", "rendezvous", "--host", "0.0.0.0", "--port", str(port)],
                env=env,
                stdout=self._rendezvous_log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            log("job %d: serving the run's rendezvous on port %d, pid %d" % (self.id, port, self.rendezvous.pid))
        launcher = os.path.join(logs, "_launcher.py")
        if config.gpus > 1:
            # torchrun wants a file, not `-c`. One process per GPU, on this
            # machine only; Ravex sees the ranks and checkpoints once for all.
            # A stop reaches every one of them: they share the process group
            # the signal is sent to.
            with open(launcher, "w", encoding="utf-8") as handle:
                handle.write(_LAUNCHER)
        if self.code is not None:
            # The preparer fetches the commit and becomes the training; what
            # it needs in its environment, the token apart from the rest so
            # it can drop it before the training starts.
            os.makedirs(self.code["root"], exist_ok=True)
            env["SMITH_PREPARE"] = json.dumps(
                {
                    "repo": self.code["repo"],
                    "commit": self.code["commit"],
                    "manifest": self.code["manifest"],
                    "script": job["script"],
                    "tree": os.path.join(self.code["root"], self.code["commit"]),
                    "gpus": config.gpus,
                    "launcher": _LAUNCHER,
                    "launcher_file": launcher,
                    "flags": flags(job.get("params") or {}),
                    "caches": self.caches,
                    "cache_script": CACHE_SCRIPT,
                    "cache_reads": self.cache_reads,
                    "cache_store": self.cache_store,
                    "wheels": self.wheels,
                    "wheels_store": self.wheels_store,
                    "wheels_report": self.wheels_report,
                }
            )
            if self.code.get("token"):
                env["SMITH_GIT_TOKEN"] = str(self.code["token"])
            command = [config.python, "-c", _PREPARE]
            workdir = self.code["root"]
        else:
            if config.gpus > 1:
                command = [
                    config.python, "-m", "torch.distributed.run", "--standalone",
                    "--nproc_per_node", str(config.gpus), launcher, script,
                ]
            else:
                command = [config.python, "-c", _LAUNCHER, script]
            command += flags(job.get("params") or {})
            workdir = os.path.dirname(script)
        extra: Dict[str, Any] = {}
        if os.name == "nt":
            # Its own group, so CTRL_BREAK reaches it and not this agent.
            extra["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            extra["start_new_session"] = True
        self.process = subprocess.Popen(
            command,
            env=env,
            cwd=workdir,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **extra,
        )
        log(
            "job %d (%s, %s) started as run %s, pid %d; output in %s"
            % (self.id, job["name"], job["mode"], self.run_id, self.process.pid, self.log_path)
        )

    @staticmethod
    def _checked_code(job: Dict[str, Any], config: Config) -> Dict[str, Any]:
        """The job's repository, commit and manifest, each held to what it may
        be, and where its checkouts go; refused on an agent that runs no code."""
        if not config.code:
            raise RuntimeError("this agent runs only its own scripts (code = true to run a repository's)")
        code = job["code"]
        repo, commit = str(code.get("repo") or ""), str(code.get("commit") or "")
        manifest = str(code.get("manifest") or "")
        if not _REPO.fullmatch(repo) or ".." in repo:
            raise RuntimeError("%r is not a repository this agent fetches: an https address" % repo)
        if not _SHA.fullmatch(commit):
            raise RuntimeError("%r is not a commit: a full SHA, 40 hexadecimal characters" % commit)
        if not _RELATIVE.fullmatch(manifest):
            raise RuntimeError("%r is not a file of a repository" % manifest)
        if not _SCRIPT.fullmatch(str(job.get("script") or "")):
            raise RuntimeError("%r is not a script's name" % job.get("script"))
        # One directory per repository, its commits side by side.
        place = re.sub(r"[^A-Za-z0-9_.-]+", "-", repo.split("://", 1)[1]).strip("-")
        root = os.path.join(config.runs, "_code", place)
        return {**code, "repo": repo, "commit": commit, "manifest": manifest, "root": root}

    def stop(self) -> None:
        now = time.monotonic()
        if self.stop_requested_at is None:
            self.stop_requested_at = now
            log("job %d cancelled by the backend; interrupting pid %d" % (self.id, self.process.pid))
            try:
                if os.name == "nt":
                    os.kill(self.process.pid, signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(self.process.pid, signal.SIGTERM)
            except OSError:
                pass
        elif now - self.stop_requested_at > STOP_GRACE_SECONDS and self.process.poll() is None:
            log("job %d did not stop within %.0fs; killing it" % (self.id, STOP_GRACE_SECONDS))
            self.process.kill()

    def tail(self, lines: int = 6) -> str:
        """The end of the output: on a failure, usually the line that says why.

        Under torchrun that line is in a rank's stream, and the job's own has
        only torchrun's summary: the first rank that ended in a traceback is
        the one quoted."""
        ends = []
        for output in self.outputs:
            try:
                with open(output.path, "rb") as handle:
                    handle.seek(0, os.SEEK_END)
                    handle.seek(max(0, handle.tell() - 4096))
                    ends.append(handle.read().decode("utf-8", "replace"))
            except OSError:
                continue
        if not ends:
            return ""
        text = next((end for end in ends[1:] if "Traceback" in end), ends[0])
        # The last state of each redrawn line, without its colours: this is
        # a message, not a terminal.
        text = "\n".join(line.rsplit("\r", 1)[-1] for line in text.split("\n"))
        text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
        kept = [line for line in text.splitlines() if line.strip()][-lines:]
        # It goes to the backend as the failure's message: no secret in it.
        return redact("\n".join(kept), self.hidden)[-1500:]

    def close(self) -> None:
        self._log.close()

    def push_caches(self, config: Config, settled: Optional[float] = None) -> Optional[subprocess.Popen]:
        """Send the job's cache the kernels it compiled, in the background.

        With the interpreter the preparer recorded, the one the training ran
        in; a job that never got that far compiled nothing worth sending.
        ``settled`` is for a push while the job runs (see ``_PUSH_KERNELS``).
        At a lower priority on Linux, so it takes what the training leaves.
        """
        if not self.cache_store or self.code is None:
            return None
        ready = os.path.join(self.code["root"], self.code["commit"], ".smith-ready")
        try:
            with open(ready, encoding="utf-8") as handle:
                python = json.load(handle)["python"]
        except (OSError, ValueError, KeyError):
            return None
        kernels = [entry for entry in self.caches if entry[0] != "uv"]
        spec: Dict[str, Any] = {"python": python, "store": self.cache_store, "caches": kernels,
                                "cache_script": CACHE_SCRIPT}
        if settled:
            spec["settled"] = settled
        extra: Dict[str, Any] = {}
        if hasattr(os, "nice"):
            extra["preexec_fn"] = lambda: os.nice(10)
        out = open(os.path.join(config.runs, "_logs", "job-%d.cache.log" % self.id), "ab")
        try:
            return subprocess.Popen(
                [config.python, "-c", _PUSH_KERNELS, json.dumps(spec)],
                env=dict(os.environ, **self.cache_env),
                stdout=out,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                **extra,
            )
        finally:
            out.close()

    def push_while_running(self, config: Config) -> None:
        """Send the kernels compiled so far, when it is time and the last push
        has finished; one that cannot start yet (the job is still preparing)
        is tried again at the next interval."""
        now = time.monotonic()
        if now < self.next_push or (self.mid_push is not None and self.mid_push.poll() is None):
            return
        self.next_push = now + KERNEL_PUSH_EVERY_SECONDS
        self.mid_push = self.push_caches(config, settled=KERNEL_SETTLED_SECONDS)

    def stop_rendezvous(self) -> None:
        """Stop serving the run's rendezvous, if this job started one.

        Not when the job ends: the other nodes of the run read their peers'
        addresses from it on every exchange, and the last of them may finish
        minutes after this one. Stopping it with node 0's job took the run's
        slowest node down with it (GPU-186). It goes when this agent takes its
        next job, or with the node.
        """
        if self.rendezvous is not None:
            self.rendezvous.terminate()
            try:
                self.rendezvous.wait(10)
            except subprocess.TimeoutExpired:
                self.rendezvous.kill()
            self._rendezvous_log.close()
            self.rendezvous = None


# ─── the loop ───────────────────────────────────────────────────────


class Agent:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.backend = Backend(config.backend, config.token)
        self.current: Optional[Running] = None
        #: A finished job whose run's rendezvous this agent still serves, for
        #: the nodes of the run that have not finished yet.
        self.serving: Optional[Running] = None
        #: Stores of finished runs whose leftovers still have to reach the
        #: backend, and when to try next.
        self.to_ship: Dict[str, float] = {}
        self.shipping: Optional[tuple] = None
        #: The output of every job still being sent: the running one, and any
        #: that ended while the backend was not answering.
        self.tails: List[LogTail] = []
        #: `ravex ship` reads the token from here, not from its command line,
        #: where any user of the machine can see it.
        self._ship_env = dict(os.environ)
        if config.token:
            self._ship_env["RAVEX_METRICS_TOKEN"] = config.token
        #: Kernels of finished jobs on their way to their caches.
        self.pushing: List[subprocess.Popen] = []
        #: What is wrong with this machine, when its GPUs failed the check:
        #: said in every heartbeat, and no job is taken while it stands.
        self.fault: Optional[str] = None

    def install_runtime(self) -> None:
        """Fetch and install the heavy part of the job's Python before
        anything else, the GPU check included, which imports torch.

        Before the first heartbeat too: a backend that heard this agent would
        hand it a job it could not start yet. Fetching takes a minute or two,
        well inside the time a backend gives a node to come up. One that
        fails is this machine's fault, said in every heartbeat, so the
        backend gives its job to another instead of waiting on this one."""
        runtime = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime.py")
        log("installing the runtime from %s" % self.config.runtime)
        command, env = self.runtime_command(runtime)
        try:
            done = subprocess.run(
                command,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=RUNTIME_SECONDS,
            )
            for line in (done.stdout or "").splitlines():
                log(line)
            if done.returncode != 0:
                said = [line for line in (done.stderr or "").splitlines() if line.strip()]
                self.fault = "the runtime could not be installed: %s" % (said[-1] if said else "no answer")
        except subprocess.TimeoutExpired:
            self.fault = "the runtime was not installed in %d seconds" % RUNTIME_SECONDS
        except OSError as exc:
            self.fault = "the runtime could not start installing: %s" % exc
        if self.fault:
            log("%s; this machine takes no job" % self.fault)

    def runtime_command(self, runtime: str) -> Tuple[List[str], Dict[str, str]]:
        """The command that installs the runtime, and its environment.

        With the backend's cache (``SMITH_CACHE=backend``) the wheels come
        from its read-only ``global`` scope first, through URLs it signs for
        this agent's token, as a job's compiled dependencies do. The token
        goes in the environment, as ``SMITH_CACHE_TOKEN`` like ``cache.py``'s,
        and never on the command line, which any user of the machine can
        read."""
        command = [self.config.python, runtime, "install", "--manifest", self.config.runtime,
                   "--python", self.config.python]
        env = dict(os.environ)
        if self.config.cache_service and self.config.token:
            command += ["--signed", "%s/api/cache/global" % self.config.backend]
            env["SMITH_CACHE_TOKEN"] = self.config.token
        return command, env

    def check_gpus(self) -> None:
        """Run the GPU check once, before the first job."""
        try:
            done = subprocess.run(
                [self.config.python, "-c", _GPU_CHECK, str(self.config.gpus)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=GPU_CHECK_SECONDS,
            )
            # Its own last line; failing that, Python's, for a check that
            # died before it could say anything.
            said = [line for line in (done.stdout or "").splitlines() if line.strip()] or [
                line for line in (done.stderr or "").splitlines() if line.strip()
            ]
            last = said[-1] if said else "no answer"
            if done.returncode != 0:
                self.fault = "the GPU check failed: %s" % last
            else:
                log("GPUs checked: %s" % last)
        except subprocess.TimeoutExpired:
            self.fault = "the GPU check did not finish in %d seconds" % GPU_CHECK_SECONDS
        except OSError as exc:
            self.fault = "the GPU check could not start: %s" % exc
        if self.fault:
            log("%s; this machine takes no job" % self.fault)

    def beat(self) -> None:
        self.backend.call(
            "POST",
            "/api/agents/heartbeat",
            {
                "agent": self.config.name,
                "scripts": sorted(self.config.scripts),
                "model_params": self.config.model_params,
                "params": self.config.params,
                "hardware": self.config.hardware,
                "host": socket.gethostname(),
                "job_id": self.current.id if self.current else None,
                # Only when there is one: a backend that knows nothing of it
                # is not sent a field it would have to ignore.
                **({"fault": self.fault} if self.fault else {}),
                # A finished job's kernels still on their way to the cache.
                **({"pushing": True} if any(p.poll() is None for p in self.pushing) else {}),
            },
        )

    def report(self, job_id: int, state: str, message: Optional[str] = None, run_id: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"state": state}
        if message is not None:
            body["message"] = message
        if run_id is not None:
            body["run_id"] = run_id
        return self.backend.call("PATCH", "/api/jobs/%d" % job_id, body) or {}

    def take(self) -> None:
        job = self.backend.call(
            "POST",
            "/api/jobs/claim",
            {
                "agent": self.config.name,
                "scripts": sorted(self.config.scripts),
                "hardware": self.config.hardware,
                "code": self.config.code,
            },
        )
        if not job:
            return
        if self.serving is not None:
            self.serving.stop_rendezvous()
            self.serving = None
        try:
            self.current = Running(job, self.config)
        except Exception as exc:
            log("job %d cannot start: %s" % (job["id"], exc))
            self.report(int(job["id"]), "failed", "the agent could not start it: %s" % exc)
            return
        self.current.last_report = time.monotonic()
        self.tails.extend(self.current.outputs)
        self.report(self.current.id, "running", run_id=self.current.run_id)

    def watch(self) -> None:
        running = self.current
        assert running is not None
        code = running.process.poll()
        if code is not None:
            running.close()
            if running.rendezvous is not None:
                self.serving = running
                log("job %d ended; still serving its run's rendezvous for the other nodes" % running.id)
            for output in running.outputs:
                output.done = True
            self.send_logs()
            self.current = None
            if running.stop_requested_at is not None:
                state, message = "failed", "stopped on request"
            elif code == 0:
                state, message = "finished", None
            else:
                state, message = "failed", "exit code %d\n%s" % (code, running.tail())
            log("job %d %s (exit code %d)" % (running.id, "stopped" if running.stop_requested_at else state, code))
            self._report_until_heard(running.id, state, message, running.run_id)
            # Always, not only after an outage: it is how the final status
            # arrives when the run's own last attempt did not, and on a store
            # with nothing left it sends the run document again and no points.
            self.to_ship[running.store] = time.monotonic()
            pushing = running.push_caches(self.config)
            if pushing is not None:
                self.pushing.append(pushing)
            return
        if running.stop_requested_at is not None:
            running.stop()
            return
        running.push_while_running(self.config)
        if time.monotonic() - running.last_report >= REPORT_SECONDS:
            running.last_report = time.monotonic()
            answer = self.report(running.id, "running", run_id=running.run_id)
            if answer.get("state") == "cancelled":
                running.stop()

    def send_logs(self) -> None:
        """Send what the jobs printed since the last time; drop a finished
        job's output once all of it has arrived."""
        for tail in list(self.tails):
            try:
                caught_up = tail.send(self.backend)
            except (Unreachable, RuntimeError):
                # Kept where it was: sent again, with the same line numbers,
                # when the backend answers.
                continue
            if caught_up and tail.done:
                self.tails.remove(tail)

    def ship(self) -> None:
        """Send what finished runs left unsent, one store at a time.

        ``ravex ship`` runs as its own process, in the training environment,
        so this agent keeps needing nothing but the standard library - and a
        backend that takes its time does not hold up the queue.
        """
        # Reaped here, where nothing waits on them: a push that is still
        # going does not hold up a job, and one that failed costs only time.
        self.pushing = [process for process in self.pushing if process.poll() is None]
        if self.shipping is not None:
            store, process, first = self.shipping
            code = process.poll()
            if code is None:
                return
            self.shipping = None
            if code == 0:
                del self.to_ship[store]
                if not first:
                    log("what %s had left reached the backend" % store)
            else:
                # Said once per store, not every half minute.
                if first:
                    log(
                        "%s still has metrics the backend has not got; trying again every %.0fs"
                        % (store, SHIP_RETRY_SECONDS)
                    )
                self.to_ship[store] = -(time.monotonic() + SHIP_RETRY_SECONDS)
            return
        now = time.monotonic()
        for store, due in self.to_ship.items():
            # A negative time is a retry: the first failure was already said.
            if abs(due) <= now:
                process = subprocess.Popen(
                    [
                        self.config.python, "-m", "ravex._cli", "ship",
                        "--storage", store, "--endpoint", self.config.backend, "--wait", "60",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    env=self._ship_env,
                )
                self.shipping = (store, process, due >= 0)
                return

    def _report_until_heard(self, job_id: int, state: str, message: Optional[str], run_id: str) -> None:
        # The end of a job is the one report that must not be lost: without it
        # the backend shows a run as going forever.
        while True:
            try:
                self.report(job_id, state, message, run_id)
                return
            except Unreachable:
                time.sleep(POLL_SECONDS * 2)

    def reconcile(self) -> None:
        """Close the jobs the backend still thinks this agent is running.

        A new agent process has no child: whatever it was running died with
        the old one - a crash, a restart of its container. Left alone, those
        jobs would show as running for ever, and a job asked to stop
        would stay "stopping". Said as what it is, with the output that did
        arrive still with the backend.
        """
        try:
            jobs = self.backend.call("GET", "/api/jobs") or []
        except (Unreachable, RuntimeError):
            return
        for job in jobs:
            held = job.get("state") in ("claimed", "running") or (
                job.get("state") == "cancelled" and job.get("finished_at") is None
            )
            if job.get("agent") == self.config.name and held:
                log("job %d was left running by an earlier start of this agent; closing it" % job["id"])
                try:
                    self.report(
                        int(job["id"]),
                        "failed",
                        "the agent restarted while this job was running, and its process went with it",
                    )
                except (Unreachable, RuntimeError):
                    pass

    def run(self) -> None:
        bucket = self.config.storage
        log(
            "agent %s: %d script(s) (%s)%s, %s, backend %s%s, runs in %s"
            % (
                self.config.name,
                len(self.config.scripts),
                ", ".join(sorted(self.config.scripts)),
                " and repositories' code" if self.config.code else "",
                self.config.hardware,
                self.config.backend,
                " with a token" if self.config.token else "",
                self.config.runs
                if bucket is None
                else "s3://%s/%s, staged in %s" % (bucket.bucket, bucket.prefix, self.config.runs),
            )
        )
        if self.config.runtime:
            self.install_runtime()
        if self.config.check_gpus and self.fault is None:
            self.check_gpus()
        self.reconcile()
        polled = 0.0
        while True:
            # Before the heartbeat: it needs no answer from the backend to
            # start, and it is exactly the work to do while there is none.
            self.send_logs()
            if time.monotonic() - polled < POLL_SECONDS:
                # The output every half second, everything else every poll.
                time.sleep(LOG_SECONDS)
                continue
            polled = time.monotonic()
            self.ship()
            try:
                self.beat()
                if self.current is None and self.fault is None:
                    self.take()
                else:
                    self.watch()
            except Unreachable:
                pass
            except RuntimeError as exc:
                log(str(exc))
            time.sleep(LOG_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a backend's training jobs on this machine.")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "smith.toml"),
        help="the agent's configuration (default: smith.toml beside this file)",
    )
    args = parser.parse_args()
    agent = Agent(Config.load(args.config))
    try:
        agent.run()
    except KeyboardInterrupt:
        if agent.current is not None and agent.current.process.poll() is None:
            log("stopping; job %d keeps running as pid %d" % (agent.current.id, agent.current.process.pid))


if __name__ == "__main__":
    main()
