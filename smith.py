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
  it trains, and ``RAVEX_METRICS_TOKEN`` with the job's own token, the one
  the claim's answer carries (``job_token``), never the agent's.

**Code from a repository.** An agent started with ``code = true`` (or
``SMITH_CODE=1``) also takes jobs that carry ``code``: a repository over
HTTPS and a commit as a full SHA. The job's ``script`` is then the path of a
Python file of that commit, the one the person chose to run, instead of one
of the agent's own; nothing of the backend's lives in the repository
(GPU-497). Before it starts, the job's
process fetches that commit alone into ``runs/_code/<repository>/<commit>``,
makes a virtual environment beside it when the repository has a
``requirements.txt`` or a ``pyproject.toml`` (on top of the job's template
environment, so torch and Ravex are there already; with ``uv``) and then runs
the script with the job's parameters as flags. A
commit already fetched is used again. Every step is in the job's output, with
how long it took (GPU-219). It
is still never a command: a repository, a commit and a file of it, each
checked.
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
``MOONCLIP_S3_SECRET_KEY``, never from this file; a job whose store goes to
that bucket is given them, and Moonclip, which writes the store, reads them
there (GPU-267). No other job is (GPU-530).

**What a job is not given (GPU-530).** No variable of this agent's whose
name has a part ``KEY``, ``SECRET``, ``TOKEN``, ``PASSWORD`` or
``CREDENTIALS`` reaches the job, ``ravex ship`` or the kernels' push: not
this agent's token, not the provider's key that deletes the machine
(``RUNPOD_API_KEY``, ``CONTAINER_API_KEY``), not its bucket's keys unless the
job's store is there. What a job needs comes with the job.

**The job's own user (GPU-530).** With ``SMITH_JOB_USER`` (``job_user`` in
the file; ``job`` in both images) the training runs as that user and this
agent as root: a dependency of the job can then neither read the agent's
environment under ``/proc`` - its token, the machine's keys - nor signal it,
nor write what the next job runs: smith's own files, the template
environments under ``_envs``, a repository's checkout and its virtual
environment under ``_code``. The job writes its store, its output (through
the descriptors the agent opened) and its own folder ``_jobs/<id>`` - its
home, its uv cache, its kernels' caches, which are pulled from and pushed to
its owner's cache as before but no longer shared with the machine's other
jobs. ``ravex ship``, the kernels' push and the rendezvous run as the job's
user too, since they run the job's Python; the next job is taken only once
they are over, and starts by killing every process of that user. So does
the job's end, but for its rendezvous: torchrun starts each rank in a
session of its own, which the kill of the job's group does not reach. An agent
that is not root on Linux and is given a job user refuses to start, rather
than run the job as itself. Without the setting, as on a person's own
machine, the job runs as the agent's user, can read its environment, and
shares its caches: the machine is that person's, and so is the job.

**The token.** ``SMITH_TOKEN`` in the environment (or ``token`` in the
configuration, for a machine where that is how secrets arrive): sent as a
bearer token on every call of this agent's own - heartbeat, claim, reports,
logs. Never to the job's process, ``ravex ship`` or the kernels' push, which
run the job's packages: a dependency of the job that had it could claim the
next job, secrets and all. Those get the job's own token from the claim's
answer (``job_token``), which the backend lets write that job's run and its
workspace's cache and nothing else, for a few hours past the job's end. A
claim answered without one, by a backend from before job tokens, fails the
job rather than hand the agent's token over.

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
``interrupted``. One that has not exited a minute later is killed. - its whole
process group, with or without a backend to hear of it. The agent stops its
job the same way, and waits for it, when it is itself told to stop
(``SIGTERM``: ``docker stop``, a machine being given back): the job is in a
process group of its own, and would otherwise die with the container, its
last checkpoint unwritten. A job that ends by itself takes its group with it
too: whatever its process left running there - a TensorBoard, a profiler, a
rank - would keep its GPU memory from the next job (GPU-530). Under torchrun
the ranks are each in a session of their own, out of the group's reach: a
kill reaches them through torchrun's process tree, read just before, and
with a job user of its own every process of that user goes too, what left
the tree included.

**A bucket without keys (GPU-331).** A job's storage on AWS may come with a
region and no keys: the node's instance role writes the bucket, and the run is
given no ``MOONCLIP_S3_*`` for Moonclip (0.3.2 or later) to find the role
instead. Without a region it is refused as a bucket whose keys went missing.

**A spot node taken back (GPU-331).** At start the agent asks the metadata
service (``AWS_EC2_METADATA_SERVICE_ENDPOINT``, else the link-local address)
once, with a second to answer; where nothing does, as anywhere but on EC2, it
asks no more. Otherwise it asks every five seconds for the spot interruption
notice, and on one it stops the running job as a cancel does, so that Ravex
writes its last checkpoint in the two minutes left, and takes no further job.
It reports the job as ``interrupted`` and not ``failed``, for a backend that
carries a run on from a machine taken back; one that does not know the state
refuses it, and the job is then reported ``failed``.

**A node the backend no longer answers (GPU-365).** Budgets and idle nodes
are the backend's to enforce: its reaper gives a machine back. With the
backend down nobody does, and a customer's GPUs would stay on, billed to
their account, for as long as the outage lasts. So with ``alone_minutes``
(``SMITH_ALONE_MINUTES``; 30 in the GPU image, off elsewhere) an agent that
has heard nothing from the backend for that long - no answer at all, only
5xx, which is what a proxy in front of a stopped backend sends, or only a
401 or 403, a token it no longer takes (GPU-530) - stops its
job as a cancel does, so Ravex writes its last checkpoint to the bucket, and
then ends the machine. On RunPod and vast.ai it deletes its own pod or
instance with the key the provider puts in every one for that
(``RUNPOD_POD_ID`` and ``RUNPOD_API_KEY``, ``CONTAINER_ID`` and
``CONTAINER_API_KEY``): a container that merely exits is restarted, or kept
and billed for its disk. Elsewhere it exits with :data:`ALONE_EXIT`, and
what started it decides: on AWS the instance's user data shuts the instance
down once the agent's container ends, which its launch template turns into
a termination. A laptop running smith with the setting off is never touched.

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
cache`` asks the backend, with the job's token, to sign each operation
(``POST /api/cache/{scope}``). It pulls from ``global``, which machines may
only read, then from ``workspace``, the job's owner's own, and pushes only
there. What keeps the cache safe is what the backend lets a workspace do -
read ``global``, write only its own, up to its quota - so a script, or a
library it imports, can reach no further than its owner's own cache.

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
import http.client
import json
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: Seconds between two polls of the queue, and between two heartbeats.
POLL_SECONDS = 2.0

#: Seconds between two reports on a running job. Also how long a cancel on
#: the backend takes to reach the process, at most.
REPORT_SECONDS = 4.0

#: How long a job that was asked to stop has before it is killed: the time
#: Ravex needs for its final checkpoint, written and in the bucket. 60 s cut
#: off a 9 GB one half way (GPU-509, deepseek-1b: about 4 minutes to R2 from
#: vast.ai), and the run kept only its previous checkpoint. The node stays
#: the job's until the process exits, so this is paid for; it bounds a job
#: that ignores the interrupt. A provider taking the machine back ends it at
#: its own notice whatever this says.
STOP_GRACE_SECONDS = 30 * 60.0

#: Why a job was told to stop, as the log says it and, for the first, as the
#: failure's message does: a cancel on the backend, and a notice that the
#: provider takes the machine back.
CANCELLED = "cancelled by the backend"
RECLAIMED = "the provider is reclaiming this machine"

#: The state reported for a job stopped by that notice. A backend that carries
#: runs on from an interrupted machine does so on this word; one that does not
#: know it refuses it with a 4xx, and the report is made again as a failure.
INTERRUPTED = "interrupted"

#: Why a job was told to stop when the backend had been silent for too long,
#: and the exit code of an agent that gave up on it (GPU-365): not 0, which
#: reads as a clean end, and not 1, which reads as a crash.
ALONE = "the backend has not answered for too long"
ALONE_EXIT = 75

#: Seconds between two attempts at sending what a finished run left unsent.
SHIP_RETRY_SECONDS = 30.0

#: Most lines of output one post carries, and most bytes read per poll.
LOG_BATCH = 500
LOG_READ = 1 << 20

#: Seconds between two reads of the jobs' output. Shorter than a poll: a
#: ``print`` on the node should be on the page within a second or two, and
#: a read with nothing new costs no call (GPU-178).
LOG_SECONDS = 0.5

#: The longest between two polls of a backend that does not answer.
POLL_MOST_SECONDS = 30.0

#: The longest the output waits between two tries while the backend takes none.
LOGS_WAIT_MOST = 30.0

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
#: What makes a job's environment from its template (GPU-523), run as a
#: script with this agent's own Python.
ENVIRONMENT_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "environment.py")

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

#: Runs the script the way ``python script.py`` would, with one difference on
#: Windows: a stop request arrives as ``KeyboardInterrupt``. There the only
#: signal one process can send another's group is CTRL_BREAK, whose default is
#: to end the process on the spot - no final checkpoint, and a status left
#: saying "running". Turned into an exception, it unwinds through Ravex like
#: Ctrl-C.
#:
#: **SIGTERM is left alone.** It used to be turned into ``KeyboardInterrupt``
#: too, and then Ravex found a handler already there and stood its own down,
#: saying a stop would not save a checkpoint first. The exception arrives
#: wherever the script is, half way through an optimizer step included, and the
#: final checkpoint written on the way out could hold weights updated in part.
#: Ravex's own handler stops at the next step boundary, saves, and ends with
#: the exit status of a SIGTERM (GPU-294). A script without Ravex is ended by
#: the signal's default, with no stack trace either.
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

if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, _stop)

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


#: What a job's code may be: a repository over HTTPS, a commit as a full SHA,
#: and a Python file of it. Checked before anything runs, so a value cannot
#: become an option of git or a path out of the checkout.
_REPO = re.compile(r"^https://[A-Za-z0-9.-]+(?::\d+)?/[A-Za-z0-9_.~/-]+$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_RELATIVE = re.compile(r"^(?!/)(?!.*\.\.)[A-Za-z0-9_./-]{1,200}\.py$")
#: What a parameter's name may be: it becomes a flag, ``--name``. Checked with
#: ``fullmatch`` like the others, since ``$`` alone also matches before a
#: final newline.
_PARAM = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}")

#: Every job runs this first, as its own process (GPU-523): make the
#: environment its template asks for (``environment.py``), check the GPUs
#: once per machine, then for a repository's job fetch the commit and install
#: its requirements on top, and become the training. In the job's process
#: and not in this agent, so installs of several minutes neither stop the
#: heartbeats nor escape the job's output, and a stop reaches them like it
#: reaches the training. What it made is written to the job's ``ready`` file,
#: which the agent reads for the Python of the rendezvous, of ``ravex ship``
#: and of the kernels' cache.
#:
#: **With a job user of its own** (GPU-530) it starts as the agent's user and
#: makes as that user only what runs no code of the job's: the template's
#: environment, from the versions it names, and the fetch of a commit, which
#: runs nothing of it. What runs the job's code or writes the job's own
#: folder - the GPU check, the copy of the checkout or of the script's
#: folder the training runs in, the install of a repository's requirements,
#: whose builds are the repository's code, every pull and push of a cache -
#: runs as the job's user. A repository's checkout, where its install is
#: built (an editable one links to it), and its virtual environment are kept
#: for the next job with the same commit, so once its install is over every
#: process of the job's user is killed and both are made the agent's, which
#: the job can then read and not write. Only then does the preparer drop to the job's
#: user and become the training.
_PREPARE = r"""
import base64, json, os, shutil, signal, subprocess, sys, time, urllib.request

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


# The job's user, when it has one (GPU-530): what starts a process as that
# user, and nothing when the job runs as the agent's.
job_user = spec["job_user"]
as_job = {"user": job_user[0], "group": job_user[1], "extra_groups": []} if job_user else {}
if job_user:
    # What this process makes as the agent, the job's user must be able to
    # read: an agent started with a stricter umask would leave the job no
    # way into its own environment.
    os.umask(0o022)


# The job's own copy of a folder, made by the job's user, links kept as links.
def copy_for_job(source, target, what):
    run([sys.executable, "-I", "-c", "import shutil, sys; shutil.copytree(sys.argv[1], sys.argv[2], symlinks=True, "
         "ignore=shutil.ignore_patterns('.smith-*'), dirs_exist_ok=True)", source, target], what, **as_job)


# Every path under ``path`` given to the job's user, as the agent: only over
# what the agent made and no process of the job's user has touched yet.
def hand_over(path):
    for folder, dirs, files in os.walk(path):
        for name in [folder] + [os.path.join(folder, entry) for entry in dirs + files]:
            os.lchown(name, job_user[0], job_user[1])


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
                                  stdin=subprocess.DEVNULL, env=dict(os.environ, **cache_env), **as_job)
            if done.returncode != 0:
                say("the %s cache was not %s after %.1f s; what it would have held is compiled here" % (
                    name, "used" if action == "pull" else "sent", time.monotonic() - since))
            else:
                took("%s the %s cache" % ("pulled" if action == "pull" else "pushed", name), since)


# The wheels the backend's builder compiled for this machine's key, from the
# cache's global part, which only that builder writes: no --with, since what a
# wheel was built against is the interpreter, torch and the GPUs, not the uv
# that installs it.
def wheels_pull(python):
    if not spec["wheels_store"]:
        return
    since = time.monotonic()
    done = subprocess.run([python, spec["cache_script"], "pull", "--name", "wheels",
                           "--dir", spec["wheels"], "--store", spec["wheels_store"]],
                          stdin=subprocess.DEVNULL, env=dict(os.environ, **cache_env), **as_job)
    if done.returncode != 0:
        say("the builder's wheels were not used after %.1f s; what they would have been is compiled here"
            % (time.monotonic() - since))
    else:
        took("pulled the builder's wheels", since)


# What uv still had to compile here, by name, for the backend's builder to
# compile from PyPI's source for the next machine like this one. Only names:
# nothing this machine made goes where every machine reads. The wheels uv
# built are the ones `uv cache prune --ci` left in its sdists bucket.
def wheels_report(python):
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
        key = json.loads(subprocess.run([python, spec["cache_script"], "key", "--name", "wheels"],
                                        stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True,
                                        **as_job).stdout)
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


# The template's environment: made once per machine, said in the output.
made = os.path.join(spec["envs"], ".made-%d.json" % os.getpid())
command = [sys.executable, spec["environment_script"], "ensure", "--template", json.dumps(spec["template"]),
           "--root", spec["envs"], "--out", made]
if spec["locks"]:
    command += ["--locks", spec["locks"]]
if spec["signed"]:
    command += ["--signed", spec["signed"]]
if spec["gpu"]:
    command += ["--gpu"]
# uv's cache of the agent's, not the job's own: what makes the template's
# environment is shared by every job with the same template.
if subprocess.run(command, stdin=subprocess.DEVNULL,
                  env=dict(os.environ, UV_CACHE_DIR=spec["shared_uv"], **cache_env)).returncode != 0:
    fail("could not make the job's libraries")
with open(made, encoding="utf-8") as handle:
    made_here = json.load(handle)
python, flavour = made_here["python"], made_here["flavour"]
os.remove(made)
took("libraries ready", began)

# The GPUs, once per machine, with the torch the job will use (GPU-188): a
# machine whose container cannot reach them would train on the CPU, slowly,
# and be paid for as a GPU. Said in the fault file, which the agent reports.
if spec["gpu_check"] and not os.path.isfile(spec["gpu_checked"]):
    since = time.monotonic()
    try:
        # As the training will run: a GPU root reaches and the job's user does
        # not is a machine that cannot train.
        done = subprocess.run([python, "-c", spec["gpu_check_script"], str(spec["gpus"])], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=spec["gpu_check_seconds"], **as_job)
        said = [line for line in (done.stdout or "").splitlines() if line.strip()] or [
            line for line in (done.stderr or "").splitlines() if line.strip()]
        last, ok = (said[-1] if said else "no answer"), done.returncode == 0
    except subprocess.TimeoutExpired:
        last, ok = "it did not finish in %d seconds" % spec["gpu_check_seconds"], False
    if not ok:
        with open(spec["fault_file"], "w", encoding="utf-8") as handle:
            handle.write("the GPU check failed: %s" % last)
        say("the GPU check failed: %s" % last)
        sys.exit(3)
    open(spec["gpu_checked"], "w").close()
    took("GPUs checked: %s" % last, since)

if spec["code"]:
    code = spec["code"]
    tree = code["tree"]
    # One environment per template on top of one checkout per commit.
    ready = os.path.join(tree, ".smith-ready-" + flavour)
    short = code["commit"][:12]
    fetched = os.path.join(tree, ".smith-fetched")
    if os.path.isfile(fetched):
        say("%s at %s, fetched before" % (code["repo"], short))
    else:
        # From nothing each time: a fetch cut short leaves no marker.
        shutil.rmtree(tree, ignore_errors=True)
        os.makedirs(tree)
        say("fetching %s at %s" % (code["repo"], short))
        since = time.monotonic()
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
        if token:
            # In git's environment, where other users of the machine cannot read
            # it, rather than in the address or on the command line.
            basic = base64.b64encode(("x-access-token:" + token).encode()).decode()
            env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.extraHeader",
                       GIT_CONFIG_VALUE_0="Authorization: Basic " + basic)
        run(["git", "init", "-q", tree], "make a repository in " + tree, env=env)
        run(["git", "-C", tree, "fetch", "-q", "--depth", "1", code["repo"], code["commit"]],
            "fetch commit %s; is the repository right, and is a private one given its token?" % short, env=env)
        run(["git", "-C", tree, "checkout", "-q", "--detach", "FETCH_HEAD"], "check out " + short, env=env)
        open(fetched, "w").close()
        took("fetched", since)
    # With a user of its own the job runs in a copy of the checkout, its own
    # to write, with the repository's files where its paths expect them: the
    # checkout is the agent's, and the next job with the same commit runs it.
    own = spec["work"] or tree
    if spec["work"]:
        copy_for_job(tree, own, "copy the checkout for the job")
    if os.path.isfile(ready):
        with open(ready, encoding="utf-8") as handle:
            python = json.load(handle)["python"]
    else:
        # Before the install, with this interpreter: the wheels are built against
        # what it has, which is what the new environment starts from.
        cache("pull", python, ("uv",))
        wheels_pull(python)
        wants = "requirements.txt" if os.path.isfile(os.path.join(tree, "requirements.txt")) else (
            "pyproject.toml" if os.path.isfile(os.path.join(tree, "pyproject.toml")) else None)
        if wants:
            venv = os.path.join(tree, ".smith-venv-" + flavour)
            if job_user:
                # Built in the checkout, by the job's user, who builds what it
                # installs (GPU-530). In the checkout and not the job's copy:
                # an editable install (`-e .`, `pip install -e`) links the
                # environment to the tree it was built from, and the job's copy
                # is gone with the job, while the checkout stays for every job
                # with this commit, sealed with the environment below. Not
                # fetched until then: a job stopped halfway leaves a tree its
                # user could write, which the next job fetches again instead.
                os.remove(fetched)
                shutil.rmtree(venv, ignore_errors=True)
                os.makedirs(venv)
                hand_over(tree)
            # On top of the template's Python, whose torch and Ravex it sees.
            run([python, "-m", "venv", "--system-site-packages", venv], "make a virtual environment", **as_job)
            python = os.path.join(venv, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv, "bin", "python")
            what = ["-r", "requirements.txt"] if wants == "requirements.txt" else ["."]
            say("installing what %s asks for" % wants)
            # A wheel there wins over the sdist of the same version on PyPI, so
            # what the builder compiled is installed instead of compiled again.
            find_links = ["--find-links", spec["wheels"]] if os.path.isdir(spec["wheels"]) and os.listdir(spec["wheels"]) else []
            since = time.monotonic()
            run([shutil.which("uv") or "uv", "pip", "install", "--python", python] + find_links + what,
                "install the dependencies", cwd=tree, **as_job)
            took("installed", since)
            if spec["cache_store"]:
                # What it downloaded is out, what it built stays: the cache is
                # then only the wheels nobody publishes for this machine.
                subprocess.run([shutil.which("uv") or "uv", "cache", "prune", "--ci", "-q"], stdin=subprocess.DEVNULL,
                               **as_job)
                cache("push", python, ("uv",))
                wheels_report(python)
            if job_user:
                # Nothing the install started is left to write them, and then
                # the checkout and its environment are the agent's: the next
                # job with this commit runs both.
                run([sys.executable, "-I", "-c", spec["kill_user"]], "stop what the install left running", **as_job)
                run([sys.executable, "-I", "-c", spec["seal"], tree], "keep the checkout from the job")
                open(fetched, "w").close()
        with open(ready, "w", encoding="utf-8") as handle:
            json.dump({"python": python}, handle)
    root = os.path.realpath(own)
    script = os.path.realpath(os.path.join(root, spec["script"]))
    # Inside the checkout and a file of it: a link that leads out is refused too.
    if os.path.commonpath([script, root]) != root or not os.path.isfile(script):
        fail("%s has no file %s to run" % (short, spec["script"]))
else:
    script = spec["script"]
    if script.startswith("ravex:"):
        # A file of the Ravex release the template names: its examples and
        # benches, which the wheel does not carry.
        out = os.path.join(spec["envs"], ".tree-%d.json" % os.getpid())
        run([sys.executable, spec["environment_script"], "tree", "--ravex", spec["template"]["ravex"],
             "--root", spec["envs"], "--out", out], "fetch Ravex %s's scripts" % spec["template"]["ravex"])
        with open(out, encoding="utf-8") as handle:
            root = os.path.realpath(json.load(handle)["tree"])
        os.remove(out)
        script = os.path.realpath(os.path.join(root, script[len("ravex:"):]))
        if os.path.commonpath([script, root]) != root or not os.path.isfile(script):
            fail("Ravex %s has no %s" % (spec["template"]["ravex"], spec["script"][len("ravex:"):]))
    root = os.path.dirname(script)
    if spec["work"]:
        # A folder of the job's own to work in (GPU-530): the script's is the
        # agent's - a Ravex release's examples, smith's own scripts - and a
        # script writes ./outputs or ./checkpoints where it runs. A copy of
        # it, so what the script reads beside itself is beside it there too.
        # shortcut: the whole folder is copied each job, so a script beside a
        # large dataset should read it by an absolute path.
        work = os.path.realpath(spec["work"])
        if os.path.commonpath([work, os.path.realpath(root)]) == os.path.realpath(root):
            fail("%s is in the folder of the script %s, which a job's copy cannot hold" % (spec["work"], script))
        copy_for_job(root, work, "copy the script's folder for the job")
        script, root = os.path.join(work, os.path.basename(script)), work

# The agent reads it for the Python of the rendezvous, the cache and the run's shipping.
with open(spec["ready"], "w", encoding="utf-8") as handle:
    json.dump({"python": python}, handle)

# The kernels with the interpreter that will compile them: a torch the
# repository installed in its environment is a key of its own.
cache("pull", python, ("triton", "inductor", "tilelang"))

if spec["gpus"] > 1:
    command = [python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(spec["gpus"]),
               spec["launcher_file"], script]
else:
    command = [python, "-c", spec["launcher"], script]
command += spec["flags"]
say("running %s, %.1f s after the job started" % (os.path.basename(script), time.monotonic() - began))
os.chdir(root)
if job_user:
    # The training as the job's user, and nothing of the agent's with it: no
    # group of the agent's, no way back (GPU-530).
    os.setgroups([])
    os.setgid(job_user[1])
    os.setuid(job_user[0])
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
    #: Whether its first job checks that the GPUs work (``_GPU_CHECK``), with
    #: the torch of that job's environment, and the machine takes no other if
    #: they do not. On by default for hardware named ``gpu...``;
    #: ``check_gpus`` / ``SMITH_CHECK_GPUS`` say otherwise.
    check_gpus: bool = False
    #: Whether a job's compiled dependencies are kept in the backend's cache,
    #: through URLs it signs (see "What one machine compiled" above). Off
    #: unless asked for: a backend without one would refuse every pull.
    cache_service: bool = False
    #: The folder of the image's PyTorch locks, one per Python
    #: (``cp312.json``), which ``runtime.py`` installs into a job's
    #: environment (GPU-523); None installs PyTorch's CPU build from its index.
    locks: Optional[str] = None
    #: The Pythons a job's template may ask for here, and the PyTorch it
    #: gets: from the locks, or the file's ``pythons`` and ``torch`` without
    #: them. A backend hands this agent only the jobs it can make.
    pythons: List[str] = field(default_factory=list)
    torch: str = ""
    #: Minutes without a word from the backend after which the job is
    #: stopped and the machine ended (see "A node the backend no longer
    #: answers" above); 0 never.
    alone_minutes: float = 0.0
    #: What this machine costs an hour, in dollars, when whoever rented it
    #: says so: a job is told it as ``RAVEX_PRICE_PER_HOUR``, and the run
    #: reports its cost per token. None says nothing.
    price_per_hour: Optional[float] = None
    #: The user the job's process runs as (``SMITH_JOB_USER``), when not this
    #: agent's own: see "The job's own user" above. Empty runs the job as
    #: this agent's user, which is what a person's own machine does.
    job_user: str = ""

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
            if str(script).startswith("ravex:"):
                # A file of the job's Ravex release (GPU-523), fetched with it.
                if not _RELATIVE.fullmatch(str(script)[len("ravex:"):]):
                    raise SystemExit("scripts.%s: %s is not a file of Ravex's source" % (key, script))
                scripts[key] = str(script)
            else:
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
        locks = resolve(raw["locks"]) if raw.get("locks") else None
        pythons, torch = runtimes_of(locks) if locks else (
            [str(python) for python in raw.get("pythons") or []], str(raw.get("torch") or ""))
        if not pythons or not torch:
            raise SystemExit("%s says no Pythons or PyTorch a job may ask for: locks, or pythons and torch" % path)
        config = cls(
            backend=str(env("SMITH_BACKEND") or raw.get("backend", "http://127.0.0.1:8600")).rstrip("/"),
            name=str(env("SMITH_NAME") or raw.get("name") or socket.gethostname()),
            hardware=str(env("SMITH_HARDWARE") or raw.get("hardware", "local-cpu")),
            gpus=max(1, int(env("SMITH_GPUS") or raw.get("gpus", 1))),
            runs=resolve(raw.get("runs", "runs")),
            # A path is the file's, like every other in it; a bare name
            # (``python3``) is the PATH's to find.
            python=(
                resolve(raw["python"])
                if raw.get("python") and (os.sep in raw["python"] or "/" in raw["python"])
                else raw.get("python") or sys.executable
            ),
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
            locks=locks,
            pythons=pythons,
            torch=torch,
            alone_minutes=max(0.0, float(env("SMITH_ALONE_MINUTES") or raw.get("alone_minutes", 0) or 0)),
            price_per_hour=float(env("SMITH_PRICE_PER_HOUR") or raw.get("price_per_hour", 0) or 0) or None,
            job_user=str(env("SMITH_JOB_USER") or raw.get("job_user") or ""),
        )
        try:
            job_identity(config)
        except RuntimeError as exc:
            # Refused here, not at the first job: an agent told to run jobs
            # as another user that cannot would otherwise claim jobs and fail
            # every one of them.
            raise SystemExit(str(exc))
        return config


def job_identity(config: Config) -> Optional[Tuple[int, int]]:
    """The uid and gid the job's process runs as, or None for this agent's
    own (GPU-530). Never silently the agent's when another was asked for: a
    machine set up to keep the job away from the agent's token and files
    that cannot is refused, not run as if it could."""
    if not config.job_user:
        return None
    if os.name == "nt" or not os.path.isdir("/proc") or os.geteuid() != 0:
        raise RuntimeError(
            "SMITH_JOB_USER is %r, and only an agent running as root on Linux can start a job as "
            "another user; run the agent as root, or unset SMITH_JOB_USER to run jobs as this "
            "agent's own user" % config.job_user
        )
    import pwd

    try:
        entry = pwd.getpwnam(config.job_user)
    except KeyError:
        raise RuntimeError("SMITH_JOB_USER is %r, and this machine has no such user" % config.job_user) from None
    if entry.pw_uid == 0:
        raise RuntimeError("SMITH_JOB_USER is %r, which is root: a job would hold everything the agent does"
                           % config.job_user)
    return entry.pw_uid, entry.pw_gid


def as_user(identity: Optional[Tuple[int, int]]) -> Dict[str, Any]:
    """What ``subprocess.Popen`` takes to start a process as that user, with
    none of this agent's supplementary groups; nothing for the agent's own."""
    if identity is None:
        return {}
    return {"user": identity[0], "group": identity[1], "extra_groups": []}


#: Kills every process of the user it runs as (GPU-530): ``kill(-1)`` reaches
#: all a process may signal, which for a user that is not root is its own,
#: and on Linux never the caller. The preparer runs it as the job's user once
#: a repository's install is over, before the environment it made is sealed.
_KILL_USER = r"""
import os, signal
try:
    os.kill(-1, signal.SIGKILL)
except ProcessLookupError:
    pass
"""

#: Makes every path under the ones named the caller's, and writable by it
#: alone (GPU-530): what the job's user wrote there - a repository's virtual
#: environment, a runs volume from before the job had a user of its own -
#: can no longer be changed by it. Links are handed over and never followed;
#: set-user-ID and set-group-ID bits go too, or a file the job made would
#: become a program of root's that anyone could run. Only once no process of
#: the job's user is left, which could otherwise swap a path for a link
#: between the check and the change.
_SEAL = r"""
import os, stat, sys
uid, gid = os.getuid(), os.getgid()
for top in sys.argv[1:]:
    for folder, dirs, files in os.walk(top):
        for path in [folder] + [os.path.join(folder, name) for name in dirs + files]:
            mode = os.lstat(path).st_mode
            os.lchown(path, uid, gid)
            if not stat.S_ISLNK(mode):
                os.chmod(path, stat.S_IMODE(mode) & 0o755)
"""


def descendants(pid: int) -> List[int]:
    """Every process under ``pid``, by the parents ``/proc`` names; none
    where there is no ``/proc``. What a kill of the job's group misses under
    torchrun, which starts each rank in a session of its own (GPU-530)."""
    if not os.path.isdir("/proc"):
        return []
    children: Dict[int, List[int]] = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                with open("/proc/%s/stat" % name, encoding="utf-8") as handle:
                    parent = int(handle.read().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            children.setdefault(parent, []).append(int(name))
    found: List[int] = []
    todo = [pid]
    while todo:
        for child in children.get(todo.pop(), []):
            found.append(child)
            todo.append(child)
    return found


def kill_user(identity: Optional[Tuple[int, int]], spare: Tuple[int, ...] = ()) -> None:
    """Kill every process of the job's user but those in the sessions
    ``spare`` names (GPU-530). What a ``killpg`` of the job's group does not
    reach: torchrun starts each rank in a session of its own, so its ranks
    and whatever they started outlived a SIGKILL of torchrun and kept their
    GPU memory; and a daemon that left with ``setsid`` would run beside the
    next job, under the same uid, where it could read that job's environment.
    By the real uid in ``/proc/<pid>/status``, which a process cannot hide
    the way it can make its ``/proc`` folder root's."""
    if identity is None:
        return
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/%s/status" % name, encoding="utf-8") as handle:
                uids = next(line for line in handle if line.startswith("Uid:")).split()[1:]
            with open("/proc/%s/stat" % name, encoding="utf-8") as handle:
                session = int(handle.read().rsplit(")", 1)[1].split()[3])
            if str(identity[0]) in uids and session not in spare:
                os.kill(int(name), signal.SIGKILL)
        except (OSError, StopIteration, ValueError, IndexError):
            continue


def runtimes_of(locks: str) -> Tuple[List[str], str]:
    """The Pythons the image has a PyTorch lock for, and that PyTorch, read
    from the locks' file names and torch's wheel in them."""
    pythons, torch = [], ""
    for name in sorted(os.listdir(locks)):
        found = re.fullmatch(r"cp3([0-9]{1,2})\.json", name)
        if not found:
            continue
        pythons.append("3." + found.group(1))
        with open(os.path.join(locks, name), encoding="utf-8") as handle:
            for entry in json.load(handle).get("files", []):
                wheel = re.match(r"torch-([^-]+)-", entry.get("name", ""))
                if wheel:
                    torch = wheel.group(1).replace("%2B", "+")
    return sorted(pythons, key=lambda p: int(p.split(".")[1])), torch


@dataclass
class Bucket:
    """Where the runs' stores go when they do not stay on this machine."""

    type: str
    bucket: str
    prefix: str = ""
    endpoint: Optional[str] = None
    region: Optional[str] = None
    path_style: bool = False
    #: The bucket's keys: those a backend handed with a job, or for this
    #: agent's bucket its MOONCLIP_S3_*, read once at start so that they reach
    #: a job only when its store goes there (GPU-530). Out of the repr, so
    #: that printing a bucket never prints them.
    access_key: Optional[str] = field(default=None, repr=False)
    secret_key: Optional[str] = field(default=None, repr=False)
    #: With temporary keys, the token they sign nothing without: a backend
    #: may hand ones scoped to this bucket, or to its prefix, for a time.
    session_token: Optional[str] = field(default=None, repr=False)

    @classmethod
    def from_job(cls, table: Dict[str, Any]) -> "Bucket":
        """The bucket a backend handed with a job, keys included: that job's
        store goes there instead of this agent's bucket.

        Keys may be left out when the bucket is on AWS and names its region:
        the node then has a role of its own that may write there, and the run
        is given no key variables so that Moonclip takes the role's. Without a
        region there is nothing to tell a bucket on AWS from a backend that
        forgot the keys, and it is refused."""
        kind = str(table.get("type") or "s3")
        if kind not in ("s3", "r2") or not table.get("bucket"):
            raise RuntimeError("the job's storage is not a bucket this agent can use")
        keyed = bool(table.get("access_key") and table.get("secret_key"))
        if not keyed and (table.get("access_key") or table.get("secret_key") or not table.get("region")):
            raise RuntimeError("the job's storage came without its keys")
        return cls(
            type=kind,
            bucket=str(table["bucket"]),
            prefix=str(table.get("prefix") or "").strip("/"),
            endpoint=table.get("endpoint"),
            region=table.get("region"),
            path_style=bool(table.get("path_style")),
            access_key=str(table["access_key"]) if keyed else None,
            secret_key=str(table["secret_key"]) if keyed else None,
            session_token=str(table["session_token"]) if keyed and table.get("session_token") else None,
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
            access_key=os.environ["MOONCLIP_S3_ACCESS_KEY"],
            secret_key=os.environ["MOONCLIP_S3_SECRET_KEY"],
            session_token=os.environ.get("MOONCLIP_S3_SESSION_TOKEN") or None,
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


# ─── the spot interruption notice ───────────────────────────────────


class SpotWatcher:
    """Tells the agent that AWS is taking this machine back.

    A spot instance gets a notice two minutes before it is terminated, on the
    metadata service (IMDSv2) at ``spot/instance-action``; it answers 404
    until then. Two minutes is enough for a run that is told to stop to write
    its last checkpoint, and not for one that finds out when the machine goes.

    ``start`` asks once, with a one-second timeout, and where nothing answers
    - any machine that is not on EC2 - the watcher stays off and costs the
    start nothing. Otherwise a daemon thread asks again every ``poll_seconds``
    and calls ``on_notice`` once. The thread does nothing else: what to stop,
    and how, is the agent's.
    """

    def __init__(self, on_notice: Any, poll_seconds: float = 5.0, timeout: float = 1.0) -> None:
        self.on_notice = on_notice
        self.poll_seconds = poll_seconds
        self.timeout = timeout
        # The variable the AWS SDKs read too, which is also how a test or an
        # emulator stands in for the address that exists only on EC2.
        self.endpoint = (os.environ.get("AWS_EC2_METADATA_SERVICE_ENDPOINT") or "http://169.254.169.254").rstrip("/")
        self.thread: Optional[threading.Thread] = None

    def notice(self) -> bool:
        """Whether a notice is posted. An OSError, a refused or unanswered
        request, means the service is not there to ask."""
        # A token each time: it is two requests to a local address, and one
        # that expired or was lost in a restart of the service cannot go wrong.
        token = urllib.request.Request(
            self.endpoint + "/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        with urllib.request.urlopen(token, timeout=self.timeout) as answer:
            session = answer.read().decode("ascii", "replace").strip()
        action = urllib.request.Request(
            self.endpoint + "/latest/meta-data/spot/instance-action",
            headers={"X-aws-ec2-metadata-token": session},
        )
        try:
            with urllib.request.urlopen(action, timeout=self.timeout):
                return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            raise

    def start(self) -> bool:
        """Probe once; True if the watcher is running."""
        try:
            posted = self.notice()
        except (OSError, ValueError):
            return False
        if posted:
            self.on_notice()
            return True
        self.thread = threading.Thread(target=self._poll, name="spot-watcher", daemon=True)
        self.thread.start()
        return True

    def _poll(self) -> None:
        while True:
            time.sleep(self.poll_seconds)
            try:
                posted = self.notice()
            except (OSError, ValueError):
                # A service that answered at start and misses a beat is not
                # one that has stopped: ask again.
                continue
            if posted:
                self.on_notice()
                return


# ─── the backend ────────────────────────────────────────────────────


class Unreachable(Exception):
    pass


class Refusal(RuntimeError):
    """The backend answered with an error status, kept as ``status`` so that
    a caller can tell a refusal of what was asked (4xx) from a backend that
    is failing for a moment (5xx, as during a deploy)."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


#: What a backend answers a token it does not take. Not an answer to count
#: as heard (GPU-530): a node whose token was revoked, or whose row is gone,
#: polled every two seconds for ever and never gave its machine back.
TOKEN_REFUSED = (401, 403)


class Backend:
    def __init__(self, base: str, token: Optional[str] = None) -> None:
        self.base = base
        self._down_since: Optional[float] = None
        #: When the backend last answered with anything but a 5xx: from the
        #: start, so that one never heard is given the same time.
        self.heard_at = time.monotonic()
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
            if exc.code < 500 and exc.code not in TOKEN_REFUSED:
                # A refusal is an answer: whoever refused is there, and so
                # is its reaper. A 5xx is not one - it is what a CDN or a
                # proxy says for a backend that has stopped - nor is a token
                # refused: that backend will never hear this machine again,
                # and only the agent itself can give it back.
                self.heard_at = time.monotonic()
            raise Refusal("%s %s: %d %s" % (method, path, exc.code, detail), exc.code) from None
        except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as exc:
            # An answer that is not JSON, or cut short, is a proxy's page or a
            # backend going down mid-answer, not an answer: the same as none.
            # Left to rise it ended the agent, and its job with it (GPU-453).
            if self._down_since is None:
                self._down_since = time.monotonic()
                log("backend %s unreachable (%s); jobs already running carry on" % (self.base, exc))
            raise Unreachable(str(exc)) from None
        self.heard_at = time.monotonic()
        if self._down_since is not None:
            log("backend reachable again")
            self._down_since = None
        return answer

    def silent_for(self) -> float:
        """Seconds since the backend last answered."""
        return time.monotonic() - self.heard_at


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


def _run_name(value: Any) -> str:
    """``value`` as a run's id, which is a folder's name under this agent's
    runs and a prefix in its bucket: letters, digits, ``.``, ``-`` and ``_``,
    and never ``..``. ``../../app/smith`` as a run's id wrote its store over
    the agent itself (GPU-465)."""
    name = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or ".." in name:
        raise RuntimeError("%r is not a name a run can have" % name)
    return name


def _inside(root: str, path: str) -> bool:
    """Whether ``path`` is under ``root``, links and ``..`` resolved."""
    root, path = os.path.realpath(root), os.path.realpath(path)
    return path == root or path.startswith(root + os.sep)


#: The names a job's secret may not have (:func:`job_secrets`): smith's own
#: and its libraries' settings, and what the dynamic loader, Python and git
#: take as code to load or commands to run.
SECRET_PREFIXES_NOT_TAKEN = ("RAVEX_", "SMITH_", "MOONCLIP_", "PYTHON", "LD_", "DYLD_", "GIT_")
SECRET_NAMES_NOT_TAKEN = frozenset({"PATH", "HOME", "BROWSER", "BASH_ENV", "ENV", "IFS"})


#: The words that make a variable a secret (GPU-530), as a part of its name
#: between underscores: ``SMITH_TOKEN``, ``RUNPOD_API_KEY``,
#: ``CONTAINER_API_KEY``, ``MOONCLIP_S3_SECRET_KEY``, ``AWS_ACCESS_KEY_ID``.
SECRET_WORDS = frozenset({"KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIALS"})


def without_secrets(environ: Any) -> Dict[str, str]:
    """``environ`` without the secrets in it, for a process that runs code a
    job can influence: the training, ``ravex ship`` and the kernels' push,
    which run in the job's environment, from packages it installed.

    This agent's environment holds what is the machine's and not the job's
    (GPU-530): the provider's key that deletes the pod, which a script would
    have taken along with everything else, and this agent's own bucket keys.
    So a secret is passed on only by name, by smith, where it is the job's:
    its ``secrets``, its bucket's keys - or this agent's, when its store goes
    to this agent's bucket - the run's token, the metrics token. A secret set
    in the shell that starts smith on a laptop goes no further either; one
    that a job needs comes with it, as a secret of the job.

    Not a boundary on its own: the job runs as this agent's user, which can
    read the agent's environment under /proc. What it stops is a script, or
    a library that reports its environment, carrying them off by accident."""
    return {name: value for name, value in environ.items() if not SECRET_WORDS.intersection(name.upper().split("_"))}


def job_secrets(job: Dict[str, Any]) -> Dict[str, str]:
    """The secrets a backend gave this job: names and values for its
    process's environment. A name smith sets itself, or one the libraries
    under the run read their settings from, is dropped rather than let a
    secret replace the run's own settings: a ``MOONCLIP_S3_*`` secret would
    send the run's checkpoints to another bucket.

    So is a name the loader, the interpreter or git reads as what to run
    (GPU-463): ``LD_PRELOAD``, ``PYTHONPATH``, ``GIT_CONFIG_PARAMETERS``,
    ``PATH``. A secret is a value somebody typed into a backend; under one of
    those names it is code on this machine, which runs only the scripts its
    config names unless it said otherwise (``code``). Each one dropped is
    said, so a job that misses it is not a mystery."""
    given = job.get("secrets")
    if not isinstance(given, dict):
        return {}
    kept = {}
    for name, value in given.items():
        name = str(name)
        if name.upper().startswith(SECRET_PREFIXES_NOT_TAKEN) or name.upper() in SECRET_NAMES_NOT_TAKEN:
            log("job %s: the secret %s is not put in its environment: the name is one this machine reads" % (
                job.get("id"), name))
            continue
        kept[name] = str(value)
    return kept


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
            try:
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
            except Refusal as refused:
                if refused.status not in NEVER_TAKEN:
                    raise
                # Sent again it is refused again: the same megabyte, twice a
                # second, for as long as the job runs, and nothing after it
                # ever arrives (GPU-466). Passed by, and said once a batch.
                log("job %d: the backend will not take %d line(s) of output (%d); passed by" % (
                    self.job_id, len(lines[first : first + LOG_BATCH]), refused.status))
        self.offset += end
        self.line += len(lines)
        self.screen = screen
        self.partial = partial
        return not full


#: The backend's answers to a batch of output that the same batch will get
#: again: a job that is not there, lines it cannot read, a body too large.
NEVER_TAKEN = (400, 404, 410, 413, 422)


class Running:
    """A job this agent started, and the process carrying it out."""

    def __init__(self, job: Dict[str, Any], config: Config) -> None:
        #: The rendezvous this agent serves for the run, if it hosts it.
        self.rendezvous: Optional[subprocess.Popen] = None
        #: The port it serves the run's rendezvous on once its environment is
        #: ready, which the rendezvous needs Ravex from; None when it serves none.
        self.serve_port: Optional[int] = None
        try:
            self._start(job, config)
        except BaseException:
            # Started before the script, which dials it: a job that then
            # cannot start - a parameter that is no flag's name - left it
            # listening on the run's port with nobody to stop it (GPU-466).
            self.stop_rendezvous()
            raise

    def _start(self, job: Dict[str, Any], config: Config) -> None:
        self.job = job
        self.id = int(job["id"])
        self.stop_requested_at: Optional[float] = None
        #: Why it was told to stop, for the log and the failure's message.
        self.stop_reason = CANCELLED
        self.last_report = 0.0
        #: When its kernels next go to its cache while it runs, and the push
        #: under way, so that two never overlap.
        self.next_push = time.monotonic() + KERNEL_PUSH_FIRST_SECONDS
        self.mid_push: Optional[subprocess.Popen] = None

        #: The repository's commit this job runs, checked; None for one of
        #: this agent's own scripts.
        self.code = self._checked_code(job, config) if job.get("code") else None
        script = config.scripts[job["script"]] if self.code is None else ""
        #: What it trains with (GPU-523), which this machine must be able to make.
        self.template = self._checked_template(job, config)
        #: The user the job's process runs as, when not this agent's own
        #: (GPU-530), and what starts a process as that user.
        self.identity = job_identity(config)
        self.as_job = as_user(self.identity)
        #: With a user of its own, the job's own folder, the one place under
        #: the runs it may write besides its store: its home and its caches.
        #: Everything else under the runs - the environments, the checkouts
        #: and their virtual environments, the logs - is the agent's, and a
        #: job that could write it would run its code in the next job.
        self.job_dir = os.path.join(config.runs, "_jobs", str(self.id)) if self.identity else ""
        env = without_secrets(os.environ)
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
        # The bucket this job's store goes to: the one the backend handed with
        # it - the job's owner's own - or else this agent's.
        own = Bucket.from_job(job["storage"]) if job.get("storage") else None
        bucket = own if own is not None else config.storage
        # What is this agent's alone is not in the job's environment
        # (``without_secrets``, GPU-465, GPU-530). Its token: the run is given
        # it as RAVEX_METRICS_TOKEN, where it is needed, and nowhere else. Its
        # bucket's keys: only when the job's store goes to that bucket
        # (``Bucket.environment``); one the job brings without keys is written
        # with the machine's role, and with the agent's keys there Moonclip
        # took those instead.
        #: Every value the job's output must not carry back: its secrets, its
        #: bucket's keys, and this agent's token, which a script that prints
        #: its environment would otherwise send to whoever reads the logs.
        self.hidden = dict(self.secrets)
        if config.token:
            self.hidden["SMITH_TOKEN"] = config.token
        #: What the job's process, ``ravex ship`` and the kernels' push call
        #: the backend with: the job's own token, worth its run and its
        #: workspace's cache (GPU-530). Never this agent's, which claims jobs:
        #: a backend that hands none gets the job failed, not the agent's
        #: token in its place.
        self.job_token = str(job.get("job_token") or "")
        if config.token and not self.job_token:
            raise RuntimeError("the backend handed no token for the job (job_token), and this agent gives "
                               "a job none of its own; the backend is older than this agent")
        if self.job_token:
            self.hidden["JOB_TOKEN"] = self.job_token
        if token:
            self.hidden["RAVEX_JOB_TOKEN"] = token
        if bucket is not None and bucket.access_key and bucket.secret_key:
            self.hidden["STORAGE_ACCESS_KEY"] = bucket.access_key
            self.hidden["STORAGE_SECRET_KEY"] = bucket.secret_key
            if bucket.session_token:
                self.hidden["STORAGE_SESSION_TOKEN"] = bucket.session_token
        if self.code is not None and self.code.get("token"):
            self.hidden["GIT_TOKEN"] = str(self.code["token"])
        #: The compiled directories: on this machine for every job, or with a
        #: job user of its own the job's alone (GPU-530). Shared, they were
        #: kernels the next job loads and runs, written by whatever this one
        #: ran: a job's kernels then come from its owner's cache, which is
        #: pulled before it starts, and not from a machine's other jobs.
        cache_root = os.path.join(self.job_dir, "cache") if self.identity else os.path.join(config.runs, "_cache")
        self.caches = [(name, os.path.join(cache_root, name), library) for name, _variable, library in CACHES]
        if self.identity:
            # Its own, which uv and torch keep their caches under.
            env["HOME"] = os.path.join(self.job_dir, "home")
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
        self.wheels = os.path.join(cache_root, "wheels")
        self.wheels_store = ""
        self.wheels_report = ""
        if self.code is not None and config.cache_service and self.job_token:
            signed = "sign+%s/api/cache/" % config.backend
            self.cache_reads = [signed + "global", signed + "workspace"]
            self.cache_store = signed + "workspace"
            self.wheels_store = signed + "global"
            self.wheels_report = "%s/api/cache-builds" % config.backend
            self.cache_env = {"SMITH_CACHE_TOKEN": self.job_token}
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
        if self.job_token:
            env["RAVEX_METRICS_TOKEN"] = self.job_token
        if config.price_per_hour:
            env["RAVEX_PRICE_PER_HOUR"] = str(config.price_per_hour)
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
            elif not _inside(config.runs, source):
                # A path a backend names is this machine's to bound: anywhere
                # else a job could read, and a resume write, what is not a
                # run's (GPU-465).
                raise RuntimeError(
                    "the store of run %s is %r, which is not under this agent's runs"
                    % (job.get("source_run_id"), source)
                )
            elif not os.path.isdir(source):
                raise RuntimeError(
                    "the store of run %s is %r, which is not a directory on this machine"
                    % (job.get("source_run_id"), source)
                )
        if job["mode"] in ("resume", "update"):
            # The same run carrying on, from its newest checkpoint: the store
            # is the run's own, and Ravex resumes from what is in it.
            self.run_id = _run_name(job["source_run_id"])
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
            self.run_id = _run_name(job.get("run_ref") or new_run_id())
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
        if self.identity:
            self._hand_over(config, store)
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
            if self.identity:
                # Made here and given to the job's user, which opens each from
                # its rank: the folder they are in is the agent's.
                for rank in range(config.gpus):
                    open(ranked % rank, "ab").close()
                    os.chown(ranked % rank, *self.identity)
        serve = outer.get("serve")
        if serve:
            port = int(serve)
            if not 1024 <= port <= 65535:
                raise RuntimeError("port %d is not one a rendezvous can listen on" % port)
            # Started once the environment is ready (``serve_when_ready``):
            # the script, which dials it, starts after that too.
            self.serve_port = port
        self.env = env
        #: What the preparer made, and why the machine failed, if it did.
        self.ready_path = os.path.join(logs, "job-%d.ready" % self.id)
        self.fault_path = os.path.join(logs, "job-%d.fault" % self.id)
        for stale in (self.ready_path, self.fault_path):
            if os.path.exists(stale):
                os.remove(stale)
        launcher = os.path.join(logs, "_launcher.py")
        if config.gpus > 1:
            # torchrun wants a file, not `-c`. One process per GPU, on this
            # machine only; Ravex sees the ranks and checkpoints once for all.
            # A stop reaches every one of them: they share the process group
            # the signal is sent to.
            with open(launcher, "w", encoding="utf-8") as handle:
                handle.write(_LAUNCHER)
        # The preparer makes the environment, fetches a repository's commit,
        # and becomes the training; what it needs in its environment, the
        # token apart from the rest so it can drop it before the training.
        if self.code is not None:
            os.makedirs(self.code["root"], exist_ok=True)
        envs = os.path.join(config.runs, "_envs")
        os.makedirs(envs, exist_ok=True)
        env["SMITH_PREPARE"] = json.dumps(
            {
                "template": self.template,
                "envs": envs,
                "environment_script": ENVIRONMENT_SCRIPT,
                "locks": config.locks,
                "signed": "%s/api/cache/global" % config.backend if config.cache_service and self.job_token else None,
                "gpu": config.hardware.startswith("gpu"),
                "gpu_check": config.check_gpus,
                "gpu_check_script": _GPU_CHECK,
                "gpu_check_seconds": GPU_CHECK_SECONDS,
                "gpu_checked": os.path.join(config.runs, "_logs", ".gpus-checked"),
                "fault_file": self.fault_path,
                "ready": self.ready_path,
                "code": None if self.code is None else {
                    "repo": self.code["repo"], "commit": self.code["commit"],
                    "tree": os.path.join(self.code["root"], self.code["commit"]),
                },
                "script": job["script"] if self.code is not None else script,
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
                # The job's user, and what the preparer does as the agent
                # before it becomes the training as that user (GPU-530).
                "job_user": list(self.identity) if self.identity else None,
                "shared_uv": os.path.join(config.runs, "_cache", "uv"),
                # Where it runs: a copy of its checkout or of its script's folder.
                "work": os.path.join(self.job_dir, "work") if self.identity else None,
                "kill_user": _KILL_USER,
                "seal": _SEAL,
            }
        )
        if config.cache_service and self.job_token:
            # The runtime's wheels come from the backend's cache, through URLs
            # it signs for the job's token (``runtime.py --signed``).
            self.cache_env.setdefault("SMITH_CACHE_TOKEN", self.job_token)
        if self.code is not None and self.code.get("token"):
            env["SMITH_GIT_TOKEN"] = str(self.code["token"])
        command = [config.python, "-c", _PREPARE]
        workdir = config.runs
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

    def _hand_over(self, config: Config, store: str) -> None:
        """Make what the job's user writes, and give it that user (GPU-530):
        its own folder, made new, and its run's store. Nothing of a previous
        job's is left running first, so none of it can be holding a path
        that is about to change hands."""
        kill_user(self.identity)
        # Every job's, kernels included: their pushes are over, since no job
        # is taken while one goes on (``Agent.takes_jobs``).
        shutil.rmtree(os.path.join(config.runs, "_jobs"), ignore_errors=True)
        for path in [os.path.join(self.job_dir, "home"), self.wheels] + [entry[1] for entry in self.caches]:
            os.makedirs(path, exist_ok=True)
        relative = os.path.relpath(store, config.runs).split(os.sep)
        if relative[0].startswith(("_", ".")):
            # A run's name never starts so; the agent's own folders do, and a
            # job given one of them as its store would be given them.
            raise RuntimeError("%r is not a run's store under this agent's runs" % store)
        os.makedirs(store, exist_ok=True)
        # The whole run, a resumed one as it was written before included.
        for top in (self.job_dir, os.path.join(config.runs, relative[0])):
            for folder, dirs, files in os.walk(top):
                for path in [folder] + [os.path.join(folder, name) for name in dirs + files]:
                    os.lchown(path, *self.identity)

    @staticmethod
    def _checked_template(job: Dict[str, Any], config: Config) -> Dict[str, str]:
        """The job's template, which this machine must be able to make: a
        Python it has a PyTorch for, and that PyTorch's version (GPU-523)."""
        template = job.get("template")
        if not isinstance(template, dict):
            raise RuntimeError("the job says nothing of the libraries it trains with (its template)")
        python, torch = str(template.get("python") or ""), str(template.get("torch") or "")
        if python not in config.pythons:
            raise RuntimeError("this machine makes Python %s, not %s" % (", ".join(config.pythons), python))
        if torch.split("+", 1)[0] != config.torch.split("+", 1)[0]:
            raise RuntimeError("this machine has PyTorch %s, not %s" % (config.torch, torch))
        return {key: str(template.get(key) or "") for key in ("python", "torch", "ravex", "moonclip")}

    def python(self) -> Optional[str]:
        """The Python the training ran in, once the preparer made it."""
        try:
            with open(self.ready_path, encoding="utf-8") as handle:
                return str(json.load(handle)["python"])
        except (OSError, ValueError, KeyError):
            return None

    def fault(self) -> Optional[str]:
        """What the preparer found wrong with this machine: its GPUs."""
        try:
            with open(self.fault_path, encoding="utf-8") as handle:
                return handle.read().strip() or None
        except OSError:
            return None

    def serve_when_ready(self) -> None:
        """Start serving the run's rendezvous once the environment is made,
        with its Python, which has Ravex. Before the script, which dials it,
        as the preparer starts that only after writing the ready file; the
        other nodes' scripts wait for it, within their deadline."""
        if self.serve_port is None or self.rendezvous is not None:
            return
        python = self.python()
        if python is None:
            return
        logs = os.path.dirname(self.ready_path)
        # Its output beside the job's; its token from the environment, never its command line.
        self._rendezvous_log = open(os.path.join(logs, "job-%d.rendezvous.log" % self.id), "ab")
        self.rendezvous = subprocess.Popen(
            [python, "-m", "ravex._cli", "rendezvous", "--host", "0.0.0.0", "--port", str(self.serve_port)],
            env=self.env,
            stdout=self._rendezvous_log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            # With the job's Python, so as the job's user (GPU-530), and in a
            # session of its own: the one spared when the job's end kills
            # every other process of that user.
            start_new_session=True,
            **self.as_job,
        )
        log("job %d: serving the run's rendezvous on port %d, pid %d" % (self.id, self.serve_port, self.rendezvous.pid))

    @staticmethod
    def _checked_code(job: Dict[str, Any], config: Config) -> Dict[str, Any]:
        """The job's repository, commit and script - a Python file of that
        commit - each held to what it may be, and where its checkouts go;
        refused on an agent that runs no code."""
        if not config.code:
            raise RuntimeError("this agent runs only its own scripts (code = true to run a repository's)")
        code = job["code"]
        repo, commit = str(code.get("repo") or ""), str(code.get("commit") or "")
        if not _REPO.fullmatch(repo) or ".." in repo:
            raise RuntimeError("%r is not a repository this agent fetches: an https address" % repo)
        if not _SHA.fullmatch(commit):
            raise RuntimeError("%r is not a commit: a full SHA, 40 hexadecimal characters" % commit)
        if not _RELATIVE.fullmatch(str(job.get("script") or "")):
            raise RuntimeError("%r is not a Python file of a repository" % job.get("script"))
        # One directory per repository, its commits side by side.
        place = re.sub(r"[^A-Za-z0-9_.-]+", "-", repo.split("://", 1)[1]).strip("-")
        root = os.path.join(config.runs, "_code", place)
        return {**code, "repo": repo, "commit": commit, "root": root}

    def stop(self, reason: str = CANCELLED) -> None:
        now = time.monotonic()
        if self.stop_requested_at is None:
            self.stop_requested_at = now
            self.stop_reason = reason
            log("job %d %s; interrupting pid %d" % (self.id, reason, self.process.pid))
            try:
                if os.name == "nt":
                    os.kill(self.process.pid, signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(self.process.pid, signal.SIGTERM)
            except OSError:
                pass
        elif now - self.stop_requested_at > STOP_GRACE_SECONDS and self.process.poll() is None:
            log("job %d did not stop within %.0fs; killing it" % (self.id, STOP_GRACE_SECONDS))
            self.kill()

    def kill(self) -> None:
        """Kill the job's whole process group, as the interrupt went to it:
        the process is torchrun, and killing it alone left its ranks alive,
        holding the GPUs under a job the backend counts as ended (GPU-453)."""
        # The whole tree first, while torchrun still holds it together: its
        # ranks are each in a session of their own, out of the group's reach,
        # and once torchrun is dead they no longer descend from it (GPU-530).
        # Not once it was reaped: its pid may be another process's by then.
        tree = descendants(self.process.pid) if os.name != "nt" and self.process.poll() is None else []
        try:
            if os.name == "nt":
                self.process.kill()
            else:
                os.killpg(self.process.pid, signal.SIGKILL)
        except OSError:
            pass
        for pid in tree:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        # torchrun's ranks are in sessions of their own, out of the group's
        # reach: with a job user, everything of that user goes but the run's
        # rendezvous, which the other nodes still need (GPU-530).
        rendezvous = getattr(self, "rendezvous", None)
        kill_user(getattr(self, "identity", None), (rendezvous.pid,) if rendezvous is not None else ())

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
        python = self.python()
        if python is None:
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
                # In the job's environment, from its packages: no secret of this
                # agent's but the cache's own (GPU-530).
                env=dict(without_secrets(os.environ), **self.cache_env),
                stdout=out,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                # As the job's user, which owns the kernels it reads and whose
                # Python it runs (GPU-530): as the agent, a link the job left
                # among them would be followed with the agent's rights.
                **extra,
                **self.as_job,
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
        #: Per store, the Python its job trained in, which ``ravex ship`` runs
        #: with: this agent's own has no Ravex (GPU-523).
        self.ship_python: Dict[str, str] = {}
        self.shipping: Optional[tuple] = None
        #: The output of every job still being sent: the running one, and any
        #: that ended while the backend was not answering.
        self.tails: List[LogTail] = []
        #: After a batch of output the backend did not take: how long the
        #: next try waits, doubled each time, and until when.
        self.logs_wait = 0.0
        self.logs_after = 0.0
        #: `ravex ship` reads the token from its environment, not from its
        #: command line, where any user of the machine can see it: the job's
        #: own, per store (``ship_token``), and no secret of this agent's,
        #: since it runs the job's Ravex, from the job's environment (GPU-530).
        self._ship_env = without_secrets(os.environ)
        #: Per store, the token of the job that trained it.
        self.ship_token: Dict[str, str] = {}
        #: Kernels of finished jobs on their way to their caches.
        self.pushing: List[subprocess.Popen] = []
        #: What is wrong with this machine, when its GPUs failed the check:
        #: said in every heartbeat, and no job is taken while it stands.
        self.fault: Optional[str] = None
        #: Set, from the spot watcher's thread, when AWS has said it takes this
        #: machine back. An event and not a call into the job: the main loop
        #: is the only one that touches the process.
        self.reclaimed = threading.Event()
        #: Whether the jobs an earlier start of this agent left running have
        #: been closed; none is taken before, or its own would be among them.
        self.reconciled = False
        self.spot: Optional[SpotWatcher] = None
        #: The user jobs run as, when not this agent's (GPU-530).
        self.identity = job_identity(config)

    def takes_jobs(self) -> bool:
        """Whether to claim a job: not with a fault, and not on a machine
        that is about to go, where it would start work to lose.

        With a job user of its own, not while what a finished job left -
        ``ravex ship``, its kernels' push - still runs as that user either
        (GPU-530): each runs that job's Python, and beside the next job,
        under the same uid, it could read that job's environment. The next
        job starts by killing every process of that user."""
        if self.identity is not None and (
            self.shipping is not None or any(process.poll() is None for process in self.pushing)
        ):
            return False
        return self.fault is None and not self.reclaimed.is_set() and self.reconciled

    def start_spot_watch(self) -> None:
        self.spot = SpotWatcher(self.on_spot_notice)
        if self.spot.start():
            log("watching for a spot interruption notice")

    def on_spot_notice(self) -> None:
        if not self.reclaimed.is_set():
            log("the provider will reclaim this machine soon")
            self.reclaimed.set()

    def alone(self) -> bool:
        """Whether the backend has been silent for longer than the agent
        waits for it (GPU-365)."""
        limit = self.config.alone_minutes
        return limit > 0 and self.backend.silent_for() > limit * 60

    def give_up(self) -> None:
        """Stop the job, end the machine, and exit.

        The job is stopped as a cancel stops it, and waited for: Ravex
        writes its last checkpoint to the bucket, which does not need the
        backend. Not reported: there is nobody to report to, and when the
        backend comes back its reaper finds the node gone and the run where
        the checkpoint left it."""
        log(
            "the backend has not answered for %.0f minutes; stopping and giving this machine back"
            % (self.backend.silent_for() / 60)
        )
        self.stop_job(ALONE)
        self.end_machine()
        raise SystemExit(ALONE_EXIT)

    def end_machine(self) -> None:
        """Delete this pod or instance with the key its provider gave it for
        that, where there is one. Every failure is said and left: the exit
        that follows is the last resort, and on AWS it is enough."""
        env = os.environ.get
        if env("RUNPOD_POD_ID") and env("RUNPOD_API_KEY"):
            pod = urllib.parse.quote(env("RUNPOD_POD_ID", ""))
            # Both of RunPod's REST APIs: the platform drives pods with v2,
            # and RunPod's own documentation shows the pod's key with v1,
            # which RunPod retires on 2026-11-15. Which one accepts a key
            # scoped to the pod is not confirmed yet (GPU-365).
            urls = ["https://api.runpod.io/v2/pods/" + pod, "https://rest.runpod.io/v1/pods/" + pod]
            self._delete_self("RunPod pod", urls, env("RUNPOD_API_KEY", ""))
        elif env("CONTAINER_ID") and env("CONTAINER_API_KEY"):
            instance = urllib.parse.quote(env("CONTAINER_ID", ""))
            self._delete_self("vast.ai instance", ["https://console.vast.ai/api/v0/instances/%s/" % instance],
                              env("CONTAINER_API_KEY", ""))

    def _delete_self(self, what: str, urls: List[str], key: str) -> None:
        for url in urls:
            request = urllib.request.Request(
                url, method="DELETE", headers={"Authorization": "Bearer " + key, "User-Agent": "smith"}
            )
            try:
                with urllib.request.urlopen(request, timeout=TIMEOUT):
                    pass
            except urllib.error.HTTPError as exc:
                log("%s: deleting it was refused at %s: %d" % (what, url, exc.code))
                continue
            except (urllib.error.URLError, OSError) as exc:
                log("%s: could not ask to delete it at %s: %s" % (what, url, exc))
                continue
            log("%s: deletion asked for" % what)
            return

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
                "job_id": self.holding(),
                # Only when there is one: a backend that knows nothing of it
                # is not sent a field it would have to ignore.
                **({"fault": self.fault} if self.fault else {}),
                # A finished job's kernels still on their way to the cache.
                **({"pushing": True} if any(p.poll() is None for p in self.pushing) else {}),
            },
        )

    def holding(self) -> Optional[int]:
        """The job this machine is still busy with: the one running, or one
        whose output has not all reached the backend yet.

        A backend gives a machine back once its agent says it holds no job.
        Said as soon as the process exited, the machine went while the last
        lines of the job's output - its shutdown, the reason a run stopped
        the way it did - were still here to be sent: on 2026-11-23 a run
        stopped from the console lost every line after the signal, on both
        of its nodes (GPU-504)."""
        if self.current is not None:
            return self.current.id
        return next((tail.job_id for tail in self.tails), None)

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
                # What a job's template may ask for here (GPU-523).
                "pythons": self.config.pythons,
                "torch": self.config.torch,
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
            # Until heard, like the end of a job that ran (GPU-530): a 5xx or no
            # answer here left the job "claimed" for good, since only a start
            # of the agent closes what it holds.
            self._report_until_heard(int(job["id"]), "failed", "the agent could not start it: %s" % exc, None)
            return
        self.current.last_report = time.monotonic()
        self.tails.extend(self.current.outputs)
        self.report(self.current.id, "running", run_id=self.current.run_id)

    def watch(self) -> None:
        running = self.current
        assert running is not None
        running.serve_when_ready()
        code = running.process.poll()
        if code is not None:
            # What the job left behind in its group goes with it (GPU-530): a
            # TensorBoard, a profiler or a rank that outlived the main process
            # kept its CUDA context, and the next job on the GPU ran out of
            # memory. The job was started in a session of its own
            # (start_new_session), so its group is never this agent's.
            running.kill()
            running.close()
            fault = running.fault()
            if fault:
                # The GPU check, run by the preparer: this machine takes no
                # job, and its heartbeat says why.
                self.fault = fault
                log("%s; this machine takes no job" % fault)
            if running.rendezvous is not None:
                self.serving = running
                log("job %d ended; still serving its run's rendezvous for the other nodes" % running.id)
            for output in running.outputs:
                output.done = True
            self.send_logs()
            self.current = None
            if running.stop_requested_at is not None and running.stop_reason == RECLAIMED:
                state, message = INTERRUPTED, "the provider reclaimed the machine"
            elif running.stop_requested_at is not None:
                state, message = "failed", "stopped on request"
            elif code == 0:
                state, message = "finished", None
            elif fault:
                state, message = "failed", fault
            else:
                state, message = "failed", "exit code %d\n%s" % (code, running.tail())
            log("job %d %s (exit code %d)" % (running.id, "stopped" if running.stop_requested_at else state, code))
            try:
                self._report_until_heard(running.id, state, message, running.run_id)
            except Refusal as exc:
                # A backend that refuses the end (a 404, a 401) still gets the
                # run's metrics below (GPU-530): raised, it skipped them, and
                # what Ravex kept on disk through an outage was never sent.
                log("job %d: the backend refused its end (%s)" % (running.id, exc))
            # Always, not only after an outage: it is how the final status
            # arrives when the run's own last attempt did not, and on a store
            # with nothing left it sends the run document again and no points.
            python = running.python()
            if python is not None:
                # With the training's own Python, which has Ravex.
                self.to_ship[running.store] = time.monotonic()
                self.ship_python[running.store] = python
                self.ship_token[running.store] = running.job_token
            pushing = running.push_caches(self.config)
            if pushing is not None:
                self.pushing.append(pushing)
            return
        if self.reclaimed.is_set() and running.stop_requested_at is None:
            # The same interruption as a cancel, so that Ravex writes its last
            # checkpoint in the time the notice leaves.
            running.stop(RECLAIMED)
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

    def keep_stopping(self) -> None:
        """What stopping a job needs of this machine alone: the provider's
        notice that the machine is going, and the kill once the grace is
        over. Before the heartbeat and whatever it answers (GPU-466): both
        waited on a backend that answered, so with the backend down a spot
        notice stopped nothing, and a job that ignored its interrupt was
        never killed."""
        running = self.current
        if running is None or running.process.poll() is not None:
            return
        if self.reclaimed.is_set() and running.stop_requested_at is None:
            running.stop(RECLAIMED)
        elif running.stop_requested_at is not None:
            running.stop(running.stop_reason)

    def stop_job(self, reason: str) -> None:
        """Stop the job as a cancel stops it, and wait for it: Ravex writes
        its last checkpoint to the bucket, which needs no backend."""
        running = self.current
        if running is not None:
            running.stop(reason)
            while running.process.poll() is None:
                time.sleep(LOG_SECONDS)
                # A second call kills a job still there after the grace.
                running.stop(reason)
            running.close()
            log("job %d stopped" % running.id)
        for serving in (self.serving, running):
            if serving is not None and getattr(serving, "rendezvous", None) is not None:
                serving.stop_rendezvous()

    def send_logs(self) -> None:
        """Send what the jobs printed since the last time; drop a finished
        job's output once all of it has arrived."""
        if time.monotonic() < self.logs_after:
            return
        for tail in list(self.tails):
            try:
                caught_up = tail.send(self.backend)
            except (Unreachable, RuntimeError):
                # Kept where it was: sent again, with the same line numbers,
                # when the backend answers - not twice a second meanwhile,
                # each time the megabyte it could not take (GPU-466).
                self.logs_wait = min(LOGS_WAIT_MOST, max(LOG_SECONDS, self.logs_wait * 2))
                self.logs_after = time.monotonic() + self.logs_wait
                return
            if caught_up and tail.done:
                self.tails.remove(tail)
        self.logs_wait = 0.0

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
                self.ship_python.pop(store, None)
                self.ship_token.pop(store, None)
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
        if self.identity is not None and self.current is not None:
            # A finished run's Python is not started beside the next job,
            # under the same user (GPU-530): it is tried again once that ends.
            return
        now = time.monotonic()
        for store, due in self.to_ship.items():
            # A negative time is a retry: the first failure was already said.
            if abs(due) <= now:
                process = subprocess.Popen(
                    [
                        self.ship_python[store], "-m", "ravex._cli", "ship",
                        "--storage", store, "--endpoint", self.config.backend, "--wait", "60",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    env=dict(self._ship_env, RAVEX_METRICS_TOKEN=self.ship_token.get(store, "")),
                    # The job's Python, reading the store the job wrote: as
                    # the job's user, never the agent's (GPU-530).
                    **as_user(self.identity),
                )
                self.shipping = (store, process, due >= 0)
                return

    def _report_until_heard(self, job_id: int, state: str, message: Optional[str], run_id: Optional[str]) -> None:
        # The end of a job is the one report that must not be lost: without it
        # the backend shows a run as going forever.
        while True:
            try:
                self.report(job_id, state, message, run_id)
                return
            except Unreachable:
                pass
            except Refusal as exc:
                if exc.status < 500:
                    if state != INTERRUPTED or exc.status not in (400, 422):
                        raise
                    # A backend older than this state refuses it (a 400, or
                    # a 422 from its validation), and asking again would be
                    # refused for ever: the job ended, and that is what it
                    # is told.
                    log("job %d: the backend does not take %r (%s); reporting it failed" % (job_id, state, exc))
                    state = "failed"
                    continue
                # A backend failing for a moment, as during a deploy: the
                # same word again, whatever the word. Raised, a job that had
                # finished stayed "running" on the backend for good, since
                # nothing here would say it again (GPU-474).
            if self.alone():
                # Nobody to tell for longer than this agent waits: the loop
                # gives the machine back, and the backend's reaper finds the
                # node gone and settles the job.
                log("job %d: the backend never heard it %s" % (job_id, state))
                return
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
        except (Unreachable, RuntimeError) as exc:
            # Asked again at the next poll, and no job is taken until it has
            # been answered (GPU-474): asked once, an agent that started
            # while the backend was away left those jobs running for ever.
            # A refusal is an answer, and asking again would get the same.
            if isinstance(exc, Refusal) and exc.status < 500:
                self.reconciled = True
            return
        self.reconciled = True
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

    def seal_runs(self) -> None:
        """Make the runs and the folders every job reads this agent's, where
        they are not (GPU-530): a runs volume from an image whose jobs ran as
        the agent's user, which the job's user now is, held environments,
        checkouts and logs a job could rewrite. Each folder once, and only
        when it is not the agent's already; a run's store is given to the
        job's user when a job writes it."""
        runs = self.config.runs
        os.makedirs(runs, exist_ok=True)
        tops = [runs] + [os.path.join(runs, name) for name in ("_envs", "_code", "_cache", "_logs")]
        stale = [top for top in tops if os.path.exists(top) and os.lstat(top).st_uid != os.getuid()]
        if stale:
            log("making %s this agent's alone; a job's user wrote them" % ", ".join(stale))
            kill_user(self.identity)
            subprocess.run([sys.executable, "-I", "-c", _SEAL] + stale, check=True, stdin=subprocess.DEVNULL)

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
        if self.identity is not None:
            self.seal_runs()
        self.reconcile()
        self.start_spot_watch()
        polled = 0.0
        #: How long until the next poll: POLL_SECONDS while the backend
        #: answers, longer each time it does not.
        every = POLL_SECONDS
        while True:
            # Before the heartbeat: it needs no answer from the backend to
            # start, and it is exactly the work to do while there is none.
            self.send_logs()
            self.keep_stopping()
            if time.monotonic() - polled < every:
                # The output every half second, everything else every poll.
                time.sleep(LOG_SECONDS)
                continue
            polled = time.monotonic()
            self.ship()
            try:
                self.beat()
                every = POLL_SECONDS
                if not self.reconciled:
                    self.reconcile()
                if self.current is None:
                    # None taken on a machine that is going, or with a
                    # fault: the loop still beats, sends logs and serves a
                    # rendezvous, with nothing to watch.
                    if self.takes_jobs():
                        self.take()
                else:
                    self.watch()
            except Unreachable:
                # Not every two seconds while it is away, nor all at the
                # same moment when it is back: hundreds of nodes asking in
                # step is what keeps a backend that just restarted down
                # (GPU-474).
                every = backoff(every)
            except RuntimeError as exc:
                log(str(exc))
                if isinstance(exc, Refusal) and (exc.status >= 500 or exc.status in TOKEN_REFUSED):
                    every = backoff(every)
            if self.alone():
                self.give_up()
            time.sleep(LOG_SECONDS)


def backoff(every: float) -> float:
    """The wait after ``every`` while the backend does not answer: doubled,
    at most :data:`POLL_MOST_SECONDS`, and spread by half either way so the
    agents of a fleet do not come back together."""
    return min(POLL_MOST_SECONDS, every * 2) * random.uniform(0.5, 1.5)


class Terminated(BaseException):
    """SIGTERM reached the agent; raised where it was, out of the loop."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a backend's training jobs on this machine.")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "smith.toml"),
        help="the agent's configuration (default: smith.toml beside this file)",
    )
    args = parser.parse_args()
    agent = Agent(Config.load(args.config))

    def terminated(_signal: int, _frame: Any) -> None:
        raise Terminated()

    # `docker stop`, or a provider ending the machine: the job is in a
    # process group of its own, so the signal reaches the agent alone, and
    # without this the job was killed with its container a moment later,
    # its last checkpoint unwritten (GPU-466).
    signal.signal(signal.SIGTERM, terminated)
    try:
        agent.run()
    except Terminated:
        log("asked to stop; stopping the job first")
        agent.stop_job(CANCELLED)
        raise SystemExit(143)
    except KeyboardInterrupt:
        if agent.current is not None and agent.current.process.poll() is None:
            log("stopping; job %d keeps running as pid %d" % (agent.current.id, agent.current.process.pid))


if __name__ == "__main__":
    main()
