# smith

An agent for training nodes. It runs where the training is — a laptop's venv, a container,
a rented GPU node — takes jobs from a backend's queue, starts them, streams
their output and says how they went. The backend never starts a training itself.

It runs **scripts it was given, never commands it was sent**: a job names a script
from the `[scripts]` table of smith's own configuration, and its parameters become
`--name value` flags. `smith.py` explains the rest — environment, tokens, where a
run's store lives, stopping, and what happens to metrics a run could not send.

Only the standard library, Python 3.11 or newer. Three files: `smith.py`, the
agent; `cache.py`, the cache of compiled wheels and kernels, which smith runs as
a script with the job's own interpreter (it was `ravex cache` until Ravex
0.8.0; a bucket store goes through `moonclip.Bucket`, which the job's
environment has); and `builder.py`, which compiles for that cache what nodes
reported having had to compile.

## Running it

On the host, from the training environment's Python:

```bash
python smith.py --config smith.toml
```

In a container on CPU (`Dockerfile`, config `smith.docker.toml`), or on a
provider's GPU node (`Dockerfile.gpu`, config `smith.gpu.toml`). Both images take
Moonclip and Ravex from one of two places:

```bash
# From PyPI, at MOONCLIP_VERSION / RAVEX_VERSION: this repository alone.
docker build -t smith .

# Built from checkouts, for a Ravex or a Moonclip ahead of PyPI.
docker build -t smith --build-arg LIBS=source \
  --build-context moonclip=../moonclip --build-context ravex=../ravex .
```

The GPU image is published by pushing to the `deploy` branch
(`git push origin main:deploy`): `.github/workflows/image.yml` builds it from
PyPI and pushes it as `ghcr.io/<owner>/smith-gpu:<commit>`, smith's short
commit, and as `:latest`, then checks both tags are in the registry and that
no layer is over 250 MB.

**The GPU image does not carry torch.** torch with its CUDA libraries is
about 7 GB, and a host's docker that drops a layer pulls it again from its
first byte: on slow hosts the image never arrived. So the image carries only
the lock, `runtime.json` (each wheel's name, address and sha256, resolved
from `runtime.in` when it is built), and `runtime.py` fetches the wheels when
the node starts, before the first heartbeat: in pieces, sixteen at once, each
retried on its own, each file checked against its hash, then installed with
uv. Each file comes from the first source that has it: the backend's cache
when the node uses it (`SMITH_CACHE=backend`), at `runtime/<wheel file name>`
in its read-only `global` scope, through a URL the backend signs for each file
(the protocol of `cache.py`, below) and signs again when it expires mid-download;
else a mirror if `SMITH_RUNTIME_URL` is set, where each wheel is at its own
file name; else the index the wheel was resolved from. Filling the cache is the
backend's side: this repository's CI holds no bucket key.

## What it expects of a backend

JSON over HTTP, with `Authorization: Bearer <SMITH_TOKEN>` when a token is set:

| call | what for |
|---|---|
| `POST /api/agents/heartbeat` | smith is alive, with its name, hardware and the scripts it can run; `fault`, when its GPUs failed the check below |
| `POST /api/jobs/claim` | the next queued job this node can run, or nothing |
| `PATCH /api/jobs/{id}` | a job's state: running, finished, failed, and the run id |
| `POST /api/jobs/{id}/logs` | the job's output, numbered line by line |
| `GET /api/jobs` | at start, jobs this node left running before a restart |
| `POST /api/cache/{scope}` | only with `cache = "backend"`: `cache.py` asks it to sign each operation on the compiled-dependency cache, `global` (read) and `workspace` (read and write); see its `sign+https://` store |

A claimed job may carry `secrets`, an object of names and values - an
`HF_TOKEN`, a W&B key. smith puts them in that job's process environment only,
never its own, and never one named `RAVEX_*`, `SMITH_*` or `MOONCLIP_*`; and it replaces
their values with `[secret NAME]` in the output lines and failure messages it
sends back.

A claimed job may also carry `storage`, a bucket with its keys
(`type`, `bucket`, `prefix`, `endpoint`, `region`, `path_style`, `access_key`,
`secret_key`, and `session_token` when the keys are temporary ones): that
job's store goes there instead of the agent's own bucket, the keys reach only
its process (as `MOONCLIP_S3_*`), and they are kept out of what is sent back like
secrets.

A claimed job that is one node's part of a run trained on several (Ravex's
outer loop) carries `outer`: `member`, the name of this node's store inside the
run's; `token`, the run's `RAVEX_JOB_TOKEN`, set in the job's environment and
kept out of what is sent back; and, on the node that hosts it, `serve`, a port
on which smith runs `ravex rendezvous` beside the script and keeps it up after,
for the nodes still training, until it takes its next job.
The rendezvous address and the run's other settings come in the job's
`config`, as for any run.

With `code = true` in its configuration (or `SMITH_CODE=1`), smith claims
with `"code": true` and may get a job that carries `code`: `repo` (an https
address), `commit` (a full SHA), `manifest` (a file in the repository with a
`[scripts]` table, like smith's own) and, for a private repository, `token`.
The job's `script` is then one of the manifest's. smith fetches that commit
alone, installs its `requirements.txt` or `pyproject.toml` in a virtual
environment on top of its own packages (with `uv` when there is one), and runs
the script with the job's parameters. It runs whatever the repository holds,
so leave it off on a machine shared with people the repository is not theirs.
The token reaches git through its environment only, and never the run.

The runs are [Ravex](https://github.com/gpuzero-dev/ravex) scripts: smith hands
them their store, name and the backend's address through `RAVEX_*` variables,
and ships what a finished run could not send with `ravex ship`.

## Configuration

On a node, what differs between machines comes from the environment:
`SMITH_BACKEND`, `SMITH_TOKEN`, `SMITH_NAME`,
`SMITH_HARDWARE`, `SMITH_GPUS`, `SMITH_STORAGE_*`, and the
bucket's `MOONCLIP_S3_ACCESS_KEY` / `MOONCLIP_S3_SECRET_KEY`, which Moonclip
reads itself.
`SMITH_IMAGE`, which the GPU image sets to its own tag, is written as the first
line of every job's output: `[smith] image ghcr.io/.../smith-gpu:<commit>`.

**The GPU check.** On hardware named `gpu...`, smith first asks the
interpreter that runs the training whether its GPUs work: torch sees CUDA, as
many GPUs as `SMITH_GPUS`, and a small sum on each. A machine whose container
cannot reach its GPU would otherwise train on the CPU, slowly, while it is paid
for as a GPU. If the check fails smith takes no job, and says what is wrong as
`fault` in every heartbeat, for the backend to give the machine back.
`check_gpus = false` (or `SMITH_CHECK_GPUS=0`) turns it off; `true` turns it
on for hardware named otherwise.

**The runtime.** With `runtime = "<manifest>"` in its configuration, as the
GPU image has, smith installs that manifest's wheels before anything else,
the GPU check included (see above, and `runtime.py`). With the backend's
cache, the node first asks `POST /api/cache/global` with its token and
`{"op": "get", "path": "runtime/<wheel file name>"}` for each wheel, and
expects `{"url": "<presigned GET that takes Range requests>"}`; a URL that
answers 404, or a refusal to sign, sends the node to the next source. One that
cannot be installed is a `fault` like a GPU that does not work: no job is taken.


Apache-2.0 — [GPU Zero](https://gpuzero.dev)

