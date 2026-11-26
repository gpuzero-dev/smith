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
provider's GPU node (`Dockerfile.gpu`, config `smith.gpu.toml`):

```bash
docker build -t smith .
docker build -f Dockerfile.gpu -t smith-gpu .
```

**A job brings what it trains with** (GPU-523). Neither image carries torch,
Ravex or Moonclip. A job carries a template, the exact versions
`{"python": "3.12", "torch": "2.11.0+cu128", "ravex": "0.19.0", "moonclip": "0.4.1"}`,
and before its script `environment.py` makes that environment on the machine,
once: a Python of that version installed by uv into a folder of its own
(`<runs>/_envs/<template>/`), torch into it (from the GPU image's lock for that
Python, `runtime/cp312.json` and the others, or torch's CPU build from its
index), then Ravex, Moonclip, psutil and, on a GPU, nvidia-ml-py. The next job
with the same template finds it made. A repository's requirements are a
virtual environment on top of it, with `--system-site-packages`. A machine
claims only what it can make: it sends the backend `pythons` (the ones it has
a torch for: the GPU image's locks, or `pythons` in the file) and `torch`, and
a job without a template, or with one it cannot make, fails saying so. A
script of `[scripts]` written `ravex:<path>` is a file of the job's Ravex
release, its examples and benches, fetched once per release from its tag.

The GPU image is published by pushing to the `deploy` branch
(`git push origin main:deploy`): `.github/workflows/CD.yml`, its `gpu` job,
once `CI.yml` has passed on that commit, builds it and pushes it as `ghcr.io/<owner>/smith-gpu:<commit>`, smith's short
commit, and as `:latest`, then checks both tags are in the registry and that
no layer is over 250 MB. Last, it tells the backend which bytes it pushed:
`POST /api/agent-image` with `Authorization: Bearer <SMITH_IMAGE_TOKEN>`
and `{"digest": "sha256:...", "commit": "<smith's commit>"}`. A backend
should start nodes from that digest, not from a tag: anyone who can push a
branch of this repository can run a workflow that rewrites a tag in the
registry, while that token is a secret of the `production` environment,
which only `main` and `deploy` enter. It is a token of its own, which no
other job is given: what fills the cache cannot choose the bytes nodes
run.

**The GPU image does not carry torch.** torch with its CUDA libraries is
about 7 GB, and a host's docker that drops a layer pulls it again from its
first byte: on slow hosts the image never arrived. So the image carries only
the locks, one per Python (`runtime/cp311.json`, `cp312.json`, `cp313.json`:
each wheel's name, address and sha256, resolved from `runtime.in` when it is
built), and `runtime.py` fetches the wheels when a job's environment is made: in pieces, sixteen at once, each
retried on its own, each file checked against its hash, then installed with
uv. Each file comes from the first source that has it: the backend's cache
when the node uses it (`SMITH_CACHE=backend`), at `runtime/<wheel file name>`
in its read-only `global` scope, through a URL the backend signs for each file
(the protocol of `cache.py`, below) and signs again when it expires mid-download;
else a mirror if `SMITH_RUNTIME_URL` is set, where each wheel is at its own
file name; else the index the wheel was resolved from.

**Who fills the cache's `runtime/`.** This repository's
`.github/workflows/CD.yml`, its `runtime` job: as soon as its `gpu` job has
published an image, then every hour (or by hand), for the image nodes start
from, which the backend names by digest in its answer to
`POST /api/cache-builds/pending` (`{"queued": <n>, "image": "sha256:..."}`),
never for a tag: it takes the locks out of that image and runs `runtime.py fill` on each,
which fetches each wheel the cache lacks from its index, checks its sha256
and uploads it, one at a time. Like the wheel builder, the same workflow's
`wheels` job, it holds no bucket key, only the secrets `SMITH_BACKEND` and
`SMITH_BUILDER_TOKEN`; without them it fills nothing. The `wheels` job
compiles in that same image, which is given neither: the runner takes the
build and pushes the wheel the container left, once it has checked that it
is the package and the version that were asked for. It asks the backend,
with `Authorization: Bearer <SMITH_BUILDER_TOKEN>`, at
`POST /api/cache-runtime/sign`:
`{"op": "list", "path": "runtime/"}` → `{"paths": ["runtime/<wheel>", ...]}`
once, then `{"op": "put", "path": "runtime/<wheel>", "size": <bytes>}` →
`{"url": "<presigned PUT for that Content-Length>"}` for each wheel it
uploads. A 503 means the backend keeps no cache, and there is nothing to fill.

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
never its own, and never one named `RAVEX_*`, `SMITH_*` or `MOONCLIP_*`, nor one
the loader, Python or git read as code to load or run (`LD_*`, `DYLD_*`,
`PYTHON*`, `GIT_*`, `PATH`, `HOME`, `BROWSER`, `BASH_ENV`, `ENV`, `IFS`): a
secret under such a name is dropped, and the log says so. It replaces
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
address), `commit` (a full SHA) and, for a private repository, `token`. The
job's `script` is then the path of a Python file of that commit (`train.py`,
`scripts/train.py`), checked to be one before anything runs. smith fetches that commit
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
`SMITH_HARDWARE`, `SMITH_GPUS`, `SMITH_STORAGE_*`, `SMITH_PRICE_PER_HOUR`
(what the machine costs an hour, handed to the job as `RAVEX_PRICE_PER_HOUR`
for its cost per token), and the
bucket's `MOONCLIP_S3_ACCESS_KEY` / `MOONCLIP_S3_SECRET_KEY`, which Moonclip
reads itself. A job's storage on AWS may come with a region and no keys: the
node's instance role writes the bucket (Moonclip 0.3.2 or later), and the run is
given no `MOONCLIP_S3_*`.
On EC2, smith also watches the metadata service for a spot interruption notice
(`AWS_EC2_METADATA_SERVICE_ENDPOINT`, else `169.254.169.254`): it is asked once at
start and, where nothing answers, never again; on a notice the running job is
stopped as a cancel does, so Ravex writes its last checkpoint, and no further job
is taken. The job is reported with state `interrupted` (`failed` to a backend that
refuses it), for a backend that carries runs on from a machine taken back.
`SMITH_ALONE_MINUTES` (30 in the GPU image, off unless set elsewhere): after
that long without an answer from the backend - nothing, or only 5xx - the
running job is stopped as a cancel does, so Ravex writes its last checkpoint to
the bucket, and the machine is given back: on RunPod and vast.ai smith deletes
its own pod or instance with the key the provider puts in it (`RUNPOD_API_KEY`,
`CONTAINER_API_KEY`), elsewhere it exits with status 75 and leaves the rest to
what started it. With the backend down nobody else ends a node, and its GPUs
stay billed.
`SMITH_IMAGE`, which the GPU image sets to its own tag, is written as the first
line of every job's output: `[smith] image ghcr.io/.../smith-gpu:<commit>`.

**The GPU check.** On hardware named `gpu...`, the first job's environment,
once made, is asked whether its GPUs work: torch sees CUDA, as many GPUs as
`SMITH_GPUS`, and a small sum on each (the image has no torch of its own to
ask before, GPU-523). A machine whose container cannot reach its GPU would
otherwise train on the CPU, slowly, while it is paid for as a GPU. If the
check fails that job fails saying so, smith takes no other, and says what is
wrong as `fault` in every heartbeat, for the backend to give the machine back.
`check_gpus = false` (or `SMITH_CHECK_GPUS=0`) turns it off; `true` turns it
on for hardware named otherwise.

**The runtime.** With `locks = "<folder>"` in its configuration, as the GPU
image has, a job's environment gets torch from the lock of its Python there,
before the GPU check (see above, and `runtime.py`). With the backend's
cache, the node first asks `POST /api/cache/global` with its token and
`{"op": "get", "path": "runtime/<wheel file name>"}` for each wheel, and
expects `{"url": "<presigned GET that takes Range requests>"}`; a URL that
answers 404, or a refusal to sign, sends the node to the next source. One that
cannot be installed is a `fault` like a GPU that does not work: no job is taken.


Apache-2.0 — [GPU Zero](https://gpuzero.dev)

