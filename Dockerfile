# syntax=docker/dockerfile:1
#
# smith in a container on CPU: a training environment that takes jobs from
# a backend's queue.
#
#   docker build -t smith .
#
# No torch, Ravex or Moonclip in it (GPU-523): a job carries its template -
# its Python, a torch version, the Ravex and Moonclip it trains with - and
# smith makes that environment with uv when the job starts (environment.py),
# torch's CPU build included, and keeps it for the next job with the same one.
FROM python:3.12-bookworm

ENV PYTHONUNBUFFERED=1     PIP_DISABLE_PIP_VERSION_CHECK=1     PIP_NO_CACHE_DIR=1

# Pinned (GPU-530): uv makes every job's environment, and an unpinned one was
# whatever PyPI had on the day the image was built. Moved by hand, to a
# release at least two weeks old; Dependabot reads no pin in a RUN line.
RUN pip install uv==0.13.8

# Every job runs as `job`, never root (GPU-403) and never smith's own user
# (GPU-530): a script is code from a repository, and a dependency of it that
# misbehaves could otherwise do what root can here, or - as smith's user -
# read smith's token under /proc and rewrite what the next job runs. smith
# itself runs as root, which is what lets it start a process as another user
# (SMITH_JOB_USER); it owns /runs and everything every job reads there, and
# hands each job its store and its own folder. Same uid as the `smith` user
# of the images before, so a runs volume they wrote is made smith's again
# when it starts (Agent.seal_runs), not left writable by the jobs.
RUN useradd --no-create-home --uid 10001 --user-group job && mkdir -p /runs
ENV SMITH_JOB_USER=job
# smith's own files are root's, and only readable to `job` (GPU-530): owned
# by the user the jobs run as, a job could rewrite smith.py, cache.py or
# environment.py, and the agent - or the next job - would run its code.
COPY . /app/smith
RUN chmod -R u=rwX,go=rX /app/smith
WORKDIR /app

CMD ["python", "smith/smith.py", "--config", "smith/smith.docker.toml"]
