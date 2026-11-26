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

RUN pip install uv

# smith and every job it starts run as `smith`, not root (GPU-403): a script
# is code from a repository, and in a container that runs as root a mistake in
# one, or a dependency that misbehaves, can do whatever root can there. The
# user owns what a job writes - the runs (`runs` in smith.docker.toml), the
# environments made under them, and its home, where uv and torch keep their
# caches. A runs volume made by an image from before this one is root's:
# `docker compose run --rm --user root <service> chown -R smith:smith /runs`
# hands it over once.
RUN useradd --create-home --uid 10001 --user-group smith     && mkdir -p /runs     && chown -R smith:smith /runs
# smith's own files are root's, and only readable to `smith` (GPU-530): owned
# by the user the jobs run as, a job could rewrite smith.py, cache.py or
# environment.py, and the agent - or the next job - would run its code.
COPY . /app/smith
RUN chmod -R u=rwX,go=rX /app/smith
WORKDIR /app
USER smith

CMD ["python", "smith/smith.py", "--config", "smith/smith.docker.toml"]
