"""A node the backend no longer answers (GPU-365).

What these hold it to: the silence is measured from the backend's last
answer, and a 5xx - what a proxy says for a stopped backend - is not one,
while a 4xx is; the limit is off unless configured; once past it the agent
stops its job as a cancel does, waits for it, asks the provider to delete
the machine where the provider gave it a key for that, and exits with
ALONE_EXIT; and with no such key it only exits, leaving the host to act.
"""

import http.server
import io
import os
import threading
import time
import types
import urllib.error

import pytest

import smith
from made import RUNTIMES, TEMPLATE, made


def _agent(minutes=30.0):
    config = smith.Config(backend="http://127.0.0.1:1", name="n", hardware="h", runs="r", python="p",
                          scripts={}, alone_minutes=minutes)
    return smith.Agent(config)


def _error(code):
    return urllib.error.HTTPError("http://x", code, "status", {}, io.BytesIO(b""))


def test_a_5xx_is_silence_and_a_4xx_is_an_answer(monkeypatch):
    backend = smith.Backend("http://x")
    backend.heard_at = time.monotonic() - 100

    def answer(code):
        def urlopen(*_args, **_kwargs):
            raise _error(code)
        monkeypatch.setattr(smith.urllib.request, "urlopen", urlopen)

    answer(502)
    with pytest.raises(smith.Refusal):
        backend.call("GET", "/api/jobs")
    assert backend.silent_for() >= 100
    # A token refused is no answer either (GPU-530): the machine is alone.
    for code in (401, 403):
        answer(code)
        with pytest.raises(smith.Refusal):
            backend.call("GET", "/api/jobs")
        assert backend.silent_for() >= 100
    answer(404)
    with pytest.raises(smith.Refusal):
        backend.call("GET", "/api/jobs")
    assert backend.silent_for() < 5


def test_an_answer_that_is_not_json_is_a_backend_that_did_not_answer(monkeypatch):
    """A proxy's HTML with a 200, or an answer cut short, ended the agent
    and its job (GPU-453)."""
    import http.client

    backend = smith.Backend("http://x")

    class Page(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(smith.urllib.request, "urlopen", lambda *_a, **_k: Page(b"<html>Bad gateway</html>"))
    with pytest.raises(smith.Unreachable):
        backend.call("GET", "/api/jobs")

    def cut(*_args, **_kwargs):
        raise http.client.IncompleteRead(b"{")

    monkeypatch.setattr(smith.urllib.request, "urlopen", cut)
    with pytest.raises(smith.Unreachable):
        backend.call("GET", "/api/jobs")


@pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX's")
def test_the_kill_after_the_grace_reaches_the_ranks(monkeypatch):
    """torchrun's children are in its group, and die with it."""
    killed = []
    monkeypatch.setattr(smith.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    running = smith.Running.__new__(smith.Running)
    running.process = type("P", (), {"pid": 4242})()
    running.kill()
    assert killed == [(4242, smith.signal.SIGKILL)]


def test_the_limit_is_off_at_zero_and_counts_minutes():
    agent = _agent(0)
    agent.backend.heard_at = time.monotonic() - 10 ** 6
    assert agent.alone() is False
    agent = _agent(30)
    agent.backend.heard_at = time.monotonic() - 29 * 60
    assert agent.alone() is False
    agent.backend.heard_at = time.monotonic() - 31 * 60
    assert agent.alone() is True


def test_the_setting_comes_from_the_environment(tmp_path, monkeypatch):
    path = tmp_path / "smith.toml"
    path.write_text('code = true\npythons = ["3.12"]\ntorch = "2.11.0+cpu"\n')
    assert smith.Config.load(str(path)).alone_minutes == 0
    monkeypatch.setenv("SMITH_ALONE_MINUTES", "30")
    assert smith.Config.load(str(path)).alone_minutes == 30


def test_the_price_comes_from_the_environment(tmp_path, monkeypatch):
    path = tmp_path / "smith.toml"
    path.write_text('code = true\npythons = ["3.12"]\ntorch = "2.11.0+cpu"\n')
    assert smith.Config.load(str(path)).price_per_hour is None
    monkeypatch.setenv("SMITH_PRICE_PER_HOUR", "1.89")
    assert smith.Config.load(str(path)).price_per_hour == 1.89


def _running():
    exits = iter([None, None, 143])
    stopped = []
    running = types.SimpleNamespace(
        id=7,
        rendezvous=None,
        process=types.SimpleNamespace(poll=lambda: next(exits, 143)),
        close=lambda: stopped.append("closed"),
    )
    running.stop = lambda reason=smith.CANCELLED: stopped.append(reason)
    return running, stopped


def test_giving_up_stops_the_job_waits_for_it_and_exits(monkeypatch):
    for name in ("RUNPOD_POD_ID", "RUNPOD_API_KEY", "CONTAINER_ID", "CONTAINER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(smith.time, "sleep", lambda _s: None)
    agent = _agent()
    agent.current, stopped = _running()
    deleted = []
    monkeypatch.setattr(agent, "_delete_self", lambda *args: deleted.append(args))
    with pytest.raises(SystemExit) as exited:
        agent.give_up()
    assert exited.value.code == smith.ALONE_EXIT
    assert stopped[0] == smith.ALONE and stopped[-1] == "closed"
    assert deleted == []


@pytest.mark.parametrize("refusal", [smith.Unreachable("down"), smith.Refusal("POST /api/agents/heartbeat: 401", 401)])
def test_the_loop_gives_up_once_the_backend_has_been_silent(monkeypatch, refusal):
    """Silent, or refusing its token (GPU-530): a node whose token is
    refused is never heard again, and gives its machine back too."""
    agent = _agent()
    agent.backend.heard_at = time.monotonic() - 31 * 60
    for name in ("reconcile", "start_spot_watch", "send_logs", "ship", "take"):
        monkeypatch.setattr(agent, name, lambda: None)

    def beat():
        raise refusal

    monkeypatch.setattr(agent, "beat", beat)
    monkeypatch.setattr(agent, "end_machine", lambda: None)
    with pytest.raises(SystemExit) as exited:
        agent.run()
    assert exited.value.code == smith.ALONE_EXIT


class _Provider(http.server.BaseHTTPRequestHandler):
    calls: list = []
    status = 200

    def log_message(self, *_args):
        pass

    def do_DELETE(self):
        type(self).calls.append((self.path, self.headers.get("Authorization")))
        self.send_response(type(self).status)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def provider():
    _Provider.calls = []
    _Provider.status = 200
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield _Provider, "http://127.0.0.1:%d" % server.server_address[1]
    server.shutdown()
    server.server_close()


def test_a_refused_deletion_tries_the_next_address(provider):
    handler, base = provider
    handler.status = 403
    agent = _agent()
    agent._delete_self("pod", [base + "/v2/pods/p", base + "/v1/pods/p"], "k")
    assert handler.calls == [("/v2/pods/p", "Bearer k"), ("/v1/pods/p", "Bearer k")]


def test_a_deletion_that_is_accepted_stops_there(provider):
    handler, base = provider
    agent = _agent()
    agent._delete_self("pod", [base + "/v2/pods/p", base + "/v1/pods/p"], "k")
    assert handler.calls == [("/v2/pods/p", "Bearer k")]


def test_the_provider_is_told_by_its_own_variables(monkeypatch):
    agent = _agent()
    asked = []
    monkeypatch.setattr(agent, "_delete_self", lambda what, urls, key: asked.append((what, urls, key)))
    for name in ("RUNPOD_POD_ID", "RUNPOD_API_KEY", "CONTAINER_ID", "CONTAINER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    agent.end_machine()
    assert asked == []
    monkeypatch.setenv("CONTAINER_ID", "123")
    monkeypatch.setenv("CONTAINER_API_KEY", "vk")
    agent.end_machine()
    assert asked[-1] == ("vast.ai instance", ["https://console.vast.ai/api/v0/instances/123/"], "vk")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod1")
    monkeypatch.setenv("RUNPOD_API_KEY", "rk")
    agent.end_machine()
    assert asked[-1][0] == "RunPod pod" and asked[-1][2] == "rk"
    assert asked[-1][1][0].endswith("/pods/pod1")


def test_a_secret_is_never_a_name_this_machine_reads_as_code():
    """LD_PRELOAD or PYTHONPATH as a secret's name is a command on a machine
    that runs only its own scripts (GPU-463)."""
    kept = smith.job_secrets({"id": 1, "secrets": {
        "HF_TOKEN": "h", "WANDB_API_KEY": "w", "GITHUB_TOKEN": "g",
        "LD_PRELOAD": "/tmp/x.so", "PYTHONPATH": "/tmp", "pythonwarnings": "x", "GIT_CONFIG_PARAMETERS": "x",
        "PATH": "/tmp", "BROWSER": "sh -c x", "SMITH_TOKEN": "t", "MOONCLIP_S3_BUCKET": "b",
    }})
    assert kept == {"HF_TOKEN": "h", "WANDB_API_KEY": "w", "GITHUB_TOKEN": "g"}


def test_a_runs_id_is_a_folders_name_and_never_a_way_out(tmp_path):
    """A run's id is a folder under the agent's runs and a prefix in its
    bucket; ``../../app/smith`` wrote the store over the agent (GPU-465)."""
    assert smith._run_name("harold-001") == "harold-001"
    for name in ("../../app/smith", "a/b", "..", "", ".hidden", "a..b", "/etc"):
        with pytest.raises(RuntimeError, match="not a name a run can have"):
            smith._run_name(name)
    runs = tmp_path / "runs"
    (runs / "harold-001").mkdir(parents=True)
    assert smith._inside(str(runs), str(runs / "harold-001"))
    assert not smith._inside(str(runs), str(runs / ".." / "elsewhere"))
    assert not smith._inside(str(runs), str(tmp_path))


@pytest.mark.skipif(os.name == "nt", reason="a job is a process group, POSIX's")
def test_a_job_is_not_given_what_is_the_agents_alone(tmp_path, monkeypatch):
    """The agent's token under its own name, and its bucket's keys when the
    job brings a bucket of its own (GPU-465)."""
    import json
    import sys

    script = tmp_path / "show.py"
    script.write_text("import json, os; print(json.dumps(dict(os.environ)))")
    monkeypatch.setenv("SMITH_TOKEN", "agent-token")
    monkeypatch.setenv("MOONCLIP_S3_ACCESS_KEY", "agents-key")
    monkeypatch.setenv("MOONCLIP_S3_SECRET_KEY", "agents-secret")
    config = smith.Config(backend="http://127.0.0.1:1", name="n", hardware="h", runs=str(tmp_path / "runs"),
                          python=sys.executable, scripts={"show": str(script)}, token="agent-token",
                          price_per_hour=1.89, **RUNTIMES)
    made(tmp_path / "runs")
    job = {"id": 1, "script": "show", "mode": "fresh", "name": "n", "run_ref": "r-001", "params": {},
           "storage": {"type": "s3", "bucket": "theirs", "region": "eu-west-1"}, "template": TEMPLATE,
           "job_token": "gzjt_the-jobs"}
    running = smith.Running(job, config)
    running.process.wait(timeout=60)
    with open(running.log_path, encoding="utf-8") as handle:
        env = json.loads(handle.read().strip().splitlines()[-1])
    assert "SMITH_TOKEN" not in env
    assert "MOONCLIP_S3_ACCESS_KEY" not in env and "MOONCLIP_S3_SECRET_KEY" not in env
    assert env["RAVEX_STORAGE_BUCKET"] == "theirs"
    # The run reports its metrics with the job's own token, never the
    # agent's (GPU-530), and the output carries neither back.
    assert env["RAVEX_METRICS_TOKEN"] == "gzjt_the-jobs"
    assert "agent-token" not in json.dumps(env)
    assert running.hidden["SMITH_TOKEN"] == "agent-token" and running.hidden["JOB_TOKEN"] == "gzjt_the-jobs"
    # A backend that hands no token of the job's gets no job run with the agent's.
    del job["job_token"]
    with pytest.raises(RuntimeError, match="job_token"):
        smith.Running(job, config)
    # What the machine costs, for the run's cost per token (GPU-522).
    assert env["RAVEX_PRICE_PER_HOUR"] == "1.89"


def test_the_machine_s_secrets_are_not_the_job_s(monkeypatch):
    """The provider's key that deletes the pod, and this agent's bucket keys,
    went to the job and to `ravex ship`, which run the job's code (GPU-530)."""
    environ = {"PATH": "/bin", "RUNPOD_API_KEY": "rk", "CONTAINER_API_KEY": "vk", "SMITH_TOKEN": "t",
               "MOONCLIP_S3_ACCESS_KEY": "a", "AWS_SECRET_ACCESS_KEY": "s", "SMITH_RUNTIME_URL": "https://m",
               "CUDA_VISIBLE_DEVICES": "0", "MONKEY_BUSINESS": "kept"}
    assert smith.without_secrets(environ) == {"PATH": "/bin", "SMITH_RUNTIME_URL": "https://m",
                                              "CUDA_VISIBLE_DEVICES": "0", "MONKEY_BUSINESS": "kept"}
    monkeypatch.setenv("RUNPOD_API_KEY", "rk")
    agent = _agent()
    agent.config.token = None
    assert "RUNPOD_API_KEY" not in smith.Agent(agent.config)._ship_env


@pytest.mark.skipif(os.name == "nt", reason="a job is a process group, POSIX's")
def test_the_agent_s_bucket_keys_go_only_where_the_store_goes(tmp_path, monkeypatch):
    """To a job whose store is in this agent's bucket, and to no other
    (GPU-530)."""
    import json
    import sys

    script = tmp_path / "show.py"
    script.write_text("import json, os; print(json.dumps(dict(os.environ)))")
    monkeypatch.setenv("MOONCLIP_S3_ACCESS_KEY", "agents-key")
    monkeypatch.setenv("MOONCLIP_S3_SECRET_KEY", "agents-secret")
    monkeypatch.setenv("RUNPOD_API_KEY", "rk")
    made(tmp_path / "runs")

    def env_of(storage, job_id):
        config = smith.Config(backend="http://127.0.0.1:1", name="n", hardware="h", runs=str(tmp_path / "runs"),
                              python=sys.executable, scripts={"show": str(script)}, storage=storage, **RUNTIMES)
        job = {"id": job_id, "script": "show", "mode": "fresh", "name": "n", "run_ref": "r-00%d" % job_id,
               "params": {}, "template": TEMPLATE}
        running = smith.Running(job, config)
        running.process.wait(timeout=60)
        with open(running.log_path, encoding="utf-8") as handle:
            return json.loads(handle.read().strip().splitlines()[-1]), running

    bucket = smith.Bucket.from_table({"type": "s3", "bucket": "agents", "region": "eu-west-1"}, "smith.toml")
    env, running = env_of(bucket, 1)
    assert env["RAVEX_STORAGE_BUCKET"] == "agents"
    assert env["MOONCLIP_S3_ACCESS_KEY"] == "agents-key" and env["MOONCLIP_S3_SECRET_KEY"] == "agents-secret"
    assert running.hidden["STORAGE_SECRET_KEY"] == "agents-secret"
    assert "RUNPOD_API_KEY" not in env
    env, _running = env_of(None, 2)
    assert "MOONCLIP_S3_ACCESS_KEY" not in env and "RUNPOD_API_KEY" not in env


def _gone(pid):
    """Whether ``pid`` is dead: a zombie nobody reaped counts, since it holds
    nothing (without --init, pytest may be the container's PID 1)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:
        with open("/proc/%d/stat" % pid, encoding="utf-8") as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return True


@pytest.mark.skipif(os.name == "nt", reason="a job is a process group, POSIX's")
def test_what_a_job_left_in_its_group_dies_when_it_ends(tmp_path):
    """A child that outlived the job's process kept its CUDA context, and the
    next job ran out of memory (GPU-530)."""
    import sys

    child = tmp_path / "child.pid"
    script = tmp_path / "spawn.py"
    script.write_text(
        "import subprocess, sys\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        "open(%r, 'w').write(str(p.pid))\n" % str(child)
    )
    config = smith.Config(backend="http://127.0.0.1:1", name="n", hardware="h", runs=str(tmp_path / "runs"),
                          python=sys.executable, scripts={"spawn": str(script)}, **RUNTIMES)
    made(tmp_path / "runs")
    job = {"id": 1, "script": "spawn", "mode": "fresh", "name": "n", "run_ref": "r-001", "params": {},
           "template": TEMPLATE}
    agent = smith.Agent(config)
    agent.current = smith.Running(job, config)
    agent.current.process.wait(timeout=60)
    pid = int(child.read_text())
    assert not _gone(pid)
    agent.send_logs = lambda: None
    agent._report_until_heard = lambda *args: None
    agent.watch()
    deadline = time.monotonic() + 10
    while not _gone(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _gone(pid)


def test_output_the_backend_will_never_take_is_passed_by(tmp_path):
    """Sent again it was refused again, twice a second, and nothing after it
    ever arrived (GPU-466)."""
    path = tmp_path / "job.log"
    path.write_bytes(b"one\ntwo\n")
    tail = smith.LogTail(7, str(path))
    asked = []

    class Backend:
        def call(self, method, url, body=None):
            asked.append(body["lines"])
            raise smith.Refusal("POST %s: 422 lines it cannot read" % url, 422)

    assert tail.send(Backend()) is True
    assert asked == [["one", "two"]]
    with open(path, "ab") as handle:
        handle.write(b"three\n")

    class Taking:
        def call(self, method, url, body=None):
            asked.append(body["lines"])

    tail.send(Taking())
    assert asked[-1] == ["three"]

    # A backend that is failing for a moment is another matter: kept, and sent again.
    class Down:
        def call(self, method, url, body=None):
            raise smith.Refusal("POST %s: 502" % url, 502)

    with open(path, "ab") as handle:
        handle.write(b"four\n")
    with pytest.raises(smith.Refusal):
        tail.send(Down())
    tail.send(Taking())
    assert asked[-1] == ["four"]


def test_a_machine_is_busy_until_its_last_job_s_output_has_arrived(tmp_path):
    """The backend gives a machine back once its agent holds no job; the
    last lines of a stopped job went with it (GPU-504)."""
    agent = _agent()
    assert agent.holding() is None
    path = str(tmp_path / "out.log")
    with open(path, "wb") as handle:
        handle.write(b"SIGTERM received\n")
    tail = smith.LogTail(7, path)
    tail.done = True
    agent.tails.append(tail)
    assert agent.holding() == 7
    sent = []

    class Taking:
        def call(self, method, url, body=None):
            sent.append(body["lines"])

    agent.backend = Taking()
    agent.send_logs()
    assert sent == [["SIGTERM received"]]
    assert agent.holding() is None


def test_a_stop_goes_on_without_the_backend():
    """The spot notice and the kill after the grace waited on a heartbeat
    that was answered (GPU-466)."""
    agent = _agent()
    stops = []

    class Job:
        stop_requested_at = None
        stop_reason = smith.CANCELLED
        process = type("P", (), {"poll": lambda self: None})()

        def stop(self, reason=smith.CANCELLED):
            stops.append(reason)
            self.stop_requested_at = 1.0

    agent.current = Job()
    agent.keep_stopping()
    assert stops == []
    agent.reclaimed.set()
    agent.keep_stopping()
    agent.keep_stopping()
    assert stops == [smith.RECLAIMED, smith.CANCELLED]


def test_the_end_of_a_job_is_said_again_through_a_backend_that_is_failing(monkeypatch):
    """A 502 during a deploy on "finished" was raised, and the job stayed
    "running" on the backend for good (GPU-474)."""
    agent = _agent()
    said = []

    def report(job_id, state, message=None, run_id=None):
        said.append(state)
        if len(said) < 3:
            raise smith.Refusal("PATCH /api/jobs/1: 502", 502)
        return {}

    monkeypatch.setattr(agent, "report", report)
    monkeypatch.setattr(smith.time, "sleep", lambda _seconds: None)
    agent._report_until_heard(1, "finished", None, "r-001")
    assert said == ["finished", "finished", "finished"]
    # A refusal of what was said is still raised: asking again changes nothing.
    monkeypatch.setattr(agent, "report", lambda *a, **k: (_ for _ in ()).throw(smith.Refusal("404", 404)))
    with pytest.raises(smith.Refusal):
        agent._report_until_heard(1, "finished", None, "r-001")


def test_a_refused_end_still_ships_the_run_and_its_kernels():
    """A 404 or 401 on the final report skipped `ravex ship` and the cache's
    push, and the metrics Ravex kept through an outage were lost (GPU-530)."""
    agent = _agent()
    pushed = []
    agent.current = types.SimpleNamespace(
        id=7, run_id="r-001", store="store", rendezvous=None, outputs=[], stop_requested_at=None,
        process=types.SimpleNamespace(poll=lambda: 0), serve_when_ready=lambda: None, kill=lambda: None,
        close=lambda: None, fault=lambda: None, python=lambda: "python", job_token="gzjt_seven",
        push_caches=lambda _config: pushed.append(True),
    )
    agent.send_logs = lambda: None

    def refused(*_args):
        raise smith.Refusal("PATCH /api/jobs/7: 404", 404)

    agent._report_until_heard = refused
    agent.watch()
    assert agent.current is None
    assert "store" in agent.to_ship and agent.ship_python["store"] == "python"
    assert pushed == [True]
    # `ravex ship` runs the job's Ravex, so with the job's token, not the agent's (GPU-530).
    started = []
    agent.config.token = "agent-token"
    smith.subprocess.Popen, real = (lambda command, **kwargs: started.append(kwargs["env"]) or
                                    types.SimpleNamespace(poll=lambda: None)), smith.subprocess.Popen
    try:
        agent.to_ship["store"] = 0.0
        agent.ship()
    finally:
        smith.subprocess.Popen = real
    assert started[0]["RAVEX_METRICS_TOKEN"] == "gzjt_seven" and "agent-token" not in started[0].values()


def test_a_job_that_cannot_start_is_said_failed_through_a_backend_that_fails_once(monkeypatch):
    """Said once, a 502 or no answer left the job "claimed" for good
    (GPU-530)."""
    agent = _agent()
    said = []
    answers = [{"id": 9, "script": "nowhere", "mode": "fresh", "name": "n"}, smith.Unreachable("down"), {}]

    def call(method, path, body=None):
        said.append((method, path, body))
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(agent.backend, "call", call)
    monkeypatch.setattr(smith.time, "sleep", lambda _seconds: None)
    agent.take()
    assert agent.current is None
    patches = [body for method, path, body in said if method == "PATCH"]
    assert len(patches) == 2 and all(body["state"] == "failed" for body in patches)
    assert "could not start it" in patches[-1]["message"] and "run_id" not in patches[-1]


def test_no_job_is_taken_until_the_old_ones_are_closed(monkeypatch):
    """Asked once, at a start the backend missed, the jobs an earlier start
    left were never closed (GPU-474)."""
    agent = _agent()
    answers = [smith.Unreachable("down"), []]

    def call(method, path, body=None):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(agent.backend, "call", call)
    agent.reconcile()
    assert not agent.reconciled and not agent.takes_jobs()
    agent.reconcile()
    assert agent.reconciled and agent.takes_jobs()


def test_a_backend_that_is_away_is_asked_less_often_and_not_in_step():
    waits = [smith.backoff(smith.POLL_SECONDS) for _ in range(50)]
    assert all(smith.POLL_SECONDS <= wait <= 3 * smith.POLL_SECONDS * 2 / 2 for wait in waits)
    assert len(set(waits)) > 1
    assert smith.backoff(10_000.0) <= smith.POLL_MOST_SECONDS * 1.5
