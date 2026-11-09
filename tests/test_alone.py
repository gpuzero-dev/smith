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
import threading
import time
import types
import urllib.error

import pytest

import smith


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
    answer(404)
    with pytest.raises(smith.Refusal):
        backend.call("GET", "/api/jobs")
    assert backend.silent_for() < 5


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
    path.write_text('code = true\n')
    assert smith.Config.load(str(path)).alone_minutes == 0
    monkeypatch.setenv("SMITH_ALONE_MINUTES", "30")
    assert smith.Config.load(str(path)).alone_minutes == 30


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


def test_the_loop_gives_up_once_the_backend_has_been_silent(monkeypatch):
    agent = _agent()
    agent.backend.heard_at = time.monotonic() - 31 * 60
    for name in ("reconcile", "start_spot_watch", "send_logs", "ship", "take"):
        monkeypatch.setattr(agent, name, lambda: None)

    def beat():
        raise smith.Unreachable("down")

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
