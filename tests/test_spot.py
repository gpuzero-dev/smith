"""A job's storage without keys, and the AWS spot interruption notice.

What these hold it to: a bucket handed without keys but with a region is
accepted and gives the run no key variables, so that Moonclip takes the
instance's role; without a region it is still refused. The spot watcher asks
the metadata service once at start with a short timeout and stays off when
nobody answers, and when a notice appears it stops the running job the way a
cancel does, so that the run writes its last checkpoint.
"""

import http.server
import threading
import time
import types

import pytest

import smith


class _Metadata(http.server.BaseHTTPRequestHandler):
    """An IMDSv2 that wants a session token and answers 404 until ``notice``
    is set, as the real one does while no interruption is planned."""

    notice = None
    requests: list = []

    def log_message(self, *_args):
        pass

    def do_PUT(self):
        type(self).requests.append(("PUT", self.path))
        if self.path != "/latest/api/token":
            self.send_error(404)
            return
        body = b"session-token"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        type(self).requests.append(("GET", self.path))
        if self.headers.get("X-aws-ec2-metadata-token") != "session-token":
            self.send_error(401)
            return
        if self.path != "/latest/meta-data/spot/instance-action" or type(self).notice is None:
            self.send_error(404)
            return
        body = type(self).notice.encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def imds(monkeypatch):
    _Metadata.notice = None
    _Metadata.requests = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Metadata)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("AWS_EC2_METADATA_SERVICE_ENDPOINT", "http://127.0.0.1:%d" % server.server_address[1])
    yield _Metadata
    server.shutdown()
    server.server_close()


def _wait(condition, seconds=5.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_keyless_storage_with_a_region_is_a_bucket_without_keys():
    bucket = smith.Bucket.from_job({"type": "s3", "bucket": "runs-bucket", "prefix": "runs", "region": "eu-west-1"})
    assert bucket.access_key is None and bucket.secret_key is None
    environment = bucket.environment(bucket.prefix_of("abc"))
    assert environment["RAVEX_STORAGE_BUCKET"] == "runs-bucket"
    assert environment["RAVEX_STORAGE_PREFIX"] == "runs/abc"
    assert environment["RAVEX_STORAGE_REGION"] == "eu-west-1"
    assert "RAVEX_STORAGE_ENDPOINT" not in environment
    assert not [name for name in environment if name.startswith("MOONCLIP_S3_")]


def test_keyless_storage_without_a_region_is_refused():
    with pytest.raises(RuntimeError, match="without its keys"):
        smith.Bucket.from_job({"type": "s3", "bucket": "runs-bucket", "prefix": "runs"})


def test_half_a_pair_of_keys_is_still_refused():
    with pytest.raises(RuntimeError, match="without its keys"):
        smith.Bucket.from_job({"type": "s3", "bucket": "b", "region": "eu-west-1", "access_key": "only-one"})


def test_a_notice_calls_back_once(imds):
    heard = []
    watcher = smith.SpotWatcher(lambda: heard.append(1), poll_seconds=0.05)
    assert watcher.start() is True
    time.sleep(0.2)
    assert heard == []
    imds.notice = '{"action": "terminate", "time": "2026-11-02T10:00:00Z"}'
    assert _wait(lambda: heard)
    time.sleep(0.2)
    assert heard == [1]


def test_no_metadata_service_keeps_the_watcher_off_without_delay(monkeypatch):
    monkeypatch.setenv("AWS_EC2_METADATA_SERVICE_ENDPOINT", "http://127.0.0.1:9")
    watcher = smith.SpotWatcher(lambda: None)
    began = time.monotonic()
    assert watcher.start() is False
    assert time.monotonic() - began < 2.0
    assert watcher.thread is None


def test_the_probe_is_one_request_pair(imds):
    watcher = smith.SpotWatcher(lambda: None, poll_seconds=60)
    assert watcher.start() is True
    assert _wait(lambda: len(imds.requests) >= 2)
    assert imds.requests[:2] == [("PUT", "/latest/api/token"), ("GET", "/latest/meta-data/spot/instance-action")]


def _agent():
    config = smith.Config(backend="http://127.0.0.1:1", name="n", hardware="h", runs="r", python="p", scripts={})
    return smith.Agent(config)


def _running():
    stopped = []
    running = types.SimpleNamespace(
        id=7,
        run_id="abc",
        process=types.SimpleNamespace(poll=lambda: None),
        stop_requested_at=None,
        last_report=time.monotonic(),
        stop_reason=None,
        push_while_running=lambda _config: None,
    )

    def stop(reason="cancelled by the backend"):
        running.stop_requested_at = time.monotonic()
        running.stop_reason = reason
        stopped.append(reason)

    running.stop = stop
    return running, stopped


def test_a_notice_stops_the_running_job_through_stop():
    agent = _agent()
    agent.current, stopped = _running()
    agent.watch()
    assert stopped == []
    agent.reclaimed.set()
    agent.watch()
    assert len(stopped) == 1 and "reclaiming" in stopped[0]


def test_a_node_that_is_being_reclaimed_takes_no_job():
    agent = _agent()
    agent.reclaimed.set()
    assert agent.takes_jobs() is False


def test_a_notice_already_posted_at_the_probe_calls_back_and_needs_no_thread(imds):
    imds.notice = '{"action": "terminate", "time": "2026-11-02T10:00:00Z"}'
    heard = []
    watcher = smith.SpotWatcher(lambda: heard.append(1), poll_seconds=60)
    assert watcher.start() is True
    assert heard == [1] and watcher.thread is None


def test_run_starts_the_watcher_after_the_start_up_work(monkeypatch):
    agent = _agent()
    order = []
    monkeypatch.setattr(agent, "reconcile", lambda: order.append("reconcile"))
    monkeypatch.setattr(agent, "start_spot_watch", lambda: order.append("watch"))

    class _Stop(Exception):
        pass

    def stop(_seconds):
        raise _Stop

    monkeypatch.setattr(smith.time, "sleep", stop)
    with pytest.raises(_Stop):
        agent.run()
    assert order == ["reconcile", "watch"]


def _finished_by(stop_reason):
    agent = _agent()
    running, _stopped = _running()
    running.process = types.SimpleNamespace(poll=lambda: 0)
    running.stop_requested_at = time.monotonic() if stop_reason else None
    running.stop_reason = stop_reason or smith.CANCELLED
    running.close = lambda: None
    running.rendezvous = None
    running.outputs = []
    running.store = "store"
    running.push_caches = lambda _config: None
    agent.current = running
    agent.send_logs = lambda: None
    heard = []
    agent._report_until_heard = lambda *args: heard.append(args[1:3])
    agent.watch()
    return heard


def test_a_job_stopped_by_the_notice_is_reported_interrupted():
    assert _finished_by(smith.RECLAIMED) == [(smith.INTERRUPTED, "the provider reclaimed the machine")]


def test_a_cancelled_job_is_still_reported_failed_and_a_clean_exit_finished():
    assert _finished_by(smith.CANCELLED) == [("failed", "stopped on request")]
    assert _finished_by(None) == [("finished", None)]


def test_a_backend_that_refuses_interrupted_is_told_failed_instead(monkeypatch):
    agent = _agent()
    sent = []

    def report(job_id, state, message=None, run_id=None):
        sent.append(state)
        if state == smith.INTERRUPTED:
            raise smith.Refusal("PATCH /api/jobs/1: 400 state 'interrupted' is not one an agent reports", 400)
        return {}

    monkeypatch.setattr(agent, "report", report)
    agent._report_until_heard(1, smith.INTERRUPTED, "m", "r")
    assert sent == [smith.INTERRUPTED, "failed"]


def test_a_backend_failing_for_a_moment_is_told_interrupted_again(monkeypatch):
    agent = _agent()
    sent = []

    def report(job_id, state, message=None, run_id=None):
        sent.append(state)
        if len(sent) == 1:
            raise smith.Refusal("PATCH /api/jobs/1: 503 deploying", 503)
        return {}

    monkeypatch.setattr(agent, "report", report)
    monkeypatch.setattr(smith.time, "sleep", lambda _s: None)
    agent._report_until_heard(1, smith.INTERRUPTED, "m", "r")
    assert sent == [smith.INTERRUPTED, smith.INTERRUPTED]


def test_a_validation_refusal_422_also_falls_back_to_failed(monkeypatch):
    agent = _agent()
    sent = []

    def report(job_id, state, message=None, run_id=None):
        sent.append(state)
        if state == smith.INTERRUPTED:
            raise smith.Refusal("PATCH /api/jobs/1: 422 unprocessable", 422)
        return {}

    monkeypatch.setattr(agent, "report", report)
    agent._report_until_heard(1, smith.INTERRUPTED, "m", "r")
    assert sent == [smith.INTERRUPTED, "failed"]
