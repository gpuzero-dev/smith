"""The job as a user of its own (GPU-530).

What these hold it to, in a container where the agent is root and a user
``job`` exists (smith's Dockerfile): the training runs as that user; it
cannot read the agent's environment under /proc nor signal it; it cannot
write smith's files, the template environments, a repository's checkout or
the virtual environment made for it, nor the machine's shared caches; it can
write its store, its own folder and its output; its group dies when it ends
or is stopped, and so does what left the group, as torchrun's ranks do. An
agent that cannot start a job as another user refuses to, rather than run it
as itself. Elsewhere - Windows, a user that is not root - only that refusal
is checked.
"""

import json
import os
import stat
import sys
import time
import types

import pytest

import smith
from made import RUNTIMES, TEMPLATE, made

SHA = "0123456789abcdef0123456789abcdef01234567"


def _job_user():
    if os.name == "nt" or not os.path.isdir("/proc") or os.geteuid() != 0:
        return None
    import pwd

    try:
        return pwd.getpwnam("job")
    except KeyError:
        return None


needs_job_user = pytest.mark.skipif(
    _job_user() is None, reason="needs Linux, root, and a user `job`: smith's Dockerfile has them"
)

#: What the job checks of itself, as the job's user, written to its store.
PROBE = r'''
import json, os, subprocess, sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
agent, runs, envs = int(args["--agent"]), args["--runs"], args["--envs"]


def can_write(folder):
    try:
        path = os.path.join(folder, ".probe-%d" % os.getpid())
        open(path, "w").close()
        os.remove(path)
        return True
    except OSError:
        return False


def can_read(path):
    try:
        with open(path, "rb") as handle:
            handle.read()
        return True
    except OSError:
        return False


def can_signal(pid):
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return False


group = subprocess.Popen(["sleep", "300"])
escaped = subprocess.Popen(["sleep", "300"], start_new_session=True)
places = {
    "smith": args["--smith"], "runs": runs, "envs": os.path.join(runs, "_envs"), "template": envs,
    "checkout": args["--checkout"], "venv": sys.prefix, "shared cache": os.path.join(runs, "_cache"),
    "logs": os.path.join(runs, "_logs"), "store": os.environ["RAVEX_STORAGE_PATH"], "cwd": os.getcwd(),
    "home": os.environ["HOME"], "triton": os.environ["TRITON_CACHE_DIR"],
}
out = {"uid": os.getuid(), "gid": os.getgid(), "groups": os.getgroups(),
       "environ": can_read("/proc/%d/environ" % agent), "signal": can_signal(agent),
       "write": {name: can_write(path) for name, path in places.items()},
       "prefix": sys.prefix, "group": group.pid, "escaped": escaped.pid}
with open(os.path.join(os.environ["RAVEX_STORAGE_PATH"], "probe.json"), "w") as handle:
    json.dump(out, handle)
print("probe done as uid %d" % os.getuid(), flush=True)
'''


def _gone(pid):
    """Whether ``pid`` is dead; a zombie nobody reaped holds nothing."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:
        with open("/proc/%d/stat" % pid, encoding="utf-8") as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return True


def _until(check, seconds=10.0):
    deadline = time.monotonic() + seconds
    while not check() and time.monotonic() < deadline:
        time.sleep(0.1)
    return check()


@pytest.fixture
def runs(tmp_path):
    """Runs the job's user can reach: pytest makes its temporary folders
    root's alone."""
    path = str(tmp_path)
    while path not in ("/", "/tmp"):
        os.chmod(path, 0o755)
        path = os.path.dirname(path)
    return tmp_path / "runs"


def test_an_agent_that_cannot_start_a_job_as_another_user_refuses_to():
    """Never the agent's own user in silence when another was asked for."""
    config = smith.Config(backend="http://127.0.0.1:1", name="n", hardware="h", runs="r", python="p",
                          job_user="no-such-user-here", **RUNTIMES)
    if _job_user() is None:
        with pytest.raises(RuntimeError, match="only an agent running as root|no such user"):
            smith.job_identity(config)
    else:
        with pytest.raises(RuntimeError, match="no such user"):
            smith.job_identity(config)
    # Unset, a person's own machine: the job runs as the agent's user.
    config.job_user = ""
    assert smith.job_identity(config) is None and smith.as_user(None) == {}


def _code_job(config, runs, job_id):
    checkout = runs / "_code" / "example.com-r" / SHA
    job = {"id": job_id, "script": "probe.py", "mode": "fresh", "name": "n", "run_ref": "r-%03d" % job_id,
           "template": TEMPLATE, "code": {"repo": "https://example.com/r", "commit": SHA},
           "params": {"agent": os.getpid(), "runs": str(runs), "smith": os.path.dirname(smith.__file__),
                      "checkout": str(checkout), "envs": str(runs / "_envs" / made(runs))}}
    running = smith.Running(job, config)
    running.process.wait(timeout=300)
    with open(running.log_path, encoding="utf-8") as handle:
        log = handle.read()
    assert running.process.returncode == 0, log
    with open(os.path.join(running.store, "probe.json"), encoding="utf-8") as handle:
        return running, json.load(handle), log


@needs_job_user
def test_a_job_runs_as_its_own_user_and_reaches_only_what_is_its_own(runs):
    job = _job_user()
    config = smith.Config(backend="http://127.0.0.1:1", name="n", hardware="h", runs=str(runs),
                          python=sys.executable, code=True, job_user="job", **RUNTIMES)
    made(runs)
    # A commit already fetched, with requirements, so its job makes a virtual
    # environment for it - with nothing to install, which needs no network.
    checkout = runs / "_code" / "example.com-r" / SHA
    checkout.mkdir(parents=True)
    (checkout / ".smith-fetched").write_text("")
    (checkout / "requirements.txt").write_text("# nothing\n")
    (checkout / "probe.py").write_text(PROBE)
    (runs / "_cache").mkdir()
    agent = smith.Agent(config)
    agent.send_logs = lambda: None
    agent._report_until_heard = lambda *args: None

    running, seen, log = _code_job(config, runs, 1)
    assert "probe done as uid %d" % job.pw_uid in log
    # None of the agent's groups either.
    assert (seen["uid"], seen["gid"]) == (job.pw_uid, job.pw_gid) and seen["groups"] in ([], [job.pw_gid])
    # Nothing of the agent's: not its token and keys under /proc, not a signal.
    assert seen["environ"] is False and seen["signal"] is False
    written = seen["write"]
    for place in ("smith", "runs", "envs", "template", "checkout", "venv", "shared cache", "logs"):
        assert written[place] is False, place
    for place in ("store", "cwd", "home", "triton"):
        assert written[place] is True, place
    # Its own virtual environment, kept for the next job with the commit as the agent's.
    assert seen["prefix"].startswith(str(checkout) + os.sep + ".smith-venv-")
    for folder, _dirs, files in os.walk(seen["prefix"]):
        for path in [folder] + [os.path.join(folder, name) for name in files]:
            info = os.lstat(path)
            # A link's own mode means nothing; whose it is, does.
            assert info.st_uid == os.getuid(), path
            assert stat.S_ISLNK(info.st_mode) or not info.st_mode & 0o7022, path

    # Its group goes with it, and so does what left the group: torchrun's
    # ranks are each in a session of their own.
    agent.current = running
    agent.watch()
    assert _until(lambda: _gone(seen["group"]) and _gone(seen["escaped"]))
    again, second, _log = _code_job(config, runs, 2)
    # The same environment again, still not the job's to write.
    assert second["prefix"] == seen["prefix"] and second["write"]["venv"] is False
    assert not os.path.exists(running.job_dir)
    again.kill()
    again.close()
    running.close()


@needs_job_user
def test_a_stopped_job_of_its_own_user_goes_with_its_group(runs):
    """A signal to a group of another user's needs root, which the agent is."""
    child = runs.parent / "child.pid"
    # A folder of its own: the job runs in a copy of it, which cannot hold the runs.
    (runs.parent / "scripts").mkdir()
    script = runs.parent / "scripts" / "spawn.py"
    script.write_text(
        "import os, subprocess, sys, time\n"
        "p = subprocess.Popen(['sleep', '300'])\n"
        "open(%r, 'w').write('%%d %%d' %% (p.pid, os.getuid()))\n"
        "time.sleep(300)\n" % str(child)
    )
    os.chmod(runs.parent, 0o777)
    config = smith.Config(backend="http://127.0.0.1:1", name="n", hardware="h", runs=str(runs),
                          python=sys.executable, scripts={"spawn": str(script)}, job_user="job", **RUNTIMES)
    made(runs)
    running = smith.Running({"id": 3, "script": "spawn", "mode": "fresh", "name": "n", "run_ref": "r-003",
                             "params": {}, "template": TEMPLATE}, config)
    assert _until(lambda: child.exists() and child.read_text().count(" ") == 1, 60)
    pid, uid = (int(part) for part in child.read_text().split())
    assert uid == _job_user().pw_uid
    running.stop()
    assert _until(lambda: running.process.poll() is not None)
    assert _until(lambda: _gone(pid))
    running.close()


@needs_job_user
def test_runs_a_job_user_wrote_are_made_the_agents(runs):
    """A runs volume from the images whose jobs ran as smith's own user."""
    job = _job_user()
    envs = runs / "_envs" / "some"
    envs.mkdir(parents=True)
    planted = envs / "python"
    planted.write_text("")
    for path in (runs, runs / "_envs", envs, planted):
        os.chown(path, job.pw_uid, job.pw_gid)
    os.chmod(planted, 0o6777)
    agent = types.SimpleNamespace(config=types.SimpleNamespace(runs=str(runs)), identity=(job.pw_uid, job.pw_gid))
    smith.Agent.seal_runs(agent)
    for path in (runs, runs / "_envs", envs, planted):
        info = os.lstat(path)
        assert info.st_uid == os.getuid() and not info.st_mode & 0o7022, path


@needs_job_user
def test_the_end_of_a_job_spares_its_rendezvous_alone():
    """The other nodes of the run still read their peers from it (GPU-186);
    every other process of the job's user goes."""
    import subprocess

    job = _job_user()
    user = smith.as_user((job.pw_uid, job.pw_gid))
    rendezvous = subprocess.Popen(["sleep", "300"], start_new_session=True, **user)
    other = subprocess.Popen(["sleep", "300"], start_new_session=True, **user)
    ended = subprocess.Popen(["true"], start_new_session=True, **user)
    ended.wait()
    running = types.SimpleNamespace(process=ended, rendezvous=rendezvous, identity=(job.pw_uid, job.pw_gid))
    smith.Running.kill(running)
    assert _until(lambda: other.poll() is not None)
    assert rendezvous.poll() is None
    rendezvous.kill()
    rendezvous.wait()


#: A build backend in the repository itself, with nothing to fetch: an
#: editable install is a `.pth` naming the tree's `src`, as setuptools' is.
BACKEND = r'''
import base64, hashlib, os, zipfile
NAME, VERSION = "mypkg", "0.1"
INFO = "%s-%s.dist-info" % (NAME, VERSION)


def get_requires_for_build_editable(config_settings=None):
    return []


get_requires_for_build_wheel = get_requires_for_build_editable


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    files = {NAME + ".pth": os.path.abspath("src") + "\n",
             INFO + "/METADATA": "Metadata-Version: 2.1\nName: %s\nVersion: %s\n" % (NAME, VERSION),
             INFO + "/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
    wheel = "%s-%s-py3-none-any.whl" % (NAME, VERSION)
    rows = []
    with zipfile.ZipFile(os.path.join(wheel_directory, wheel), "w") as archive:
        for path, text in files.items():
            data = text.encode()
            archive.writestr(path, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            rows.append("%s,sha256=%s,%d" % (path, digest, len(data)))
        archive.writestr(INFO + "/RECORD", "\n".join(rows + [INFO + "/RECORD,,"]) + "\n")
    return wheel


build_wheel = build_editable
'''

#: The repository's script: its package, from the editable install, and
#: whether the job could change it.
IMPORTS = r'''
import json, os
import mypkg
try:
    open(mypkg.__file__, "a").close()
    writable = True
except OSError:
    writable = False
with open(os.path.join(os.environ["RAVEX_STORAGE_PATH"], "probe.json"), "w") as handle:
    json.dump({"where": mypkg.__file__, "writable": writable}, handle)
'''


@needs_job_user
def test_an_editable_install_serves_every_job_of_the_commit(runs):
    """`-e .` linked the environment to the job's copy of the checkout, gone
    with the job: the next job with the commit could not import the package."""
    config = smith.Config(backend="http://127.0.0.1:1", name="n", hardware="h", runs=str(runs),
                          python=sys.executable, code=True, job_user="job", **RUNTIMES)
    made(runs)
    checkout = runs / "_code" / "example.com-r" / SHA
    (checkout / "src" / "mypkg").mkdir(parents=True)
    (checkout / ".smith-fetched").write_text("")
    (checkout / "requirements.txt").write_text("-e .\n")
    (checkout / "pyproject.toml").write_text(
        '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n\n'
        '[project]\nname = "mypkg"\nversion = "0.1"\n')
    (checkout / "backend.py").write_text(BACKEND)
    (checkout / "src" / "mypkg" / "__init__.py").write_text("")
    (checkout / "probe.py").write_text(IMPORTS)
    seen = []
    for job_id in (1, 2):
        job = {"id": job_id, "script": "probe.py", "mode": "fresh", "name": "n", "run_ref": "r-%03d" % job_id,
               "template": TEMPLATE, "code": {"repo": "https://example.com/r", "commit": SHA}, "params": {}}
        running = smith.Running(job, config)
        running.process.wait(timeout=300)
        with open(running.log_path, encoding="utf-8") as handle:
            assert running.process.returncode == 0, handle.read()
        with open(os.path.join(running.store, "probe.json"), encoding="utf-8") as handle:
            seen.append(json.load(handle))
        running.close()
    # The checkout's own package, which outlives every job and none may write.
    for probe in seen:
        assert probe["where"] == str(checkout / "src" / "mypkg" / "__init__.py") and probe["writable"] is False
    assert os.lstat(checkout / "src" / "mypkg" / "__init__.py").st_uid == os.getuid()
    assert (checkout / ".smith-fetched").exists()


#: Reads the file beside it and writes where it runs, as scripts do.
WORKS = r'''
import json, os
with open("data.txt") as handle:
    data = handle.read()
with open("out.txt", "w") as handle:
    handle.write(data)
with open(os.path.join(os.environ["RAVEX_STORAGE_PATH"], "probe.json"), "w") as handle:
    json.dump({"cwd": os.getcwd(), "data": data}, handle)
'''


@needs_job_user
def test_a_script_reads_beside_itself_and_writes_where_it_runs(runs):
    """A Ravex release's examples and the agent's own scripts are the
    agent's to write: a script writing ./outputs there failed."""
    made(runs)
    release = runs / "_envs" / "_ravex" / TEMPLATE["ravex"]
    (release / "examples").mkdir(parents=True)
    (release / ".smith-ready").write_text("")
    (release / "examples" / "works.py").write_text(WORKS)
    (release / "examples" / "data.txt").write_text("ravex's")
    own = runs.parent / "scripts"
    own.mkdir()
    (own / "works.py").write_text(WORKS)
    (own / "data.txt").write_text("the agent's")
    config = smith.Config(backend="http://127.0.0.1:1", name="n", hardware="h", runs=str(runs),
                          python=sys.executable, job_user="job", **RUNTIMES,
                          scripts={"release": "ravex:examples/works.py", "own": str(own / "works.py")})
    for job_id, (name, data) in enumerate((("release", "ravex's"), ("own", "the agent's")), 1):
        job = {"id": job_id, "script": name, "mode": "fresh", "name": "n", "run_ref": "r-%03d" % job_id,
               "template": TEMPLATE, "params": {}}
        running = smith.Running(job, config)
        running.process.wait(timeout=120)
        with open(running.log_path, encoding="utf-8") as handle:
            assert running.process.returncode == 0, handle.read()
        with open(os.path.join(running.store, "probe.json"), encoding="utf-8") as handle:
            seen = json.load(handle)
        assert seen == {"cwd": os.path.join(running.job_dir, "work"), "data": data}
        assert not (release / "examples" / "out.txt").exists() and not (own / "out.txt").exists()
        running.close()

