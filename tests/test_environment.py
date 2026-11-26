"""A job's environment made from its template (GPU-523): what a machine may
take, what it installs, once, and Ravex's own scripts from its release."""

import io
import json
import os
import sys
import tarfile
import types

import pytest

import environment
import smith
from made import RUNTIMES, TEMPLATE, made


def _config(tmp_path, **more):
    return smith.Config(backend="http://127.0.0.1:1", name="n", hardware="h", runs=str(tmp_path / "runs"),
                        python=sys.executable, scripts={"spiral": "ravex:examples/spiral.py"}, **{**RUNTIMES, **more})


def _job(**template):
    return {"id": 1, "script": "spiral", "mode": "fresh", "name": "n", "run_ref": "r-001", "params": {},
            "template": {**TEMPLATE, **template}}


class TestWhatAMachineTakes:
    def test_a_python_it_has_no_torch_for_is_refused(self, tmp_path):
        with pytest.raises(RuntimeError, match="makes Python 3.12, not 3.13"):
            smith.Running._checked_template(_job(python="3.13"), _config(tmp_path))

    def test_another_torch_is_refused_and_another_build_of_it_is_not(self, tmp_path):
        with pytest.raises(RuntimeError, match="has PyTorch"):
            smith.Running._checked_template(_job(torch="2.10.0+cu128"), _config(tmp_path))
        assert smith.Running._checked_template(_job(torch="2.11.0+cu128"), _config(tmp_path))["ravex"] == "0.19.0"

    def test_a_job_with_no_template_is_refused(self, tmp_path):
        job = _job()
        del job["template"]
        with pytest.raises(RuntimeError, match="template"):
            smith.Running._checked_template(job, _config(tmp_path))

    def test_the_claim_says_what_it_can_make(self, tmp_path):
        asked = []
        agent = types.SimpleNamespace(config=_config(tmp_path), backend=types.SimpleNamespace(
            call=lambda method, url, body=None: asked.append(body) or {}))
        smith.Agent.take(agent)
        assert asked[0]["pythons"] == ["3.12"] and asked[0]["torch"] == "2.11.0+cpu"

    def test_the_locks_say_the_pythons_and_the_torch(self, tmp_path):
        for python in ("312", "311", "313"):
            (tmp_path / ("cp%s.json" % python)).write_text(json.dumps({"files": [
                {"name": "torch-2.11.0+cu128-cp%s-cp%s-manylinux_2_28_x86_64.whl" % (python, python)},
                {"name": "numpy-2.2.0-cp%s-cp%s-manylinux_2_28_x86_64.whl" % (python, python)}]}))
        (tmp_path / "notes.txt").write_text("")
        assert smith.runtimes_of(str(tmp_path)) == (["3.11", "3.12", "3.13"], "2.11.0+cu128")


class TestMakingIt:
    def _faked(self, monkeypatch, tmp_path):
        ran = []

        def run(command, **kwargs):
            ran.append(command)
            if command[1:3] == ["python", "find"]:
                return types.SimpleNamespace(returncode=0, stdout=str(tmp_path / "py" / "python") + "\n", stderr="")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(environment.shutil, "which", lambda name: "uv")
        monkeypatch.setattr(environment.subprocess, "run", run)
        return ran

    def test_python_then_torch_then_the_libraries_then_kept(self, monkeypatch, tmp_path):
        ran = self._faked(monkeypatch, tmp_path)
        python = environment.ensure(TEMPLATE, str(tmp_path))
        assert python == str(tmp_path / "py" / "python")
        assert ran[0] == ["uv", "python", "install", "3.12"]
        # The CPU build by name: plain torch==2.11.0 got PyPI's CUDA one.
        assert "torch==2.11.0+cpu" in ran[2] and environment.CPU_INDEX in ran[2]
        assert "ravex==0.19.0" in ran[3] and "moonclip==0.4.1" in ran[3] and "nvidia-ml-py" not in ran[3]
        # Made once: the next job with the same template installs nothing.
        assert environment.ensure(TEMPLATE, str(tmp_path)) == python and len(ran) == 4

    def test_with_the_images_lock_torch_comes_from_runtime_py(self, monkeypatch, tmp_path):
        ran = self._faked(monkeypatch, tmp_path)
        locks = tmp_path / "locks"
        locks.mkdir()
        (locks / "cp312.json").write_text("{}")
        environment.ensure({**TEMPLATE, "torch": "2.11.0+cu128"}, str(tmp_path / "envs"), str(locks),
                           signed="http://backend/api/cache/global", gpu=True)
        runtime = next(command for command in ran if "--manifest" in command)
        assert runtime[runtime.index("--manifest") + 1] == str(locks / "cp312.json")
        assert runtime[runtime.index("--signed") + 1] == "http://backend/api/cache/global"
        assert "nvidia-ml-py" in ran[-1]

    def test_a_python_the_image_has_no_lock_for_is_said(self, monkeypatch, tmp_path):
        self._faked(monkeypatch, tmp_path)
        (tmp_path / "locks").mkdir()
        with pytest.raises(RuntimeError, match="no PyTorch for Python 3.12"):
            environment.ensure(TEMPLATE, str(tmp_path / "envs"), str(tmp_path / "locks"))

    @pytest.mark.parametrize("bad", [{"python": "3.12; rm -rf /"}, {"ravex": "../x"}, {"moonclip": ""}])
    def test_versions_are_held_to_what_a_version_can_be(self, bad):
        with pytest.raises(ValueError):
            environment.checked({**TEMPLATE, **bad})


def _release(files):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        for name, text in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(text)
            archive.addfile(info, io.BytesIO(text))
    return data.getvalue()


class TestRavexsOwnScripts:
    def test_examples_and_benches_only_and_nothing_out_of_the_folder(self, monkeypatch, tmp_path):
        body = _release({
            "ravex-0.19.0/examples/spiral.py": b"print('spiral')",
            "ravex-0.19.0/bench/cost.py": b"print('cost')",
            "ravex-0.19.0/ravex/__init__.py": b"x",
            "ravex-0.19.0/examples/../../evil.py": b"x",
        })
        class Answer(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(environment.urllib.request, "urlopen", lambda request, timeout: Answer(body))
        tree = environment.release_tree(str(tmp_path), "0.19.0")
        found = sorted(os.path.relpath(os.path.join(d, f), tree) for d, _, fs in os.walk(tree) for f in fs)
        assert found == [".smith-ready", os.path.join("bench", "cost.py"), os.path.join("examples", "spiral.py")]
        assert not (tmp_path / "evil.py").exists()

    @pytest.mark.skipif(os.name == "nt", reason="a job is a process group, POSIX's")
    def test_a_ravex_script_runs_from_the_jobs_release(self, tmp_path):
        made(tmp_path / "runs")
        tree = tmp_path / "runs" / "_envs" / "_ravex" / "0.19.0" / "examples"
        tree.mkdir(parents=True)
        (tree / "spiral.py").write_text("print('from the release')")
        (tree.parent / ".smith-ready").write_text("")
        running = smith.Running(_job(), _config(tmp_path))
        assert running.process.wait(timeout=60) == 0
        with open(running.log_path, encoding="utf-8") as handle:
            assert "from the release" in handle.read()
        assert running.python() == sys.executable
