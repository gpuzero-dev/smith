"""A job that runs a repository's file (GPU-497).

The job names the path of a Python file of the commit, not a script of a
manifest the repository would have to carry: what is held here is that the
path is checked before anything runs - a Python file inside the checkout and
nothing else - and that the file the person chose is the one that runs.
"""

import json
import os
import sys

import pytest

import smith

COMMIT = "a" * 40
REPO = "https://github.com/someone/model"


def _config(tmp_path):
    return smith.Config(backend="http://127.0.0.1:1", name="n", hardware="h", runs=str(tmp_path / "runs"),
                        python=sys.executable, scripts={}, code=True)


def _job(script):
    return {"id": 1, "script": script, "mode": "fresh", "name": "n", "run_ref": "r-001", "params": {"lr": 0.1},
            "code": {"repo": REPO, "commit": COMMIT}}


def _fetched(tmp_path, files):
    """The commit as a node holds it once fetched, so nothing is fetched."""
    tree = tmp_path / "runs" / "_code" / "github.com-someone-model" / COMMIT
    for name, text in files.items():
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        (tree / name).write_text(text)
    (tree / ".smith-ready").write_text(json.dumps({"python": sys.executable}))
    return tree


@pytest.mark.parametrize("script", ["../train.py", "/etc/train.py", "train.sh", "train", "a b.py", "", "x/../y.py"])
def test_a_path_that_is_not_a_python_file_of_the_repository_is_refused(tmp_path, script):
    with pytest.raises(RuntimeError):
        smith.Running._checked_code(_job(script), _config(tmp_path))


def test_the_file_the_person_chose_runs_with_its_parameters(tmp_path):
    _fetched(tmp_path, {
        "train.py": "print('wrong file')",
        "scripts/train.py": "import sys; print('ran', sys.argv[1:])",
    })
    running = smith.Running(_job("scripts/train.py"), _config(tmp_path))
    assert running.process.wait(timeout=60) == 0
    with open(running.log_path, encoding="utf-8") as handle:
        out = handle.read()
    assert "ran ['--lr', '0.1']" in out and "wrong file" not in out


def test_a_file_the_commit_does_not_have_stops_the_job_and_says_so(tmp_path):
    _fetched(tmp_path, {"train.py": "print('x')"})
    running = smith.Running(_job("missing.py"), _config(tmp_path))
    assert running.process.wait(timeout=60) == 2
    with open(running.log_path, encoding="utf-8") as handle:
        assert "has no file missing.py to run" in handle.read()


@pytest.mark.skipif(os.name == "nt", reason="symbolic links need privileges on Windows")
def test_a_link_out_of_the_checkout_is_not_run(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("print('escaped')")
    tree = _fetched(tmp_path, {"train.py": "print('x')"})
    (tree / "link.py").symlink_to(outside)
    running = smith.Running(_job("link.py"), _config(tmp_path))
    assert running.process.wait(timeout=60) == 2
    with open(running.log_path, encoding="utf-8") as handle:
        assert "escaped" not in handle.read()
