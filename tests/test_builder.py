"""What the builder pushes is the wheel it was asked for, and nothing else
the build left (GPU-452): the build ran a package's own setup.py."""

import json
import os

import pytest

import builder
import runtime

ITEM = {"id": 7, "package": "flash_attn", "version": "2.8.3", "key": {}}


def test_a_wheel_is_the_packages_by_its_name():
    assert builder.named("flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl", "flash-attn", "2.8.3")
    assert builder.named("Flash.Attn-2.8.3-1-cp312-cp312-linux_x86_64.whl", "flash_attn", "2.8.3")
    assert not builder.named("numpy-99.0-py3-none-any.whl", "flash-attn", "2.8.3")
    assert not builder.named("flash_attn-2.8.4-cp312-cp312-linux_x86_64.whl", "flash-attn", "2.8.3")
    assert not builder.named("flash_attn-2.8.3.tar.gz", "flash-attn", "2.8.3")


def test_only_the_one_wheel_asked_for_is_taken(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "numpy-99.0-py3-none-any.whl").write_bytes(b"not what was asked")
    with pytest.raises(RuntimeError, match="not its wheel"):
        builder.the_wheel(ITEM, str(out))

    (out / "flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl").write_bytes(b"wheel")
    with pytest.raises(RuntimeError, match="2 wheels"):
        builder.the_wheel(ITEM, str(out))

    os.remove(out / "numpy-99.0-py3-none-any.whl")
    assert builder.the_wheel(ITEM, str(out)) == "flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"

    # What the build said went wrong is what the backend hears.
    (out / builder.ERROR_FILE).write_text("nvcc was killed")
    with pytest.raises(RuntimeError, match="nvcc was killed"):
        builder.the_wheel(ITEM, str(out))


def test_pending_names_no_build_without_an_image_digest(monkeypatch, capsys):
    monkeypatch.setenv("SMITH_BACKEND", "https://backend.example")
    monkeypatch.setenv("SMITH_BUILDER_TOKEN", "token")
    monkeypatch.setattr(builder, "ask", lambda path, body=None: {"queued": 3, "image": "smith-gpu:latest"})
    assert builder.main(["--pending"]) == 0
    assert capsys.readouterr().out.split() == ["pending=0", "image="]

    digest = "sha256:" + "a" * 64
    monkeypatch.setattr(builder, "ask", lambda path, body=None: {"queued": 3, "image": digest})
    assert builder.main(["--pending"]) == 0
    assert capsys.readouterr().out.split() == ["pending=1", "image=" + digest]


@pytest.mark.parametrize("name", ["../../x.whl", "a/b.whl", "/etc/x.whl", "..whl", "torch.tar.gz"])
def test_a_lock_that_names_a_path_is_refused(tmp_path, name):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({"version": 1, "files": [{"name": name, "url": "", "sha256": "ab"}]}))
    with pytest.raises(SystemExit, match="not a wheel's name"):
        runtime.read_manifest(str(path))
