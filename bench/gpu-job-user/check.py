"""The job as a user of its own, on a real GPU (GPU-530).

The tests hold smith's privilege drop to what it must do in the CPU image;
what only a GPU can say is whether the training still works once it no
longer runs as root: CUDA from the job's user, NCCL between ranks under
torchrun, Triton and Inductor compiling into the job's own folder, TileLang
calling nvcc there. This runs smith's own code - ``Running``, its preparer,
its launcher, ``Agent.watch`` - on the machine, with no backend, and says
PASS or FAIL for each, exiting 1 if any failed.

    sudo python3 bench/gpu-job-user/check.py

Run from a copy of smith's tree (it is copied again under ``--root``, as
root's, the way the image has it; nothing of smith is fetched). It needs
root, Linux, an NVIDIA driver, Python 3.11 or newer with ``venv``, gcc
(Triton builds its launcher with it), nvcc for TileLang (skipped without
it), the network to PyPI for torch, uv and TileLang, and about 15 GB of
disk. Two GPUs or more, so NCCL has ranks on different devices: on one,
torchrun runs one rank, since NCCL refuses two on the same GPU. It makes a
user ``job`` (uid 10001, as in the images) if there is none. No secret: the
job's metrics go nowhere, and what Ravex would send is not involved.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SMITH = os.path.dirname(os.path.dirname(HERE))
SHA = "0123456789abcdef0123456789abcdef01234567"
#: Names only: the folder the template's environment is found in. The torch
#: in it is whatever PyPI (or --torch-index) has, which smith only compares
#: with the agent's own ``torch``, set to the same name.
TEMPLATE = {"python": "3.12", "torch": "0.0.0-check", "ravex": "0.19.0", "moonclip": "0.4.1"}

#: The repository's script the job runs, once per rank under torchrun.
PROBE = r'''
import json, os, subprocess, sys, time
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
results = {}


def check(name, fn):
    try:
        results[name] = [True, str(fn())]
    except BaseException as exc:
        results[name] = [False, "%s: %s" % (type(exc).__name__, exc)]


def can_write(folder):
    try:
        path = os.path.join(folder, ".probe-%d" % os.getpid())
        open(path, "w").close()
        os.remove(path)
        return True
    except OSError:
        return False


def files(folder):
    return sum(len(names) for _f, _d, names in os.walk(folder))


import torch
import torch.distributed as dist


def cuda():
    assert torch.cuda.is_available(), "torch sees no CUDA device as uid %d" % os.getuid()
    x = torch.ones(1 << 20, device="cuda")
    return "%s, sum %s" % (torch.cuda.get_device_name(), (x * 2).sum().item())


def nccl():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")) % torch.cuda.device_count())
    dist.init_process_group("nccl")
    t = torch.full((1,), float(rank + 1), device="cuda")
    dist.all_reduce(t)
    want = world * (world + 1) / 2
    assert t.item() == want, "all_reduce gave %s, not %s" % (t.item(), want)
    dist.barrier()
    dist.destroy_process_group()
    return "%d rank(s), all_reduce %s" % (world, t.item())


def compiled():
    fn = torch.compile(lambda x: torch.sin(x) * 2 + x)
    fn(torch.randn(4096, device="cuda"))
    torch.cuda.synchronize()
    triton, inductor = os.environ["TRITON_CACHE_DIR"], os.environ["TORCHINDUCTOR_CACHE_DIR"]
    assert files(triton) and files(inductor), "nothing written to %s or %s" % (triton, inductor)
    return "%d file(s) in %s, %d in %s" % (files(triton), triton, files(inductor), inductor)


def tilelang_kernel():
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        return "SKIP: not installed (%s)" % exc

    @tilelang.jit
    def add(n, block=128, dtype="float32"):
        @T.prim_func
        def main(a: T.Tensor((n,), dtype), b: T.Tensor((n,), dtype), c: T.Tensor((n,), dtype)):
            with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
                for i in T.Parallel(block):
                    c[bx * block + i] = a[bx * block + i] + b[bx * block + i]
        return main

    kernel = add(1024)
    a, b = torch.randn(1024, device="cuda"), torch.randn(1024, device="cuda")
    c = torch.empty_like(a)
    kernel(a, b, c)
    assert torch.allclose(c, a + b)
    folder = os.environ["TILELANG_CACHE_DIR"]
    assert files(folder), "nothing written to %s" % folder
    return "%d file(s) in %s" % (files(folder), folder)


check("cuda", cuda)
check("nccl", nccl)
if rank == 0:
    check("torch.compile", compiled)
    check("tilelang", tilelang_kernel)
    agent = int(args["--agent"])

    def environ():
        try:
            open("/proc/%d/environ" % agent, "rb").read()
        except PermissionError:
            return "refused"
        raise AssertionError("the agent's environment is readable")

    def signal_agent():
        try:
            os.kill(agent, 0)
        except PermissionError:
            return "refused"
        raise AssertionError("the agent can be signalled")

    check("uid", lambda: os.getuid() if os.getuid() == int(args["--uid"]) else 1 / 0)
    check("agent environ", environ)
    check("agent signal", signal_agent)
    for name in ("smith", "envs", "checkout", "shared-cache", "logs"):
        check("no write " + name, lambda name=name: "refused" if not can_write(args["--" + name]) else 1 / 0)
    check("no write venv", lambda: "refused" if not can_write(sys.prefix) else 1 / 0)
    for name, folder in (("store", os.environ["RAVEX_STORAGE_PATH"]), ("home", os.environ["HOME"]),
                         ("cwd", os.getcwd())):
        check("write " + name, lambda folder=folder: "ok" if can_write(folder) else 1 / 0)
    results["pids"] = [subprocess.Popen(["sleep", "600"]).pid,
                       subprocess.Popen(["sleep", "600"], start_new_session=True).pid]
with open(os.path.join(os.environ["RAVEX_STORAGE_PATH"], "probe-%d.json" % rank), "w") as handle:
    json.dump(results, handle)
print("probe rank %d done" % rank, flush=True)
'''


def say(ok, name, detail=""):
    print("%s %s%s" % ("PASS" if ok else "FAIL", name, (": " + detail) if detail else ""), flush=True)
    return ok


def gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:
        with open("/proc/%d/stat" % pid) as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return True


def until(check, seconds=15.0):
    deadline = time.monotonic() + seconds
    while not check() and time.monotonic() < deadline:
        time.sleep(0.2)
    return check()


def sh(*command, **kwargs):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="/opt/smith-check", help="where smith, the tools and the runs go")
    parser.add_argument("--no-tilelang", action="store_true", help="do not try to install TileLang")
    parser.add_argument("--torch-index", default="", help="an index for torch, e.g. "
                        "https://download.pytorch.org/whl/cu128 for a driver older than PyPI's build wants")
    parser.add_argument("--no-gpu-check", action="store_true", help="skip smith's GPU check, to try this "
                        "script's plumbing on a machine without a GPU (the CUDA checks then fail)")
    args = parser.parse_args()
    if os.name == "nt" or os.geteuid() != 0:
        raise SystemExit("run as root on Linux: smith drops to the job's user, which needs root")
    root = args.root
    # The node's image: smith root's and read-only to the job, uv on the PATH.
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    os.chmod(root, 0o755)
    smith_dir = os.path.join(root, "smith")
    shutil.copytree(SMITH, smith_dir, ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv", "runs"))
    sh("chmod", "-R", "u=rwX,go=rX", smith_dir)
    sh(sys.executable, "-m", "venv", os.path.join(root, "tools"))
    sh(os.path.join(root, "tools", "bin", "pip"), "install", "-q", "uv==0.13.8")
    os.environ["PATH"] = os.path.join(root, "tools", "bin") + os.pathsep + os.environ["PATH"]
    try:
        import pwd
        pwd.getpwnam("job")
    except KeyError:
        sh("useradd", "--no-create-home", "--uid", "10001", "--user-group", "job")
    import pwd
    job = pwd.getpwnam("job")
    os.environ["SMITH_JOB_USER"] = "job"
    sys.path.insert(0, smith_dir)
    import environment
    import smith

    for device in sorted(glob.glob("/dev/nvidia*")):
        print("info %s mode %o" % (device, os.stat(device).st_mode & 0o777))
    gpus = int(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout.count("GPU "))
    print("info %d GPU(s)" % gpus)

    # The template's environment as environment.py makes it: a Python of uv's
    # with torch installed into it, root's. PyPI's torch is CUDA's.
    runs = os.path.join(root, "runs")
    envs = os.path.join(runs, "_envs")
    home = environment.folder(envs, environment.checked(TEMPLATE))
    os.makedirs(home)
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=os.path.join(home, "python"),
               UV_CACHE_DIR=os.path.join(runs, "_cache", "uv"))
    sh("uv", "python", "install", TEMPLATE["python"], env=env)
    python = subprocess.run(["uv", "python", "find", "--managed-python", TEMPLATE["python"]], env=env,
                            capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    # As an extra index, which with uv outranks PyPI: given as the main one,
    # PyPI's own CUDA build still came instead.
    index = ["--extra-index-url", args.torch_index] if args.torch_index else []
    sh("uv", "pip", "install", "-q", "--python", python, "--break-system-packages", "torch", "numpy", *index, env=env)
    if not args.no_tilelang:
        if subprocess.run(["uv", "pip", "install", "-q", "--python", python, "--break-system-packages", "tilelang"],
                          env=env).returncode != 0:
            print("info TileLang could not be installed; its check is skipped", flush=True)
    print("info torch %s" % subprocess.run([python, "-c", "import torch; print(torch.__version__)"],
                                           capture_output=True, text=True, check=True).stdout.strip())
    template, made = TEMPLATE, home
    with open(os.path.join(made, ".smith-ready"), "w") as handle:
        json.dump({"python": python}, handle)

    # A repository's commit as fetched, whose job makes a virtual environment.
    checkout = os.path.join(runs, "_code", "example.com-r", SHA)
    os.makedirs(checkout)
    for name, text in ((".smith-fetched", ""), ("requirements.txt", "# nothing\n"), ("probe.py", PROBE)):
        with open(os.path.join(checkout, name), "w") as handle:
            handle.write(text)
    # In a folder of its own: the job runs in a copy of the script's folder,
    # which must not hold the runs.
    os.makedirs(os.path.join(root, "scripts"))
    sleep = os.path.join(root, "scripts", "sleep.py")
    with open(sleep, "w") as handle:
        handle.write("import subprocess, time\nsubprocess.Popen(['sleep', '600'])\ntime.sleep(600)\n")

    config = smith.Config(backend="http://127.0.0.1:1", name="check", hardware="gpu-check", runs=runs,
                          python=sys.executable, code=True, check_gpus=not args.no_gpu_check, gpus=gpus, job_user="job",
                          scripts={"sleep": sleep}, pythons=[TEMPLATE["python"]],
                          torch=TEMPLATE["torch"])
    agent = smith.Agent(config)
    agent.send_logs = lambda: None
    agent._report_until_heard = lambda *a: None
    agent.seal_runs()
    os.makedirs(os.path.join(runs, "_cache"), exist_ok=True)
    claimed = {"id": 1, "script": "probe.py", "mode": "fresh", "name": "check", "run_ref": "r-check",
           "template": template, "code": {"repo": "https://example.com/r", "commit": SHA},
           "params": {"agent": os.getpid(), "uid": job.pw_uid, "smith": smith_dir, "envs": made,
                      "checkout": checkout, "shared-cache": os.path.join(runs, "_cache"),
                      "logs": os.path.join(runs, "_logs")}}
    running = smith.Running(claimed, config)
    agent.current = running
    code = running.process.wait(timeout=3600)
    ok = say(code == 0, "job exit code", str(code))
    for output in running.outputs:
        try:
            print("---- %s\n%s" % (output.path, open(output.path, errors="replace").read()[-4000:]))
        except OSError:
            pass
    rank0 = {}
    for rank in range(gpus):
        try:
            with open(os.path.join(running.store, "probe-%d.json" % rank)) as handle:
                seen = json.load(handle)
        except OSError as exc:
            ok = say(False, "rank %d results" % rank, str(exc)) and ok
            continue
        if rank == 0:
            rank0 = seen
        for name, (passed, detail) in sorted((k, v) for k, v in seen.items() if k != "pids"):
            ok = say(passed, "rank %d %s" % (rank, name), detail) and ok
    if gpus == 1:
        print("info one GPU: NCCL ran one rank; rent two GPUs for ranks on different devices")

    # What the job compiled is in its own folder, the machine's shared one is untouched.
    for name in ("triton", "inductor", "tilelang"):
        shared = os.path.join(runs, "_cache", name)
        ok = say(not os.path.exists(shared) or not os.listdir(shared), "shared %s cache untouched" % name) and ok
    owners = {os.lstat(os.path.join(folder, name)).st_uid
              for folder, _dirs, names in os.walk(running.job_dir) for name in names}
    ok = say(owners <= {job.pw_uid}, "job's folder written by the job's user", str(owners)) and ok
    venv = glob.glob(os.path.join(checkout, ".smith-venv-*"))
    sealed = venv and all(os.lstat(os.path.join(folder, name)).st_uid == 0 and (
        os.path.islink(os.path.join(folder, name)) or not os.lstat(os.path.join(folder, name)).st_mode & 0o7022)
        for folder, _dirs, names in os.walk(venv[0]) for name in names)
    ok = say(bool(sealed), "repository's virtual environment is root's and read-only", str(venv)) and ok

    # The job's group goes with it, and so does what left it: under torchrun
    # each rank is in a session of its own, and so is what rank 0 started.
    in_group, escaped = rank0.get("pids", [0, 0])
    agent.watch()
    ok = say(until(lambda: gone(in_group)), "what a rank started killed when the job ended") and ok
    ok = say(until(lambda: gone(escaped)), "what left with setsid killed when the job ended") and ok
    stopped = smith.Running({"id": 2, "script": "sleep", "mode": "fresh", "name": "sleep", "run_ref": "r-sleep",
                             "params": {}, "template": template}, smith.Config(**dict(config.__dict__, gpus=1)))
    time.sleep(3)
    stopped.stop()
    ok = say(until(lambda: stopped.process.poll() is not None), "a stopped job ends") and ok
    leftover = subprocess.run(["pgrep", "-u", str(job.pw_uid), "-x", "sleep"], capture_output=True, text=True).stdout
    ok = say(until(lambda: not subprocess.run(["pgrep", "-u", str(job.pw_uid), "-x", "sleep"],
                                              capture_output=True).stdout), "a stopped job's group killed",
             leftover.strip()) and ok
    stopped.close()
    running.close()
    print("ALL PASS" if ok else "SOME FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
