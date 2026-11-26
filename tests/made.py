"""A template's environment as a machine holds it once made (GPU-523): the
marker ``environment.py`` reads, naming this test's own Python, so a job
starts without installing anything."""

import json
import os
import sys

import environment

TEMPLATE = {"python": "3.12", "torch": "2.11.0+cpu", "ravex": "0.19.0", "moonclip": "0.4.1"}
#: What a test agent says it can make.
RUNTIMES = {"pythons": ["3.12"], "torch": "2.11.0+cpu"}


def made(runs, template=TEMPLATE):
    """Mark the environment made under ``runs``; its folder's name."""
    home = environment.folder(os.path.join(str(runs), "_envs"), environment.checked(template))
    os.makedirs(home, exist_ok=True)
    with open(os.path.join(home, ".smith-ready"), "w", encoding="utf-8") as handle:
        json.dump({"python": sys.executable}, handle)
    return os.path.basename(home)
