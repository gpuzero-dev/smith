#!/bin/sh
# The lock of runtime.in for this stage's Python (Dockerfile.gpu, GPU-523):
# resolved with both indexes, written as /runtime/cp<XY>.json, then fetched and
# installed once here, so a lock that cannot be installed fails the build and
# not a node. The wheels stay in /runtime/wheels for the `bundle` target.
set -eu
TAG=$(python -c 'import sys; print("cp%d%d" % sys.version_info[:2])')
# pip reads a wheel's dependencies from the wheel itself where the index
# offers no metadata of its own, as the cu128 one does not: resolving is
# downloading it all once, so its cache is kept between builds.
PIP_NO_CACHE_DIR= pip install --dry-run --ignore-installed --quiet --report /lock/report.json \
  --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple -r /lock/runtime.in
python /lock/runtime.py lock --report /lock/report.json --out "/runtime/$TAG.json"
python /lock/runtime.py fetch --manifest "/runtime/$TAG.json" --dir /runtime/wheels
pip install --no-deps --no-index /runtime/wheels/*.whl
pip check
python -c "import torch; assert torch.__version__ == '2.11.0+cu128', torch.__version__"
