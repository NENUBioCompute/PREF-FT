#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python "$ROOT/train.py" "$@"
