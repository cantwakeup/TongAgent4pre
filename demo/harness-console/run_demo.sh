#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PORT=${PORT:-8765}
BIND=${BIND:-127.0.0.1}

cd "$REPOSITORY_ROOT"
python demo/harness-console/generate_demo_bundle.py --check

printf 'TongAgent Harness Console\n'
printf 'URL: http://%s:%s\n' "$BIND" "$PORT"
printf 'Mode: offline formal-artifact replay; no model or public network\n'

exec python -m http.server "$PORT" --bind "$BIND" --directory demo/harness-console
