#!/usr/bin/env bash
# Stable compatibility entry; the implementation belongs to ops/server.
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/ops/server/run_ecg.sh" "$@"
