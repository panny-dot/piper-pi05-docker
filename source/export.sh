#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 ]] || { echo 'Usage: bash export.sh /absolute/new-image.tar' >&2; exit 2; }
[[ ! -e "$1" ]] || { echo 'Output already exists; choose a new filename' >&2; exit 2; }
command=(docker)
if [[ -n "${PIPER_DOCKER_EXEC_CONTAINER:-}" ]]; then
  command+=(exec "$PIPER_DOCKER_EXEC_CONTAINER" docker -H unix:///run/pi05/docker.sock)
fi
"${command[@]}" image save --output "$1" "${PIPER_IMAGE:-piper-pi05-full:20261030}"
parent="$(cd -- "$(dirname -- "$1")" && pwd)"
name="$(basename -- "$1")"
(cd -- "$parent" && sha256sum -- "$name" > "$name.sha256")
