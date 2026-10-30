#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command=(docker)
if [[ -n "${PIPER_DOCKER_EXEC_CONTAINER:-}" ]]; then
  command+=(exec)
  for key in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy; do
    [[ -v "$key" ]] && command+=(-e "$key")
  done
  command+=("$PIPER_DOCKER_EXEC_CONTAINER" docker -H unix:///run/pi05/docker.sock)
fi
args=(build --network host -t "${PIPER_IMAGE:-piper-pi05-full:20261030}")
for key in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy; do
  [[ -v "$key" ]] && args+=(--build-arg "$key")
done
[[ -n "${PIPER_BASE_IMAGE:-}" ]] && args+=(--build-arg "BASE_IMAGE=$PIPER_BASE_IMAGE")
exec "${command[@]}" "${args[@]}" "$root"
