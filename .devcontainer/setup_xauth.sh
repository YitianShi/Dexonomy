#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY is not set; MuJoCo Viewer debugging requires X11/XWayland." >&2
  exit 1
fi
if [[ -z "${XAUTHORITY:-}" || ! -r "${XAUTHORITY}" ]]; then
  echo "XAUTHORITY is not set or is unreadable: ${XAUTHORITY:-<unset>}" >&2
  exit 1
fi
if ! command -v xauth >/dev/null 2>&1; then
  echo "xauth is required on the host for MuJoCo Viewer debugging." >&2
  exit 1
fi

target_file=/tmp/dexonomy-devcontainer.xauth
if [[ -d "${target_file}" ]]; then
  if ! rmdir -- "${target_file}" 2>/dev/null; then
    echo "${target_file} is a directory and cannot be replaced by the X11 auth file." >&2
    echo "Remove that empty, Docker-created directory, then reopen the dev container." >&2
    exit 1
  fi
elif [[ -e "${target_file}" && ! -f "${target_file}" ]]; then
  echo "${target_file} exists but is not a regular file." >&2
  exit 1
fi

temporary_file="$(mktemp /tmp/dexonomy-devcontainer-xauth.XXXXXX)"
cleanup() {
  [[ "${temporary_file}" == /tmp/dexonomy-devcontainer-xauth.* ]] \
    && rm -f -- "${temporary_file}"
}
trap cleanup EXIT

xauth -f "${XAUTHORITY}" nlist "${DISPLAY}" \
  | sed 's/^.... /ffff /' \
  | xauth -f "${temporary_file}" nmerge -
if ! xauth -f "${temporary_file}" list | grep -q 'MIT-MAGIC-COOKIE-1'; then
  echo "No X11 cookie was found for DISPLAY=${DISPLAY}." >&2
  exit 1
fi

chmod 600 "${temporary_file}"
mv -f -- "${temporary_file}" "${target_file}"
