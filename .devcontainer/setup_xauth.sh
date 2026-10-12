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
