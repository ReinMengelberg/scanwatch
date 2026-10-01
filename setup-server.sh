#!/usr/bin/env bash
# On the Pi: install camserver from a fresh clone into a clean ~/scanwatch, then delete the clone.
#
#   git clone --depth 1 git@github.com:ReinMengelberg/scanwatch.git /tmp/scanwatch && /tmp/scanwatch/setup-server.sh
#
# A new version = the same command again. ~/scanwatch only holds what the server runs:
# camserver/, setup.sh, requirements.txt, camserver.service, .env, plus the state that is kept
# across installs (.env, .venv, models/, spool/). Also migrates the old layout (~/scanwatch/server/).
# Options are passed on to setup.sh (e.g. --logs). Env: INSTALL_DIR (default ~/scanwatch), SERVICE (camserver).
set -euo pipefail

main() {
  local src dest stage
  src=$(cd "$(dirname "$0")" && pwd)
  dest="${INSTALL_DIR:-$HOME/scanwatch}"
  [[ -f "$src/server/setup.sh" ]] || { echo "!! run this from a clone of the repo"; exit 1; }
  [[ "$dest" != "$HOME" && "$dest" != / ]] || { echo "!! refusing INSTALL_DIR=$dest"; exit 1; }
  if [[ -e "$dest" && ! -f "$dest/setup.sh" && ! -f "$dest/server/setup.sh" ]]; then
    echo "!! $dest exists but is not a scanwatch install; move it away first"; exit 1
  fi

  # build the new install next to the old one (same filesystem: moving .venv/spool is instant)
  mkdir -p "$(dirname "$dest")"
  stage=$(mktemp -d "$(dirname "$dest")/.scanwatch-new.XXXXXX")
  cp -a "$src/server/camserver" "$src/server/setup.sh" "$src/server/requirements.txt" \
        "$src/server/camserver.service" "$src/server/.env.example" "$stage/"
  rm -f "$stage/camserver/replay.py"
  find "$stage" \( -name __pycache__ -o -name .DS_Store \) -prune -exec rm -rf {} +

  # stop the running service first, so nothing writes to the spool while it moves (setup.sh restarts it)
  local service="${SERVICE:-camserver}"
  if systemctl is-active -q "$service" 2>/dev/null; then
    echo "==> stop $service"
    if [[ $EUID == 0 ]]; then systemctl stop "$service"; else sudo systemctl stop "$service"; fi
  fi

  # keep state from the previous install: flat layout, or the old one with everything in server/
  local f old
  for f in .env .venv models spool; do
    for old in "$dest/$f" "$dest/server/$f"; do
      if [[ -e "$old" && ! -e "$stage/$f" ]]; then mv "$old" "$stage/$f"; fi
    done
  done

  cd /
  rm -rf "$dest"
  mv "$stage" "$dest"
  [[ "$src" == "$dest" ]] || rm -rf "$src"
  echo "==> installed $dest"
  exec "$dest/setup.sh" "$@"
}

main "$@"
