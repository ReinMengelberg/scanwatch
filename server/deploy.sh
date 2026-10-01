#!/usr/bin/env bash
# Deploy server/ to the Pi and (re)start camserver.
#
#   ./deploy.sh pi@<host>            or once: export PI=pi@<host>, then ./deploy.sh
#   ./deploy.sh pi@<host> --logs     follow the log afterwards
#   ./deploy.sh pi@<host> --env      also push your local server/.env (overwrites the Pi's)
#   ./deploy.sh pi@<host> --ui       afterwards, tunnel the web UI to http://localhost:8080
#   ./deploy.sh pi@<host> --ui-only  just the tunnel, no deploy
#
# <host> is anything ssh accepts: an IP, a DNS name, or a Host alias from ~/.ssh/config.
#
# Syncs code + models, then runs setup.sh on the Pi (venv, pip when requirements.txt changed,
# systemd unit, restart). Never touches the Pi's .env, .venv or spool.
# Use either this or a git clone on the Pi (./setup-server.sh), not both on the same directory.
# Env: PI_DIR (default scanwatch/server, relative to the Pi user's home), PI_SERVICE (camserver),
# UI_PORT (local tunnel port, default 8080; the Pi side uses PORT from the Pi's .env).
set -euo pipefail

LOGS=0 ENV=0 UI=0 DEPLOY=1
for a in "$@"; do
  case "$a" in
    --logs) LOGS=1 ;;
    --env) ENV=1 ;;
    --ui) UI=1 ;;
    --ui-only) UI=1 DEPLOY=0 ;;
    -*) echo "unknown option $a"; exit 1 ;;
    *) PI="$a" ;;
  esac
done
[[ -n "${PI:-}" ]] || { echo "usage: ./deploy.sh user@host [--env] [--logs] [--ui|--ui-only]   (or set PI=user@host)"; exit 1; }
DIR="${PI_DIR:-scanwatch/server}"
SERVICE="${PI_SERVICE:-camserver}"
cd "$(dirname "$0")"

tunnel() {
  # BIND/PORT as the service sees them (the Pi's .env), so the tunnel always hits the right socket
  local remote
  remote=$(ssh "$PI" "cd ~/$DIR 2>/dev/null && sed -nE 's/^(BIND|PORT)=([^ #]*).*/\\2/p' .env | paste -sd: -")
  [[ "$remote" == *:* ]] || remote="127.0.0.1:8080"
  [[ "$remote" == 0.0.0.0:* ]] && remote="127.0.0.1:${remote#*:}"
  echo "==> web UI on http://localhost:${UI_PORT:-8080}  (tunnel to $PI $remote, Ctrl-C to close)"
  exec ssh -N -o ExitOnForwardFailure=yes -L "${UI_PORT:-8080}:$remote" "$PI"
}
[[ $DEPLOY == 1 ]] || tunnel

echo "==> sync to $PI:~/$DIR"
ssh "$PI" "mkdir -p ~/$DIR"
rsync -azL --delete \
  --exclude .venv --exclude .env --exclude spool --exclude __pycache__ --exclude .DS_Store \
  ./ "$PI:$DIR/"
if [[ $ENV == 1 ]]; then
  echo "==> push local .env (overwrites the Pi's)"
  scp -q .env "$PI:$DIR/.env" && ssh "$PI" chmod 600 "$DIR/.env"
fi

echo "==> setup.sh on the Pi"
ssh "$PI" "cd ~/$DIR && SERVICE=$SERVICE ./setup.sh"

if [[ $LOGS == 1 && $UI == 1 ]]; then
  echo "(--logs and --ui together: showing logs; run ./deploy.sh --ui-only in another terminal for the UI)"
fi
if [[ $LOGS == 1 ]]; then
  ssh -t "$PI" "journalctl -u $SERVICE -f -o cat"
elif [[ $UI == 1 ]]; then
  tunnel
fi
