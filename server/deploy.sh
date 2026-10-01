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
# Syncs code + models (crop.py symlink is copied as a real file). Never touches the Pi's .env,
# .venv or spool. pip only runs when requirements.txt changed. Installs/updates the systemd unit.
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

echo "==> install + restart $SERVICE"
ssh "$PI" DIR="$DIR" SERVICE="$SERVICE" bash -s <<'EOF'
set -euo pipefail
cd ~/"$DIR"
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "!! created .env from .env.example: set BIND and S3_* in ~/$DIR/.env, then deploy again"
  exit 1
fi
bind=$(sed -nE 's/^BIND=([^ #]*).*/\1/p' .env)
if [[ -n "$bind" && "$bind" != 127.0.0.1 && "$bind" != 0.0.0.0 ]] && ! ip -4 -o addr | grep -q "inet $bind/"; then
  echo "!! BIND=$bind in ~/$DIR/.env is not an address on this Pi (old WireGuard IP?)."
  echo "!! Set BIND=127.0.0.1 and use ./deploy.sh --ui, or push your local .env with --env."
  exit 1
fi
if ! .venv/bin/python -c "" 2>/dev/null; then  # missing, or a copied macOS venv
  echo "creating .venv"
  rm -rf .venv && python3 -m venv .venv
fi
if ! cmp -s requirements.txt .venv/.deployed-requirements.txt; then
  echo "requirements changed: pip install (first time ~10 min)"
  .venv/bin/pip install -q -r requirements.txt
  cp requirements.txt .venv/.deployed-requirements.txt
fi

# unit file with this user and path
unit=$(sed -e "s#^User=.*#User=$(whoami)#" -e "s#/home/pi/scanwatch/server#$PWD#g" camserver.service)
if ! sudo -n true 2>/dev/null; then
  echo "!! sudo needs a password here; run on the Pi: sudo systemctl restart $SERVICE"
  exit 1
fi
if [[ "$unit" != "$(cat /etc/systemd/system/$SERVICE.service 2>/dev/null)" ]]; then
  echo "$unit" | sudo tee /etc/systemd/system/$SERVICE.service >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable -q "$SERVICE"
  echo "installed /etc/systemd/system/$SERVICE.service"
fi
sudo systemctl restart "$SERVICE"
sleep 4
if systemctl is-active -q "$SERVICE"; then echo "$SERVICE is running"; else echo "!! $SERVICE failed:"; fi
journalctl -u "$SERVICE" -n 15 --no-pager -o cat
EOF

if [[ $LOGS == 1 && $UI == 1 ]]; then
  echo "(--logs and --ui together: showing logs; run ./deploy.sh --ui-only in another terminal for the UI)"
fi
if [[ $LOGS == 1 ]]; then
  ssh -t "$PI" "journalctl -u $SERVICE -f -o cat"
elif [[ $UI == 1 ]]; then
  tunnel
fi
