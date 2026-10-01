#!/usr/bin/env bash
# Install / update camserver on the Pi itself. Run it on the Pi, from a clone of the repo
# (or of just server/):
#
#   ./setup.sh            install everything that is missing, then (re)start the service
#   ./setup.sh --logs     follow the log afterwards
#
# Idempotent: apt only runs for missing packages, pip only when requirements.txt changed, the
# detector models are only exported when missing, the systemd unit only rewritten when it changed.
# Never touches .env (except creating it from .env.example the first time) or the spool.
# deploy.sh runs this same script over ssh after syncing from the Mac.
# Env: SERVICE (systemd unit name, default camserver).
set -euo pipefail

LOGS=0
for a in "$@"; do
  case "$a" in
    --logs) LOGS=1 ;;
    *) echo "usage: ./setup.sh [--logs]"; exit 1 ;;
  esac
done
SERVICE="${SERVICE:-camserver}"
cd "$(dirname "$0")"

# sudo: fine to ask for a password when run by hand, but not under deploy.sh (no terminal)
if ! sudo -n true 2>/dev/null && [[ ! -t 0 ]]; then
  SUDO_HINT=1
fi

need_sudo() {
  if [[ -n "${SUDO_HINT:-}" ]]; then
    echo "!! sudo needs a password here. On the Pi, run: cd $PWD && ./setup.sh"
    exit 1
  fi
}

missing=()
for pkg in ffmpeg v4l-utils python3-venv; do
  dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "ok installed" || missing+=("$pkg")
done
if (( ${#missing[@]} )); then
  need_sudo
  echo "==> apt install ${missing[*]}"
  sudo apt-get update -q && sudo apt-get install -y -q "${missing[@]}"
fi

if [[ ! -f .env ]]; then
  cp .env.example .env && chmod 600 .env
  echo "!! created $PWD/.env from .env.example: fill in S3_* (and the rest), then run setup again"
  exit 1
fi
bind=$(sed -nE 's/^BIND=([^ #]*).*/\1/p' .env)
if [[ -n "$bind" && "$bind" != 127.0.0.1 && "$bind" != 0.0.0.0 ]] && ! ip -4 -o addr | grep -q "inet $bind/"; then
  echo "!! BIND=$bind in $PWD/.env is not an address on this Pi (old WireGuard IP?). Set BIND=127.0.0.1."
  exit 1
fi

if ! .venv/bin/python -c "" 2>/dev/null; then  # missing, or a copied macOS venv
  echo "==> creating .venv"
  rm -rf .venv && python3 -m venv .venv
fi
if ! cmp -s requirements.txt .venv/.deployed-requirements.txt; then
  echo "==> requirements changed: pip install (first time ~10 min)"
  .venv/bin/pip install -q -r requirements.txt
  cp requirements.txt .venv/.deployed-requirements.txt
fi

# Detector models are git-ignored: export them here when they weren't shipped by deploy.sh
for size in 416 640; do
  if [[ ! -d models/yolo11n_${size}_ncnn_model ]]; then
    echo "==> exporting models/yolo11n_${size}_ncnn_model (once, a few minutes)"
    mkdir -p models
    .venv/bin/python -c "import pnnx" 2>/dev/null || .venv/bin/pip install -q pnnx
    (cd models && ../.venv/bin/yolo export model=yolo11n.pt format=ncnn imgsz=$size \
      && mv yolo11n_ncnn_model yolo11n_${size}_ncnn_model)
  fi
done

# unit file with this user and path
unit=$(sed -e "s#^User=.*#User=$(whoami)#" -e "s#/home/pi/scanwatch/server#$PWD#g" camserver.service)
if [[ "$unit" != "$(cat /etc/systemd/system/$SERVICE.service 2>/dev/null)" ]]; then
  need_sudo
  echo "$unit" | sudo tee /etc/systemd/system/$SERVICE.service >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable -q "$SERVICE"
  echo "==> installed /etc/systemd/system/$SERVICE.service"
fi
need_sudo
echo "==> restart $SERVICE"
sudo systemctl restart "$SERVICE"
sleep 4
if systemctl is-active -q "$SERVICE"; then echo "$SERVICE is running"; else echo "!! $SERVICE failed:"; fi
journalctl -u "$SERVICE" -n 15 --no-pager -o cat

if [[ $LOGS == 1 ]]; then
  exec journalctl -u "$SERVICE" -f -o cat
fi
