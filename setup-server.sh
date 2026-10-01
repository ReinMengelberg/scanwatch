#!/usr/bin/env bash
# On the Pi, from a clone of this repo: install / update camserver. See server/setup.sh.
#   git clone git@github.com:ReinMengelberg/scanwatch.git && cd scanwatch && ./setup-server.sh
#   update: git pull && ./setup-server.sh
exec "$(dirname "$0")/server/setup.sh" "$@"
