#!/usr/bin/env bash
# On the Mac: download the S3 bucket into ./dataset (car/, scancar/, empty/ as on S3).
#
#   ./fetch-dataset.sh            only new/changed files
#
# Credentials from server/.env (S3_*). Needs the aws CLI (brew install awscli). Never deletes local
# files: dataset/scancar/*.jpg (bootstrap frames) live next to the S3 tracks. To index the tracks for
# labelling, run train/pull.py (it does the same download first).
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
env_file="$root/server/.env"
[[ -f "$env_file" ]] || { echo "!! no $env_file"; exit 1; }

get() { sed -nE "s/^$1=([^ #]*).*/\1/p" "$env_file" | tr -d "'\""; }  # not `source`: ROI contains ';'
export AWS_ACCESS_KEY_ID=$(get S3_ACCESS_KEY_ID) AWS_SECRET_ACCESS_KEY=$(get S3_SECRET_ACCESS_KEY)
export AWS_DEFAULT_REGION=$(get S3_REGION)
bucket=$(get S3_BUCKET) endpoint=$(get S3_ENDPOINT)
[[ -n "$bucket" && -n "$AWS_ACCESS_KEY_ID" ]] || { echo "!! S3_BUCKET / S3_ACCESS_KEY_ID empty in $env_file"; exit 1; }

aws s3 sync "s3://$bucket/" "$root/dataset/" --endpoint-url "$endpoint" --only-show-errors
echo "==> s3://$bucket -> $root/dataset ($(find "$root/dataset" -name meta.json | wc -l | tr -d ' ') tracks)"
