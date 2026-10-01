"""Spool (SD card) -> S3. Keys mirror the spool path under S3_PREFIX.

A track folder is uploaded with meta.json last, so on S3 "meta.json exists" means "track complete".
Uploaded units are deleted locally. If S3 is unreachable the spool just grows; when the card gets
below MIN_FREE_GB the oldest spooled days are dropped.
"""
import os
import shutil
import time

from . import config as C

KINDS = ("scancar", "car", "empty")  # spool/S3 top-level folders
status = {"pending": 0, "uploaded": 0, "errors": 0, "last_error": ""}


def _client():
    import boto3

    return boto3.client("s3", endpoint_url=C.S3_ENDPOINT or None, region_name=C.S3_REGION,
                        aws_access_key_id=C.S3_ACCESS_KEY_ID or None,
                        aws_secret_access_key=C.S3_SECRET_ACCESS_KEY or None)


def units() -> list[str]:
    """Complete spool units, oldest first: track folders and empty-street files."""
    out = []
    for kind in KINDS:
        base = os.path.join(C.SPOOL_DIR, kind)
        if not os.path.isdir(base):
            continue
        for day in sorted(os.listdir(base)):
            d = os.path.join(base, day)
            if os.path.isdir(d):
                out += [os.path.join(d, n) for n in sorted(os.listdir(d)) if not n.startswith(".")]
    return sorted(out, key=lambda p: (os.path.basename(os.path.dirname(p)), os.path.basename(p)))


def prune():
    """Keep MIN_FREE_GB free by dropping whole spooled days, oldest first (never today)."""
    os.makedirs(C.SPOOL_DIR, exist_ok=True)
    today = time.strftime("%Y-%m-%d")
    days = sorted({(day, kind) for kind in KINDS
                   for day in (os.listdir(os.path.join(C.SPOOL_DIR, kind)) if os.path.isdir(os.path.join(C.SPOOL_DIR, kind)) else [])
                   if day < today})
    while days and shutil.disk_usage(C.SPOOL_DIR).free < C.MIN_FREE_GB * 1e9:
        day, kind = days.pop(0)
        shutil.rmtree(os.path.join(C.SPOOL_DIR, kind, day), ignore_errors=True)
        print(f"low disk space: dropped spool {kind}/{day}", flush=True)


def upload(s3, unit: str):
    rel = os.path.relpath(unit, C.SPOOL_DIR)
    key = lambda name: "/".join(p for p in (C.S3_PREFIX, rel, name) if p)
    if os.path.isdir(unit):
        names = sorted(os.listdir(unit), key=lambda n: n == "meta.json")  # meta.json last
        for n in names:
            ctype = "application/json" if n.endswith(".json") else "image/jpeg"
            s3.upload_file(os.path.join(unit, n), C.S3_BUCKET, key(n), ExtraArgs={"ContentType": ctype})
        shutil.rmtree(unit)
    else:
        s3.upload_file(unit, C.S3_BUCKET, key(""), ExtraArgs={"ContentType": "image/jpeg"})
        os.remove(unit)


def run():
    last_prune, s3, backoff = 0.0, None, 5
    while True:
        if time.monotonic() - last_prune > 60:
            last_prune = time.monotonic()
            prune()
        todo = units()
        status["pending"] = len(todo)
        if not C.UPLOAD or not todo:
            time.sleep(5)
            continue
        try:
            s3 = s3 or _client()
            for u in todo:
                upload(s3, u)
                status["uploaded"] += 1
                status["pending"] -= 1
            backoff = 5
        except Exception as e:  # network/S3 down: keep spooling, retry with backoff
            status["errors"] += 1
            status["last_error"] = f"{time.strftime('%H:%M:%S')} {type(e).__name__}: {e}"[:300]
            print(f"upload failed, retry in {backoff}s: {status['last_error']}", flush=True)
            s3 = None
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
