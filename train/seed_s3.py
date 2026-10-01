"""Upload the labelled bootstrap crops to S3 in the camserver layout, as seed data.

  scancar/seed/<track_id>/{000.jpg.., frame.jpg, meta.json}   tracks labelled scancar
  car/seed/<track_id>/...                                     tracks labelled other / hard_neg

Same layout as the Pi uploads, plus "seed": true and the label in meta.json. Re-runnable: tracks already on S3 are skipped (--force re-uploads).
Credentials from ../server/.env (S3_*), or the environment.

  python seed_s3.py --dry-run
"""
import argparse
import json
import os
from collections import defaultdict

import cv2

from scancar.common import ROOT, labelled_crops


def s3_client():
    env = ROOT.parent / "server" / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("S3_") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.split(" #")[0].strip())
    import boto3

    return boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT"], region_name=os.environ.get("S3_REGION"),
                        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
                        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"]), os.environ["S3_BUCKET"]


def raw_frame(frame_path: str) -> bytes:
    img = cv2.resize(cv2.imread(str(ROOT / frame_path)), (960, 540), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-upload tracks that are already on S3")
    a = ap.parse_args()

    tracks = defaultdict(list)
    for r in labelled_crops():
        if r["synthetic"] != "1":
            tracks[r["track_id"]].append(r)

    s3, bucket = (None, None) if a.dry_run else s3_client()
    done = skipped = 0
    for tid, rows in sorted(tracks.items()):
        labels = {r["label"] for r in rows}
        label = "scancar" if "scancar" in labels else ("hard_neg" if "hard_neg" in labels else "other")
        prefix = f"{'scancar' if label == 'scancar' else 'car'}/seed/{tid}"
        if s3 and not a.force and s3.list_objects_v2(Bucket=bucket, Prefix=prefix + "/meta.json").get("KeyCount"):
            skipped += 1
            continue
        files = {f"{i:03d}.jpg": (ROOT / r["crop"]).read_bytes() for i, r in enumerate(rows)}
        files["frame.jpg"] = raw_frame(rows[0]["frame"])
        meta = dict(track_id=tid, event_id=rows[0]["event_id"], seed=True, label=label,
                    ts_start=rows[0]["ts"] or None, frame_size=[960, 540],
                    crops=[dict(file=f"{i:03d}.jpg", box=[int(r[k]) for k in ("x1", "y1", "x2", "y2")],
                                conf=float(r["conf"]), source_frame=os.path.basename(r["frame"]))
                           for i, r in enumerate(rows)],
                    score=None, route="scancar" if label == "scancar" else "car")
        files["meta.json"] = json.dumps(meta, indent=1).encode()
        print(f"{prefix}/  {len(rows)} crops")
        if s3:
            for name in sorted(files, key=lambda n: n == "meta.json"):  # meta.json last = complete
                ctype = "application/json" if name.endswith(".json") else "image/jpeg"
                s3.put_object(Bucket=bucket, Key=f"{prefix}/{name}", Body=files[name], ContentType=ctype)
        done += 1
    print(f"{'would upload' if a.dry_run else 'uploaded'} {done} tracks, {skipped} already on S3")


if __name__ == "__main__":
    main()
