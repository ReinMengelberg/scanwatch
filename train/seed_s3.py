"""Upload the labelled bootstrap crops to S3 in the camserver layout, as seed data.

  scancar/seed/<track_id>/{000.jpg.., frame.jpg, meta.json}   tracks labelled scancar
  car/seed/<track_id>/...                                     tracks labelled other / hard_neg

Same layout as the Pi uploads, plus "seed": true and the label in meta.json. Full frames get
plate zones and people blurred first (AVG). Re-runnable: tracks already on S3 are skipped (--force re-uploads).
Credentials from ../server/.env (S3_*), or the environment.

  python seed_s3.py --dry-run
"""
import argparse
import json
import os
from collections import defaultdict

import cv2

from scancar.common import ROOT, labelled_crops, read_index
from scancar.crop import blur_plate_zone


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


def anonymized_frame(frame_path: str, boxes, person_model) -> bytes:
    img = cv2.resize(cv2.imread(str(ROOT / frame_path)), (960, 540), interpolation=cv2.INTER_AREA)
    r = person_model.predict(img, imgsz=640, conf=0.05, classes=[0, 2, 3, 5, 7], verbose=False)[0]
    dets = list(zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()))
    out = blur_plate_zone(img, boxes + [b for b, c in dets if int(c) != 0])
    for x1, y1, x2, y2 in [b for b, c in dets if int(c) == 0]:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (0, 0), 15)
    return cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-upload tracks that are already on S3")
    a = ap.parse_args()

    boxes_by_frame = defaultdict(list)  # every vehicle box per frame, for blurring
    for r in read_index():
        if r["x1"]:
            boxes_by_frame[r["frame"]].append([float(r[k]) for k in ("x1", "y1", "x2", "y2")])

    tracks = defaultdict(list)
    for r in labelled_crops():
        if r["synthetic"] != "1":
            tracks[r["track_id"]].append(r)

    s3, bucket = (None, None) if a.dry_run else s3_client()
    from ultralytics import YOLO

    person_model = YOLO(str(ROOT / "yolo11n.pt"))
    done = skipped = 0
    for tid, rows in sorted(tracks.items()):
        labels = {r["label"] for r in rows}
        label = "scancar" if "scancar" in labels else ("hard_neg" if "hard_neg" in labels else "other")
        prefix = f"{'scancar' if label == 'scancar' else 'car'}/seed/{tid}"
        if s3 and not a.force and s3.list_objects_v2(Bucket=bucket, Prefix=prefix + "/meta.json").get("KeyCount"):
            skipped += 1
            continue
        files = {f"{i:03d}.jpg": (ROOT / r["crop"]).read_bytes() for i, r in enumerate(rows)}
        files["frame.jpg"] = anonymized_frame(rows[0]["frame"], boxes_by_frame[rows[0]["frame"]], person_model)
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
