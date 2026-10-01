"""S3 -> local: mirror the bucket to ../datasets/s3, then index the Pi's tracks for labelling.

Per track: the crops go to data/raw/s3/<date>/<track>/ (label.py only serves crops under data/), the frame
stays in the mirror. One pass = one track = one event, so a pass never ends up in both train and val.
Seed tracks (seed/) are skipped: they came from bootstrap and are indexed already.

Tracks from car/ that no model has scored yet get an auto "other" label (scan cars are rare); review them
in label.py. Everything else stays unlabelled. Re-runnable: indexed tracks and downloaded files are skipped.

  python pull.py
"""
import json
import shutil
from pathlib import Path

import cv2

from scancar.common import RAW, ROOT, Deduper, append_index, append_labels, read_index, s3_client

MIRROR = ROOT.parent / "datasets" / "s3"


def sync() -> int:
    s3, bucket = s3_client()
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        for o in page.get("Contents", []):
            dst = MIRROR / o["Key"]
            if dst.exists() and dst.stat().st_size == o["Size"]:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, o["Key"], str(dst))
            n += 1
    return n


def main():
    print(f"downloaded {sync()} new files -> {MIRROR}")
    known = {r["track_id"] for r in read_index()}
    dd = Deduper()
    index_rows, label_rows, tracks = [], [], 0
    for meta_path in sorted(MIRROR.glob("*/*/*/meta.json")):  # <kind>/<date>/<track>/meta.json
        d = meta_path.parent
        kind, date = d.parent.parent.name, d.parent.name
        tid = f"{date}-{d.name}"
        if date == "seed" or tid in known:
            continue
        meta = json.loads(meta_path.read_text())
        auto = kind == "car" and meta.get("model") is None
        frame = str((d / "frame.jpg").relative_to(ROOT, walk_up=True))
        for c in meta["crops"]:
            src = d / c["file"]
            out = RAW / "s3" / date / d.name / c["file"]
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out)
            dup_of, ph = dd.check(cv2.imread(str(out)))
            rel = str(out.relative_to(ROOT))
            x1, y1, x2, y2 = c["box"]
            index_rows.append(dict(crop="" if dup_of else rel, track_id=tid, event_id=tid, ts=c["ts"], source="s3",
                                   synthetic=0, frame=frame, x1=x1, y1=y1, x2=x2, y2=y2, conf=c["conf"],
                                   phash=ph, dup_of=dup_of or ""))
            if dup_of:
                out.unlink()
                continue
            dd.add(ph, rel)
            if auto:
                label_rows.append(dict(crop=rel, track_id=tid, event_id=tid, label="other", source="auto"))
        tracks += 1
    append_index(index_rows)
    append_labels(label_rows)
    print(f"indexed {tracks} new tracks, {sum(1 for r in index_rows if r['crop'])} crops, "
          f"{len(label_rows)} auto-labelled other")


if __name__ == "__main__":
    main()
