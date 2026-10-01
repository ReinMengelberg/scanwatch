"""Upload the bootstrap scan car crops to S3 in the camserver layout, as seed data.

  scancar/seed/<track_id>/{000.jpg.., frame.jpg, annotated.jpg, meta.json}   tracks labelled scancar

The crops are scored with the deployed classifier (../server/models/scancar_cls_ncnn_model) and
annotated.jpg is drawn by the Pi's own annotate(): what the Pi would make of this pass. The frame is the
one with the best-scoring crop; the path joins the crops' boxes (they come from different photos).

Negatives are not seeded: they come from cars driving past (the Pi's car/ tracks).

Same layout as the Pi uploads, plus "seed": true and the label in meta.json. Re-runnable: tracks already on S3 are skipped (--force re-uploads).
Credentials from ../server/.env (S3_*), or the environment.

  python seed_s3.py --dry-run
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import cv2

from scancar.common import ROOT, labelled_crops, read_index, s3_client
from scancar.score import score_crops

sys.path.insert(0, str(ROOT.parent / "server"))
from camserver import config as C  # noqa: E402  (reads server/.env: ROI, ROUTE_THRESHOLD)
from camserver.pipeline import annotate  # noqa: E402
from camserver.tracker import Track  # noqa: E402

MODEL = ROOT.parent / "server" / "models" / "scancar_cls_ncnn_model"


def raw_frame(frame_path: str):
    return cv2.resize(cv2.imread(str(ROOT / frame_path)), (960, 540), interpolation=cv2.INTER_AREA)


def enc(img) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def box(r) -> tuple:
    return tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-upload tracks that are already on S3")
    a = ap.parse_args()

    tracks = defaultdict(list)
    for r in labelled_crops():
        if r["source"] == "bootstrap" and r["label"] == "scancar" and r["synthetic"] != "1":  # Pi tracks are on S3 already
            tracks[r["track_id"]].append(r)

    info = json.loads((MODEL / "threshold.json").read_text())
    boxes_by_frame = defaultdict(list)  # every detected vehicle per photo, drawn grey
    for r in read_index():
        if r.get("x1"):
            boxes_by_frame[r["frame"]].append(box(r))

    s3, bucket = (None, None) if a.dry_run else s3_client()
    done = skipped = 0
    for tid, rows in sorted(tracks.items()):
        labels = {r["label"] for r in rows}
        label = "scancar" if "scancar" in labels else ("hard_neg" if "hard_neg" in labels else "other")
        prefix = f"{'scancar' if label == 'scancar' else 'car'}/seed/{tid}"
        if s3 and not a.force and s3.list_objects_v2(Bucket=bucket, Prefix=prefix + "/meta.json").get("KeyCount"):
            skipped += 1
            continue
        scores = score_crops(MODEL, [ROOT / r["crop"] for r in rows])
        score = sum(sorted(scores, reverse=True)[:3]) / min(3, len(scores))  # as the Pi: mean of the top 3
        route = "scancar" if score >= C.ROUTE_THRESHOLD else "car"
        best = rows[max(range(len(rows)), key=lambda i: scores[i])]
        frame = raw_frame(best["frame"])
        t0 = os.path.getmtime(ROOT / best["frame"])  # seed photos have no capture time
        tr = Track(0, t0, t0, path=[(t0, box(r), float(r["conf"])) for r in rows])
        files = {f"{i:03d}.jpg": (ROOT / r["crop"]).read_bytes() for i, r in enumerate(rows)}
        files["frame.jpg"] = enc(frame)
        files["annotated.jpg"] = enc(annotate(frame, tr, box(best), float(best["conf"]),
                                              boxes_by_frame[best["frame"]], route, score))
        meta = dict(track_id=tid, event_id=rows[0]["event_id"], seed=True, label=label,
                    ts_start=rows[0]["ts"] or None, frame_size=[960, 540],
                    crops=[dict(file=f"{i:03d}.jpg", box=[int(v) for v in box(r)], conf=float(r["conf"]),
                                score=round(sc, 4), source_frame=os.path.basename(r["frame"]))
                           for i, (r, sc) in enumerate(zip(rows, scores))],
                    score=round(score, 4), threshold=float(C.CLS_THRESHOLD or info["threshold"]),
                    model=info.get("run"), route=route)
        files["meta.json"] = json.dumps(meta, indent=1).encode()
        print(f"{prefix}/  {len(rows)} crops, P(scancar) {score:.2f} {[round(x, 2) for x in scores]} -> {route}")
        if s3:
            for name in sorted(files, key=lambda n: n == "meta.json"):  # meta.json last = complete
                ctype = "application/json" if name.endswith(".json") else "image/jpeg"
                s3.put_object(Bucket=bucket, Key=f"{prefix}/{name}", Body=files[name], ContentType=ctype)
        done += 1
    print(f"{'would upload' if a.dry_run else 'uploaded'} {done} tracks, {skipped} already on S3")


if __name__ == "__main__":
    main()
