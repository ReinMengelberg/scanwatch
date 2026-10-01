"""Test images through the pipeline: each image on its own, no motion gate, no tracking.

  python -m camserver.test ../dataset/scancar/*.jpg            store, upload to <route>/seed/, alert
  python -m camserver.test img.jpg --no-upload --no-notify      only print and keep the units in --spool

Per image: resize to the camera's 960x540, detect, keep the vehicles whose wheels are in the ROI, crop,
classify. Every vehicle on the road becomes a one-frame track, written like a live one (crops, frame.jpg,
annotated.jpg, meta.json) under <route>/seed/<image>[-<n>], then uploaded: scancar/seed/ or car/seed/ on S3.
Alerts go to Discord as usual, tagged "test"; DISCORD_MIN_GAP does not apply.
"""
import argparse
import os
import tempfile
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--det", help="detector model (default DET_MODEL)")
    ap.add_argument("--cls", help="classifier model (default CLS_MODEL)")
    ap.add_argument("--spool", help="where units are written (default: a temp dir, emptied by the upload)")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--no-notify", action="store_true")
    a = ap.parse_args()
    # settings are read at import, so override before importing the package modules
    os.environ["SPOOL_DIR"] = a.spool or tempfile.mkdtemp(prefix="camserver-test-")
    os.environ["DISCORD_MIN_GAP"] = "0"  # every test image may alert
    if a.det:
        os.environ["DET_MODEL"] = a.det
    if a.cls is not None:
        os.environ["CLS_MODEL"] = a.cls

    import cv2

    from . import config as C
    from . import notify, uploader
    from .detect import Detector, load_classifier
    from .pipeline import Pipeline
    from .tracker import Track

    notifier = None if a.no_notify else notify.load()
    pipe = Pipeline(Detector(), load_classifier(), 960, 540, notifier=notifier)
    s3 = None if a.no_upload else uploader._client()
    n = 0
    for img_path in a.images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"{img_path}: not an image")
            continue
        frame = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
        vehicles, persons = pipe.det(frame)
        on_road = [v for v in vehicles if pipe.tracker._inside(v[0])]
        print(f"\n{img_path.name}: {len(vehicles)} vehicles, {len(on_road)} in the ROI")
        for i, det in enumerate(on_road):
            t = time.time()
            tr = Track(i, t, t)
            pipe.tracker._add(tr, t, frame, det, vehicles, persons)
            tid = img_path.stem + (f"-{i}" if len(on_road) > 1 else "")
            unit = pipe.store(tr, day="seed", track_id=tid, tag=f"test: {img_path.name}")
            n += 1
            if s3:
                uploader.upload(s3, unit)
                print(f"  uploaded s3://{C.S3_BUCKET}/{os.path.relpath(unit, C.SPOOL_DIR)}/")
    if notifier:
        notifier.q.join()
        print(f"\ndiscord: {notifier.status}")
    print(f"{len(a.images)} images, {n} tracks, {pipe.stats['alerts']} alerts"
          + ("" if s3 else f", spool {C.SPOOL_DIR}"))


if __name__ == "__main__":
    main()
