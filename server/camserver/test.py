"""Test images through the pipeline: each image on its own, no motion gate, no tracking.

  python -m camserver.test ../dataset/scancar/*.jpg            store, upload to <route>/seed/, alert
  python -m camserver.test img.jpg --no-upload --no-notify      only print and keep the units in --spool

Per image: resize to the camera's 960x540, detect, keep the vehicles whose wheels are in the ROI, crop,
classify. Every vehicle on the road becomes a one-frame track, written like a live one (crops, frame.jpg,
annotated.jpg, meta.json) under <route>/seed/<image>[-<n>]. Only annotated.jpg is uploaded, to
scancar/seed/<image>/ or car/seed/<image>/ on S3; the rest stays in --spool, or is deleted without it.
Prints the time per stage (detect, classify, store, upload), after a warm-up; run it on the Pi to see
what one frame costs there. Alerts go to Discord as usual (the annotated image), tagged "test"; DISCORD_MIN_GAP does not apply.
"""
import argparse
import os
import shutil
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
    os.environ["ANNOTATE"] = "1"  # annotated.jpg is what gets uploaded
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
    import numpy as np

    clf_ms = []  # time every classifier call made by pipe.store
    if pipe.clf:
        real = pipe.clf

        class Timed:
            threshold, version = real.threshold, real.version

            def __call__(self, crops):
                t0 = time.perf_counter()
                out = real(crops)
                clf_ms.append((time.perf_counter() - t0) * 1000)
                return out
        pipe.clf = Timed()
    blank = np.zeros((540, 960, 3), np.uint8)
    pipe.det(blank)  # warm-up: the first call loads the model
    if pipe.clf:
        real([blank[:224, :224]])
    times = []
    n = 0
    for img_path in a.images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"{img_path}: not an image")
            continue
        frame = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        vehicles, persons = pipe.det(frame)
        det_ms = (time.perf_counter() - t0) * 1000
        on_road = [v for v in vehicles if pipe.tracker._inside(v[0])]
        print(f"\n{img_path.name}: {len(vehicles)} vehicles, {len(on_road)} in the ROI")
        for i, det in enumerate(on_road):
            t = time.time()
            tr = Track(i, t, t)
            pipe.tracker._add(tr, t, frame, det, vehicles, persons)
            tid = img_path.stem + (f"-{i}" if len(on_road) > 1 else "")
            clf_ms.clear()
            t0 = time.perf_counter()
            unit = pipe.store(tr, day="seed", track_id=tid, tag=f"test: {img_path.name}")
            store_ms = (time.perf_counter() - t0) * 1000
            n += 1
            up_ms = 0.0
            t0 = time.perf_counter()
            if s3:
                key = "/".join(p for p in (C.S3_PREFIX, os.path.relpath(unit, C.SPOOL_DIR), "annotated.jpg") if p)
                s3.upload_file(os.path.join(unit, "annotated.jpg"), C.S3_BUCKET, key,
                               ExtraArgs={"ContentType": "image/jpeg"})
                print(f"  uploaded s3://{C.S3_BUCKET}/{key}")
                if not a.spool:
                    shutil.rmtree(unit)
                up_ms = (time.perf_counter() - t0) * 1000
            c_ms = sum(clf_ms)
            times.append((det_ms, c_ms, store_ms - c_ms, up_ms))
            print(f"  time: detect {det_ms:.0f} ms, classify {c_ms:.0f} ms, store {store_ms - c_ms:.0f} ms"
                  + (f", upload {up_ms:.0f} ms" if s3 else ""))
        if not on_road:
            print(f"  time: detect {det_ms:.0f} ms")
    if times:
        m = np.mean(times, axis=0)
        print(f"\nmean per track: detect {m[0]:.0f} ms, classify {m[1]:.0f} ms, store {m[2]:.0f} ms"
              + (f", upload {m[3]:.0f} ms" if s3 else ""))
    if notifier:
        notifier.q.join()
        print(f"\ndiscord: {notifier.status}")
    print(f"{len(a.images)} images, {n} tracks, {pipe.stats['alerts']} alerts"
          + ("" if s3 else f", spool {C.SPOOL_DIR}"))
    if s3 and not a.spool:
        shutil.rmtree(C.SPOOL_DIR, ignore_errors=True)  # the temp spool, emptied by the uploads


if __name__ == "__main__":
    main()
