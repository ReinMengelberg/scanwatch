"""Run the pipeline offline on a folder of frames or a video. No camera, no upload.

  python -m camserver.replay ../dataset/othercar --det ../train/yolo11n.pt --spool /tmp/spool
  python -m camserver.replay street.mp4 --fps 10

Frame times come from an index.csv next to the frames (camserver's old format), else --fps.
Use it to tune MOTION_*, MIN_TRAVEL, TRACK_* and the ROI before touching the Pi.
"""
import argparse
import csv
import os
from datetime import datetime
from pathlib import Path


def frames(src: Path, fps: float):
    import cv2

    if src.is_dir():
        times = {}
        if (src / "index.csv").exists():
            with (src / "index.csv").open() as f:
                times = {r["file"]: datetime.fromisoformat(r["time"]).timestamp() for r in csv.DictReader(f)}
        for i, p in enumerate(sorted(src.glob("*.jpg"))):
            yield times.get(p.name, i / fps), cv2.imread(str(p))
    else:
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or fps
        i = 0
        while (ok := cap.read())[0]:
            yield i / fps, ok[1]
            i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--det", help="detector model (default DET_MODEL)")
    ap.add_argument("--cls", help="classifier model (default CLS_MODEL)")
    ap.add_argument("--spool", default="/tmp/camserver-replay", help="where tracks are written")
    ap.add_argument("--fps", type=float, default=10)
    a = ap.parse_args()
    # settings are read at import, so override before importing the package modules
    os.environ["SPOOL_DIR"] = a.spool
    if a.det:
        os.environ["DET_MODEL"] = a.det
    if a.cls is not None:
        os.environ["CLS_MODEL"] = a.cls

    import cv2

    from . import config as C
    from .detect import Detector, load_classifier
    from .motion import Motion
    from .pipeline import Pipeline

    W = int(C.VIEW_WIDTH)
    mw, mh = map(int, C.MOTION_SIZE.split("x"))
    motion, pipe, n, ran = Motion(), None, 0, 0
    for t, img in frames(a.src, a.fps):
        h = round(W * img.shape[0] / img.shape[1] / 2) * 2
        img = cv2.resize(img, (W, h), interpolation=cv2.INTER_AREA)
        if pipe is None:
            pipe = Pipeline(Detector(), load_classifier(), W, h)
        motion.update(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (mw, mh), interpolation=cv2.INTER_AREA), now=t)
        n += 1
        if motion.active(t) or pipe.tracker.tracks:
            vehicles, persons = pipe.step(t, img)
            ran += 1
            print(f"{t:.1f}s motion {motion.level:.1%}: {len(vehicles)} vehicles, {len(persons)} persons, "
                  f"{len(pipe.tracker.tracks)} live tracks", flush=True)
        else:
            pipe.tick(t)
            print(f"{t:.1f}s motion {motion.level:.1%}: idle", flush=True)
    if pipe:
        pipe.flush()
        print(f"\n{n} frames, detector ran on {ran}. stats {pipe.stats}")
        for r in pipe.recent[::-1]:
            print(" ", r)
        print(f"spool: {a.spool}")


if __name__ == "__main__":
    main()
