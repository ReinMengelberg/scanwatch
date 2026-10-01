"""Motion gate -> detector -> tracker -> (classifier) -> spool. Also the hourly empty-street frame.

Spool layout (mirrors S3, see uploader.py):
  scancar/<date>/<HHMMSS-mmm>/{meta.json, 000.jpg.., frame.jpg}  moving vehicle, P(scancar) >= ROUTE_THRESHOLD
  car/<date>/<HHMMSS-mmm>/...                                     any other moving vehicle (or no model yet)
  empty/<date>/<HH-MM>.jpg                                        empty street, plates/people blurred
The folder is the Pi's verdict, not a label: labels are made on the Mac.
"""
import json
import os
import shutil
import time
from datetime import datetime

import cv2
import numpy as np

from . import config as C
from .crop import blur_plate_zone, crop
from .tracker import Track, Tracker, pick_crops

JPEG = [cv2.IMWRITE_JPEG_QUALITY, 92]


def anonymize(frame, vehicles, persons):
    """AVG: blur plate zones of every vehicle and every person entirely, before a full frame is stored."""
    out = blur_plate_zone(frame, vehicles)
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in persons:
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (0, 0), 15)
    return out


RED, GREEN, GREY, YELLOW = (77, 72, 229), (108, 164, 48), (170, 170, 170), (0, 200, 255)


def _tag(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x, y = max(0, min(int(x), img.shape[1] - tw - 8)), max(th + 8, int(y))
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 8, y + 2), color, -1)
    cv2.putText(img, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def annotate(img, tr: Track, box, conf, vehicles, route, score):
    """For a human glance on S3: ROI, other cars, the path this car drove, its box and verdict."""
    out = img.copy()
    h, w = out.shape[:2]
    if C.ROI:
        cv2.polylines(out, [np.array([(x * w, y * h) for x, y in C.ROI], np.int32)], True, YELLOW, 2, cv2.LINE_AA)
    for v in vehicles:
        if v != box:
            cv2.rectangle(out, (int(v[0]), int(v[1])), (int(v[2]), int(v[3])), GREY, 1)
    color = RED if route == "scancar" else GREEN
    pts = np.array([((b[0] + b[2]) / 2, b[3]) for _, b, _ in tr.path], np.int32)
    cv2.polylines(out, [pts], False, color, 2, cv2.LINE_AA)
    cv2.circle(out, tuple(int(v) for v in pts[0]), 5, color, -1)
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
    verdict = f"scan car {score:.0%}" if route == "scancar" else (f"car (scan {score:.0%})" if score is not None else "car")
    _tag(out, f"{verdict} | det {conf:.0%}", x1, y2 + 28 if y2 + 28 < h - 40 else y1 - 4, color)  # below: keep the pod visible
    when = datetime.fromtimestamp(tr.t_start).strftime("%Y-%m-%d %H:%M:%S")
    _tag(out, f"{when} | {len(tr.path)} dets | travel {tr.travel / w:.0%}", 8, h - 10, (40, 40, 40))
    return out


def _write_dir(final: str, files: dict[str, bytes]):
    """Write a spool unit atomically: a power cut never leaves a half track for the uploader."""
    tmp = os.path.join(C.SPOOL_DIR, ".tmp", os.path.basename(final))
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for name, data in files.items():
        with open(os.path.join(tmp, name), "wb") as f:
            f.write(data)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    os.replace(tmp, final)


def _write_file(final: str, data: bytes):
    os.makedirs(os.path.dirname(final), exist_ok=True)
    tmp = os.path.join(os.path.dirname(final), "." + os.path.basename(final))
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, final)


def _enc(img) -> bytes:
    return cv2.imencode(".jpg", img, JPEG)[1].tobytes()


class Pipeline:
    def __init__(self, detector, classifier=None, w: int = 960, h: int = 540, framing=None, notifier=None):
        self.det, self.clf, self.notifier = detector, classifier, notifier
        self.tracker = Tracker(w, h)
        self.framing = framing or (lambda: {})
        self.last_empty = 0.0
        self.stats = {"tracks": 0, "ignored": 0, "alerts": 0, "empty": 0}
        self.recent: list[dict] = []  # last finished tracks, for /status

    def anonymize(self, frame, vehicles, persons):
        bv, bp = self.det.blur_boxes(frame)
        return anonymize(frame, list(vehicles) + bv, list(persons) + bp)

    # --- per frame -----------------------------------------------------------

    def step(self, t: float, frame):
        vehicles, persons = self.det(frame)
        for tr in self.tracker.update(t, frame, vehicles, persons):
            self.finish(tr)
        return vehicles, persons

    def tick(self, t: float):
        """Call when no frame is processed, so lost tracks still end."""
        for tr in self.tracker.expire(t):
            self.finish(tr)

    def reset(self):
        """Framing changed (gimbal): boxes jump, drop live tracks without storing them."""
        self.tracker.expire(0, all_=True)

    def flush(self):
        for tr in self.tracker.expire(0, all_=True):
            self.finish(tr)

    # --- track end -----------------------------------------------------------

    def finish(self, tr: Track):
        info = dict(id=tr.id, start=datetime.fromtimestamp(tr.t_start).strftime("%H:%M:%S"),
                    dets=len(tr.path), travel=round(tr.travel / self.tracker.w, 2), in_roi=tr.in_roi)
        if not self.tracker.valid(tr):
            self.stats["ignored"] += 1
            self._recent({**info, "stored": False})
            return
        crops = pick_crops(tr)
        scores = self.clf([c[4] for c in crops]) if self.clf else []
        score = float(np.mean(sorted(scores, reverse=True)[:3])) if scores else None
        alert = score is not None and score >= self.clf.threshold
        route = "scancar" if score is not None and score >= C.ROUTE_THRESHOLD else "car"
        start = datetime.fromtimestamp(tr.t_start)
        track_id = start.strftime("%H%M%S-%f")[:-3]

        _, frame, persons, vehicles, box, conf = tr.best_frame
        files = {f"{i:03d}.jpg": _enc(c[4]) for i, c in enumerate(crops)}
        clean = self.anonymize(frame, vehicles, persons)
        files["frame.jpg"] = _enc(clean)
        annotated = _enc(annotate(clean, tr, box, conf, vehicles, route, score)) if C.ANNOTATE or alert else None
        if C.ANNOTATE:
            files["annotated.jpg"] = annotated
        if alert and self.notifier:
            zoom = crop(clean, box)  # from the blurred frame, never the raw crop: Discord is a third party
            zoom = cv2.resize(zoom, (640, round(640 * zoom.shape[0] / zoom.shape[1])), interpolation=cv2.INTER_CUBIC)
            self.notifier.alert(track_id, tr.t_start, score,
                                {"scancar.jpg": annotated, "closeup.jpg": _enc(zoom)})
        meta = dict(
            track_id=track_id,
            ts_start=start.isoformat(timespec="milliseconds"),
            ts_end=datetime.fromtimestamp(tr.t_last).isoformat(timespec="milliseconds"),
            frame_size=[self.tracker.w, self.tracker.h],
            framing=self.framing(),
            roi=C.ROI,
            travel=info["travel"],
            path=[[round(t - tr.t_start, 2), *(round(v) for v in b), round(cf, 3)] for t, b, cf in tr.path],
            crops=[dict(file=f"{i:03d}.jpg", ts=datetime.fromtimestamp(c[1]).isoformat(timespec="milliseconds"),
                        box=[round(v) for v in c[2]], conf=round(c[3], 3), quality=round(c[0], 1),
                        score=round(scores[i], 4) if scores else None) for i, c in enumerate(crops)],
            score=None if score is None else round(score, 4),
            threshold=self.clf.threshold if self.clf else None,
            model=self.clf.version if self.clf else None,
            route=route,
        )
        files["meta.json"] = json.dumps(meta, indent=1).encode()
        _write_dir(os.path.join(C.SPOOL_DIR, route, start.strftime("%Y-%m-%d"), track_id), files)

        self.stats["tracks"] += 1
        self.stats["alerts"] += alert
        self._recent({**info, "stored": True, "track_id": track_id, "route": route, "score": meta["score"], "alert": alert})
        s = f" P(scancar)={score:.2f}" if score is not None else ""
        print(f"{'SCANCAR ' if alert else ''}{route}/{track_id}: {len(tr.path)} dets, travel {info['travel']:.0%}{s}",
              flush=True)

    def _recent(self, info):
        self.recent = ([info] + self.recent)[:20]

    # --- empty street --------------------------------------------------------

    def empty_due(self, t: float) -> bool:
        return t - self.last_empty >= C.EMPTY_EVERY and not self.tracker.tracks

    def store_empty(self, t: float, frame):
        vehicles, persons = self.det(frame)
        self.last_empty = t
        if persons:  # someone on the street after all: try again next round
            self.last_empty = t - C.EMPTY_EVERY + 60
            return
        when = datetime.fromtimestamp(t)
        path = os.path.join(C.SPOOL_DIR, "empty", when.strftime("%Y-%m-%d"), when.strftime("%H-%M") + ".jpg")
        _write_file(path, _enc(self.anonymize(frame, [v[0] for v in vehicles], persons)))
        self.stats["empty"] += 1
        print(f"empty frame {os.path.relpath(path, C.SPOOL_DIR)}", flush=True)


def run_live(pipe: Pipeline, motion, camera):
    """Detector loop: only runs YOLO while there is motion in the ROI or a live track."""
    fid = -1
    while True:
        t = time.time()
        if time.monotonic() < camera.quiet_until:
            pipe.reset()
            time.sleep(0.2)
            continue
        if motion.active() or pipe.tracker.tracks:
            jpeg, fid = camera.latest_frame(fid, timeout=1)
            if jpeg is None:
                pipe.tick(time.time())
                continue
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            pipe.step(time.time(), frame)
            continue
        pipe.tick(t)
        if pipe.empty_due(t) and motion.quiet_for() >= C.EMPTY_QUIET:
            jpeg, fid = camera.latest_frame(fid, timeout=1)
            if jpeg is not None:
                pipe.store_empty(t, cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR))
        time.sleep(0.1)
