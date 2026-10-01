"""Small IoU tracker. A street has a handful of cars at a time; greedy matching is enough.

A finished track only counts if it moved (MIN_TRAVEL) and touched the ROI, so parked cars that
get re-detected whenever someone walks by never become uploads.
"""
import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config as C
from .crop import crop

MAX_CANDIDATES = 12  # best crops kept per track while it is alive


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


@dataclass(eq=False)  # identity comparison; fields hold numpy arrays
class Track:
    id: int
    t_start: float
    t_last: float
    path: list = field(default_factory=list)  # [(t, box, conf)]
    candidates: list = field(default_factory=list)  # [(quality, t, box, conf, crop_bgr)]
    best_frame: tuple | None = None  # (quality, frame_bgr, persons, vehicles, box, conf) for frame.jpg
    in_roi: bool = False

    @property
    def travel(self) -> float:
        xs = [_center(b)[0] for _, b, _ in self.path]
        return max(xs) - min(xs) if xs else 0.0

    def predict(self, t: float) -> tuple:
        """Where the box should be at time t: the last box moved on at the recent horizontal speed."""
        (t1, b1, _) = self.path[-1]
        if len(self.path) < 2:
            return b1
        t0, b0, _ = self.path[max(0, len(self.path) - 4)]  # speed over the last few detections
        vx = (_center(b1)[0] - _center(b0)[0]) / max(t1 - t0, 1e-3)
        dx = vx * (t - t1)
        return (b1[0] + dx, b1[1], b1[2] + dx, b1[3])

    def parked(self, window: float = C.PARKED_AFTER) -> bool:
        """Has not moved (less than 10% of its width) for the last `window` seconds: a parked or waiting car."""
        t_end, last = self.path[-1][0], self.path[-1][1]
        if t_end - self.path[0][0] < window:
            return False
        cx, w = _center(last)[0], last[2] - last[0]
        for t, b, _ in reversed(self.path):
            if t_end - t > window:
                return True
            if abs(_center(b)[0] - cx) > 0.1 * w:
                return False
        return True


def quality(box, conf, others, w, h) -> float:
    """Prefer big, confident, unclipped, unoccluded views (what the Pi classifies)."""
    x1, y1, x2, y2 = box
    q = conf * np.sqrt((x2 - x1) * (y2 - y1))
    if x1 <= 2 or x2 >= w - 2 or y1 <= 2:  # cut off by the frame edge
        q *= 0.3
    occl = max((iou(box, o) for o in others), default=0.0)
    return q * (1 - min(1.0, 2 * occl))


class Tracker:
    def __init__(self, w: int, h: int, roi=None):
        self.w, self.h = w, h
        roi = C.ROI if roi is None else roi
        self.roi = np.array([(x * w, y * h) for x, y in roi], np.float32) if roi else None
        self.tracks: list[Track] = []
        self._ids = itertools.count(1)

    def _inside(self, box) -> bool:
        if self.roi is None:
            return True
        bottom = ((box[0] + box[2]) / 2, box[3])
        return cv2.pointPolygonTest(self.roi, bottom, False) >= 0

    def update(self, t: float, frame, vehicles, persons) -> list[Track]:
        """Feed one detected frame. Returns tracks that just ended (moved or not; see valid())."""
        unmatched = list(range(len(vehicles)))
        inside = [self._inside(v[0]) for v in vehicles]
        pred = [tr.predict(t) for tr in self.tracks]  # fast cars move more than their own width between frames
        pairs = sorted(((iou(pred[i], vehicles[j][0]), i, j)
                        for i in range(len(self.tracks)) for j in unmatched), reverse=True)
        used_t, used_d = set(), set()
        for score, i, j in pairs:
            if i in used_t or j in used_d:
                continue
            if self.tracks[i].in_roi and not inside[j]:
                continue  # a car on the road never hops onto one parked beside it
            if score < C.PARKED_IOU and self.tracks[i].parked():
                continue  # and a parked car's track never swallows a car driving past it
            last = pred[i]
            (cx, cy), (dx, dy) = _center(last), _center(vehicles[j][0])
            # no speed yet (one detection): allow a wider sideways jump, the lanes are told apart vertically
            reach = 0.6 if len(self.tracks[i].path) > 1 else C.FIRST_JUMP
            near = abs(cx - dx) < reach * (last[2] - last[0]) and abs(cy - dy) < 0.6 * (last[3] - last[1])
            if score >= C.TRACK_IOU or near:
                used_t.add(i)
                used_d.add(j)
                self._add(self.tracks[i], t, frame, vehicles[j], vehicles, persons)
        for j in unmatched:
            if j not in used_d:
                tr = Track(next(self._ids), t, t)
                self._add(tr, t, frame, vehicles[j], vehicles, persons)
                self.tracks.append(tr)
        return self.expire(t)

    def _add(self, tr: Track, t, frame, det, vehicles, persons):
        box, conf = det
        tr.t_last = t
        tr.path.append((t, box, conf))
        tr.in_roi |= self._inside(box)
        q = quality(box, conf, [v[0] for v in vehicles if v[0] != box], self.w, self.h)
        if len(tr.candidates) < MAX_CANDIDATES or q > tr.candidates[-1][0]:
            tr.candidates.append((q, t, box, conf, crop(frame, box)))
            tr.candidates.sort(key=lambda c: -c[0])
            del tr.candidates[MAX_CANDIDATES:]
        if tr.best_frame is None or q > tr.best_frame[0]:
            tr.best_frame = (q, frame, persons, [v[0] for v in vehicles], box, conf)

    def expire(self, t: float, all_: bool = False) -> list[Track]:
        """Tracks that ended: lost for TRACK_LOST, or a pass that came to a stop (parked, waiting, or stuck
        on a parked car). A stopped pass is finished right away so it is stored and alerted on time; the
        standing car gets a fresh track on the next frame, which is ignored as parked."""
        done = [tr for tr in self.tracks if all_ or t - tr.t_last > C.TRACK_LOST or self._stopped(tr)]
        self.tracks = [tr for tr in self.tracks if tr not in done]
        return done

    def _stopped(self, tr: Track) -> bool:
        return tr.in_roi and tr.travel >= C.MIN_TRAVEL * self.w and tr.parked()

    def valid(self, tr: Track) -> bool:
        if not tr.in_roi:
            return False
        if len(tr.path) >= 2 and tr.travel >= C.MIN_TRAVEL * self.w:
            return True
        return self._short_pass(tr)

    def _short_pass(self, tr: Track) -> bool:
        """Seen only SHORT_TRACK times or fewer on the road: a fast, blurred or half-hidden car. Kept (recall
        first) unless a parked car stands right there: parked cars give long tracks, not short ones."""
        if len(tr.path) > C.SHORT_TRACK:
            return False
        box = tr.path[-1][1]
        return not any(o is not tr and o.parked() and iou(o.path[-1][1], box) > 0.3 for o in self.tracks)


def pick_crops(tr: Track, n: int = C.CROPS_PER_TRACK) -> list:
    """Best n candidates, preferring ones spread out in time (different views of the car)."""
    picked = []
    for c in tr.candidates:  # sorted by quality
        if all(abs(c[1] - p[1]) >= 0.3 for p in picked):
            picked.append(c)
        if len(picked) == n:
            break
    for c in tr.candidates:
        if len(picked) == n:
            break
        if not any(c is p for p in picked):
            picked.append(c)
    return sorted(picked, key=lambda c: c[1])
