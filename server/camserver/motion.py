"""Motion gate on the tiny gray frames. It only wakes the detector; it never decides what is stored."""
import time

import cv2
import numpy as np

from . import config as C


def roi_mask(w: int, h: int, roi=None) -> np.ndarray:
    """Boolean mask of the ROI polygon (normalised coords) at w x h. No ROI = everything."""
    roi = C.ROI if roi is None else roi
    if not roi:
        return np.ones((h, w), bool)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.array([(x * w, y * h) for x, y in roi], np.int32)], 1)
    return m.astype(bool)


class Motion:
    def __init__(self):
        self.bg = None
        self.mask = None
        self.level = 0.0  # changed fraction of the ROI in the last frame
        self.last_motion = -1e9  # monotonic time of the last frame above MOTION_MIN
        self.last_reset = time.monotonic()

    def update(self, gray: np.ndarray, gimbal_moving: bool = False, now: float | None = None):
        now = time.monotonic() if now is None else now
        cur = gray.astype(np.float32)
        if self.mask is None or self.mask.shape != cur.shape:
            self.mask = roi_mask(cur.shape[1], cur.shape[0])
        if self.bg is None or gimbal_moving:
            self.bg, self.last_reset = cur, now
            return
        changed = np.count_nonzero((np.abs(cur - self.bg) > C.MOTION_DIFF) & self.mask) / max(1, self.mask.sum())
        if changed > C.MOTION_MAX:
            self.bg, self.last_reset = cur, now  # global change (exposure jump): reset instead of firing
            return
        self.bg += C.BG_RATE * (cur - self.bg)
        self.level = changed
        if changed >= C.MOTION_MIN:
            self.last_motion = now

    def active(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return now - self.last_motion <= C.MOTION_HOLD

    def quiet_for(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        return now - max(self.last_motion, self.last_reset)
