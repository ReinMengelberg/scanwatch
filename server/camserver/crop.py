"""The one crop function. Shared by training and camserver on the Pi: numpy + cv2 only.

Lives in server/ so the Pi needs nothing from train/; train/scancar/crop.py is a symlink to it.
"""
import cv2
import numpy as np

PAD = 0.15        # all sides, relative to box size
TOP_EXTRA = 0.15  # extra headroom so the roof pod is never clipped
MIN_BOX = 40      # px (shorter side) in the 960x540 frame; smaller detections are ignored


def pad_box(box, w, h, pad=PAD, top_extra=TOP_EXTRA):
    """Pad an xyxy box and clamp it to a w x h image. Returns ints."""
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    x1 -= pad * bw
    x2 += pad * bw
    y1 -= (pad + top_extra) * bh
    y2 += pad * bh
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
    return x1, y1, x2, y2


def crop(img: np.ndarray, box, pad=PAD, top_extra=TOP_EXTRA) -> np.ndarray:
    """Crop a vehicle from a BGR frame using the padded, clamped box."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = pad_box(box, w, h, pad, top_extra)
    return img[y1:y2, x1:x2].copy()


def big_enough(box, min_box=MIN_BOX) -> bool:
    x1, y1, x2, y2 = box
    return min(x2 - x1, y2 - y1) >= min_box


def blur_plate_zone(img: np.ndarray, boxes, frac=0.45) -> np.ndarray:
    """Blur the lower part of every vehicle box (where plates are). Used for the README images only.

    No plate detection/OCR on purpose; it blurs the whole zone. Pass boxes from a low-confidence
    detection pass: a missed car keeps its plate.
    """
    out = img.copy()
    h, w = out.shape[:2]
    boxes = [tuple(b) for b in boxes]
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        if any(b != (x1, y1, x2, y2) and (b[2] - b[0]) * (b[3] - b[1]) > (x2 - x1) * (y2 - y1)
               and b[0] <= cx <= b[2] and b[1] <= y2 <= b[1] + 0.5 * (b[3] - b[1]) for b in boxes):
            continue  # part of a bigger vehicle (e.g. the roof pod detected on its own): no plate there
        x1, x2 = max(0, int(x1)), min(w, int(x2))
        y2 = min(h, int(y2))
        y1 = max(0, int(y2 - frac * (y2 - y1)))
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (0, 0), 12)
    return out
