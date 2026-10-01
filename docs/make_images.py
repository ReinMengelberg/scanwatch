"""Regenerate the README images from the bootstrap frames. Everything is anonymized first:
these images go into git, so plates (scan car included) and people are blurred.

Plates are found by a license-plate detector (YOLOv11, AGPL-3.0), only used here, never on the Pi:
  curl -L -o train/plate.pt \
    https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1s.pt
  cd train && .venv/bin/python ../docs/make_images.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from scancar.common import read_labels  # noqa: E402
from scancar.crop import crop, pad_box  # noqa: E402

OUT = ROOT / "docs" / "img"
W, H = 960, 540
RED, GREEN, GREY, YELLOW = (77, 72, 229), (108, 164, 48), (150, 150, 150), (0, 200, 255)
SCAN = "../datasets/scancar/b1307830-e51f-4a03-ab60-5e0a1c43de20.jpg"
FRONT = "../datasets/scancar/ad2eb96a-6542-4901-835d-9b2a05b865f0.jpg"
REAR = "../datasets/scancar/c23daf5b-1a29-453d-bbd4-05a9c8b48587.jpg"
OTHER = "../datasets/othercar/123732-162.jpg"  # tinted windows: no visible driver
EMPTY = "../datasets/othercar/123744-820.jpg"  # emptiest frame: only the parked Tesla
FRAMES = [SCAN, FRONT, REAR, OTHER, EMPTY]
ROI = [(0, 0.676), (0.448, 0.593), (1, 0.47), (1, 0.741), (0, 0.852)]

from ultralytics import YOLO  # noqa: E402

_m = YOLO(str(ROOT / "train" / "yolo11n.pt"))
_plates = YOLO(str(ROOT / "train" / "plate.pt"))
PLATE_CONF = 0.05  # low: a missed plate is worse than a stray blur, the shape filter drops misfires
# Plates the detector misses (small and at an angle), found by eye. Check every new frame by eye too.
MISSED_PLATES = {
    "ad2eb96a-6542-4901-835d-9b2a05b865f0.jpg": [[398, 340, 420, 362]],  # scan car, front
    "123732-162.jpg": [[704, 278, 728, 293]],  # blue Audi, front
}
_frame_persons: dict[str, list] = {}  # rel -> person boxes, to spot street furniture


def _overlap(a, b):
    """Intersection over the smaller box: also matches a loose box around the same bollard."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / (min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) + 1e-9)


def load(rel: str):
    return cv2.resize(cv2.imread(str(ROOT / "train" / rel)), (W, H), interpolation=cv2.INTER_AREA)


def _plate_shaped(p, vehicle=None):
    """Drop detector misfires: a plate is wide and flat, small, and small next to its car."""
    w, h = p[2] - p[0], p[3] - p[1]
    if h <= 0 or not 1.3 <= w / h <= 7 or w > 50:
        return False
    return vehicle is None or w <= 0.3 * (vehicle[2] - vehicle[0])


def plates(img, vehicles):
    """Plate boxes: one pass on the whole frame, plus one per vehicle upscaled (small plates far away)."""
    found = [p for p in _plates.predict(img, imgsz=1280, conf=PLATE_CONF, verbose=False)[0].boxes.xyxy.tolist()
             if _plate_shaped(p)]
    for box in vehicles:
        x1, y1, x2, y2 = pad_box(box, img.shape[1], img.shape[0], pad=0.1, top_extra=0)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        r = _plates.predict(img[y1:y2, x1:x2], imgsz=640, conf=PLATE_CONF, verbose=False)[0]
        found += [p for p in ([a + x1, b + y1, c + x1, d + y1] for a, b, c, d in r.boxes.xyxy.tolist())
                  if _plate_shaped(p, box)]
    return found


def persons(rel: str):
    """Low-confidence person boxes, minus street furniture: a box in the same spot in 3+ frames
    (taken minutes apart) is a bollard, not someone standing still."""
    if not _frame_persons:
        for f in FRAMES:
            r = _m.predict(load(f), imgsz=640, conf=0.05, classes=[0], verbose=False)[0]
            _frame_persons[f] = r.boxes.xyxy.tolist()
    return [b for b in _frame_persons[rel]
            if sum(any(_overlap(b, o) > 0.6 for o in boxes) for boxes in _frame_persons.values()) < 3]


def _blur(out, box, pad=0.0, sigma=15):
    x1, y1, x2, y2 = pad_box(box, out.shape[1], out.shape[0], pad=pad, top_extra=0)
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (0, 0), sigma)


def anonymize(rel: str, extra_boxes=()):
    """Frame `rel`, with every license plate (tight boxes) and every person blurred."""
    img = load(rel)
    r = _m.predict(img, imgsz=640, conf=0.05, classes=[2, 3, 5, 7], verbose=False)[0]
    out = img.copy()
    for box in plates(img, list(extra_boxes) + r.boxes.xyxy.tolist()) + MISSED_PLATES.get(Path(rel).name, []):
        _blur(out, box, pad=0.25, sigma=6)
    for box in persons(rel):
        _blur(out, box)
    return out


def tag(img, text, x, y, color, scale=0.7):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = max(0, min(x, img.shape[1] - tw - 8))
    y = max(th + 8, y)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 8, y + 2), color, -1)
    cv2.putText(img, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def boxes_by_frame():
    labels, by = read_labels(), defaultdict(list)
    for r in csv.DictReader(open(ROOT / "train" / "data" / "index.csv")):
        if r["x1"]:
            lab = labels.get(r["crop"] or r["dup_of"], {}).get("label", "")
            by[r["frame"]].append(([float(r[k]) for k in ("x1", "y1", "x2", "y2")], float(r["conf"]), lab, r["track_id"]))
    return by


def labelled(frame_rel, dets, draw_skip=False):
    img = anonymize(frame_rel, [d[0] for d in dets])
    for box, conf, lab, _ in dets:
        if lab == "skip" and not draw_skip:
            continue
        color = {"scancar": RED, "other": GREEN}.get(lab, GREY)
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        tag(img, f"{'scan car' if lab == 'scancar' else lab}  {conf:.0%}", x1, y1 - 4, color)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by = boxes_by_frame()
    scan, front, other = SCAN, FRONT, OTHER

    # 1. hero: detection + verdict on a real pass
    hero = labelled(scan, by[scan])
    cv2.imwrite(str(OUT / "hero.jpg"), hero, [cv2.IMWRITE_JPEG_QUALITY, 88])

    # 2. scan car vs normal traffic, side by side
    pair = np.hstack([cv2.resize(labelled(front, by[front]), (640, 360)),
                      np.full((360, 8, 3), 255, np.uint8),
                      cv2.resize(labelled(other, by[other]), (640, 360))])
    cv2.imwrite(str(OUT / "scancar_vs_traffic.jpg"), pair, [cv2.IMWRITE_JPEG_QUALITY, 88])

    # 3. what the classifier sees: crop.py crops (15% pad + extra headroom for the pod)
    tiles = []
    for f in [FRONT, SCAN, REAR]:
        box = next(d[0] for d in by[f] if d[2] == "scancar")
        c = crop(anonymize(f, [d[0] for d in by[f]]), box)
        tiles.append(cv2.copyMakeBorder(cv2.resize(c, (300, int(300 * c.shape[0] / c.shape[1]))), 0, 0, 0, 8,
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    cv2.imwrite(str(OUT / "crops.jpg"), np.hstack(tiles)[:, :-8], [cv2.IMWRITE_JPEG_QUALITY, 90])

    # 4. ROI + where moving cars actually drove; parked cars are ignored
    img = anonymize(EMPTY)
    overlay = img.copy()
    poly = np.array([(x * W, y * H) for x, y in ROI], np.int32)
    cv2.fillPoly(overlay, [poly], YELLOW)
    img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
    cv2.polylines(img, [poly], True, YELLOW, 3, cv2.LINE_AA)
    for f, dets in by.items():
        for box, _, lab, tid in dets:
            # only cars that drove past: the scan car, and the movers in the traffic frames
            if lab == "scancar" or (lab == "other" and "othercar" in f and "parked" not in tid):
                cv2.circle(img, (int((box[0] + box[2]) / 2), int(box[3])), 7, RED if lab == "scancar" else GREEN, -1)
    tag(img, "ROI: the road, not the sidewalks", 12, 40, (40, 40, 40))
    cv2.imwrite(str(OUT / "roi.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", sorted(p.name for p in OUT.glob("*.jpg")))


if __name__ == "__main__":
    main()
