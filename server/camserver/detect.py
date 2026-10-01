"""Stage 1 (YOLO car/truck, plus person) and stage 2 (scan-car classifier)."""
import json
from pathlib import Path

from . import config as C
from .crop import big_enough

PERSON, CAR, TRUCK = 0, 2, 7


class Detector:
    def __init__(self, model: str = C.DET_MODEL):
        from ultralytics import YOLO

        self.model = YOLO(model, task="detect")

    def __call__(self, bgr) -> tuple[list[tuple], list[tuple]]:
        """(vehicles [(box, conf)], persons [box]). Vehicles: car+truck merged by agnostic NMS."""
        r = self.model.predict(bgr, imgsz=C.DET_IMGSZ, conf=C.DET_CONF, classes=[PERSON, CAR, TRUCK],
                               agnostic_nms=False, verbose=False)[0]
        vehicles, persons = [], []
        for box, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            if int(cls) == PERSON:
                persons.append(tuple(box))
            elif big_enough(box, C.MIN_BOX):
                vehicles.append((tuple(box), float(conf)))
        return _merge(vehicles), persons


def _merge(dets, thr=0.7):  # 0.7 = ultralytics default NMS iou
    """Class-agnostic NMS for car vs truck boxes on the same vehicle (training uses agnostic_nms)."""
    from .tracker import iou

    out = []
    for d in sorted(dets, key=lambda d: -d[1]):
        if all(iou(d[0], o[0]) < thr for o in out):
            out.append(d)
    return out


class Classifier:
    """P(scancar) per crop. Threshold from CLS_THRESHOLD or threshold.json next to the model."""

    def __init__(self, model: str = C.CLS_MODEL):
        from ultralytics import YOLO

        self.model = YOLO(model, task="classify")
        self.idx = next(i for i, n in self.model.names.items() if n == "scancar")
        p = Path(model).resolve()
        tj = (p if p.is_dir() else p.parent) / "threshold.json"  # inside an NCNN model dir, else next to the file
        info = json.loads(tj.read_text()) if tj.exists() else {}
        self.version = info.get("run", p.name)
        self.threshold = float(C.CLS_THRESHOLD or info.get("threshold", 0.5))

    def __call__(self, crops_bgr: list) -> list[float]:
        if not crops_bgr:
            return []
        res = self.model.predict(crops_bgr, imgsz=C.CLS_IMGSZ, verbose=False)
        return [float(r.probs.data[self.idx]) for r in res]


def load_classifier():
    if not C.CLS_MODEL or not Path(C.CLS_MODEL).exists():
        print("no classifier model: collecting only", flush=True)
        return None
    clf = Classifier()
    print(f"classifier {C.CLS_MODEL} threshold {clf.threshold}", flush=True)
    return clf
