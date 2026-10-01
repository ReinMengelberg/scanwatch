"""P(scancar) for a list of crop files with a trained yolo11n-cls model."""
from pathlib import Path


def score_crops(model_path: Path, paths: list[Path], batch: int = 64) -> list[float]:
    from ultralytics import YOLO

    from .common import cfg

    m = YOLO(str(model_path))
    idx = next(i for i, n in m.names.items() if n == "scancar")
    out = []
    for i in range(0, len(paths), batch):
        res = m.predict([str(p) for p in paths[i:i + batch]], imgsz=cfg()["train"]["imgsz"], verbose=False)
        out += [float(r.probs.data[idx]) for r in res]
    return out
