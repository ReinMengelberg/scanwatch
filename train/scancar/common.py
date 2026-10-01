"""Shared helpers for the Mac-side scripts: config, crop index, labels, detector, dedupe, split."""
import csv
import hashlib
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INDEX = DATA / "index.csv"      # one row per stored crop (written by bootstrap/pull/synth)
LABELS = ROOT / "labels.csv"    # append-only; last row per crop wins
DATASET = ROOT / "datasets" / "scancar_cls"
RUNS = ROOT / "runs"

INDEX_COLS = ["crop", "track_id", "event_id", "ts", "source", "synthetic", "frame", "x1", "y1", "x2", "y2", "conf",
              "phash", "dup_of"]  # dup_of: box kept for the frame view; its crop is a near-duplicate of that crop
LABEL_COLS = ["crop", "track_id", "event_id", "label", "source", "labeled_at"]
CLASSES = ("scancar", "other")
LABEL_VALUES = ("scancar", "other", "hard_neg", "skip")


def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())


# ---- csv stores -------------------------------------------------------------

def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _append(path: Path, cols: list[str], rows: list[dict]):
    if not rows:
        return
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerows(rows)


def read_index() -> list[dict]:
    return _read(INDEX)


def append_index(rows: list[dict]):
    _append(INDEX, INDEX_COLS, rows)


def read_labels() -> dict[str, dict]:
    """crop -> latest label row. Human labels beat auto labels regardless of order."""
    out: dict[str, dict] = {}
    for r in _read(LABELS):
        prev = out.get(r["crop"])
        if prev and prev["source"] == "human" and r["source"] != "human":
            continue
        out[r["crop"]] = r
    return out


def append_labels(rows: list[dict]):
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for r in rows:
        r.setdefault("labeled_at", now)
    _append(LABELS, LABEL_COLS, rows)


def labelled_crops() -> list[dict]:
    """Index rows joined with their effective label (scancar/other/hard_neg); skips and unlabelled dropped."""
    labels = read_labels()
    out = []
    for r in read_index():
        if not r["crop"]:
            continue
        lab = labels.get(r["crop"])
        if lab and lab["label"] in ("scancar", "other", "hard_neg"):
            out.append({**r, "label": lab["label"]})
    return out


# ---- S3 ---------------------------------------------------------------------

def s3_client():
    env = ROOT.parent / "server" / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("S3_") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.split(" #")[0].strip())
    import boto3

    return boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT"], region_name=os.environ.get("S3_REGION"),
                        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
                        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"]), os.environ["S3_BUCKET"]


# ---- detector ---------------------------------------------------------------

_det = None


def detect(img) -> list[tuple[tuple[float, float, float, float], float]]:
    """Stage-1 detector exactly as on the Pi. Returns [(xyxy box, conf)], biggest first."""
    global _det
    from ultralytics import YOLO

    from .crop import big_enough

    c = cfg()["detector"]
    if _det is None:
        _det = YOLO(c["model"])
    r = _det.predict(img, imgsz=c["imgsz"], classes=c["classes"], conf=c["conf"],
                     agnostic_nms=c["agnostic_nms"], verbose=False)[0]
    dets = [(tuple(b), float(cf)) for b, cf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
    dets = [d for d in dets if big_enough(d[0])]
    return sorted(dets, key=lambda d: -(d[0][2] - d[0][0]) * (d[0][3] - d[0][1]))


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---- dedupe -----------------------------------------------------------------

class Deduper:
    """pHash near-duplicate filter, seeded with every crop already in the index."""

    def __init__(self, max_hamming: int | None = None):
        import imagehash

        self._ih = imagehash
        self.max = cfg()["dedupe"]["max_hamming"] if max_hamming is None else max_hamming
        self.seen = [(imagehash.hex_to_hash(r["phash"]), r["crop"]) for r in read_index() if r["crop"] and r["phash"]]

    def check(self, bgr) -> tuple[str | None, str]:
        """(crop it duplicates or None, hex hash). Call add() to remember a kept crop."""
        from PIL import Image

        h = self._ih.phash(Image.fromarray(bgr[:, :, ::-1]))
        return next((c for o, c in self.seen if h - o <= self.max), None), str(h)

    def add(self, phash: str, crop: str):
        self.seen.append((self._ih.hex_to_hash(phash), crop))


# ---- split ------------------------------------------------------------------

def _h(event: str) -> float:
    return int(hashlib.sha1(event.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def assign_split(rows: list[dict], val_frac: float | None = None) -> dict[str, str]:
    """event_id -> 'train'|'val'. Deterministic hash of the event id, never per frame.

    Synthetic events are always train. If a class has >= 2 real events but none hashed into val,
    its lowest-hash event goes to val so every class is represented there.
    """
    if val_frac is None:
        val_frac = cfg()["split"]["val_frac"]
    real = [r for r in rows if r["synthetic"] != "1"]
    split = {r["event_id"]: "train" for r in rows}
    if val_frac <= 0:
        return split
    for e in {r["event_id"] for r in real}:
        if _h(e) < val_frac:
            split[e] = "val"
    for cls in ("scancar", "other"):
        evs = sorted({r["event_id"] for r in real if r["label"] == cls}, key=_h)
        if len(evs) >= 2 and not any(split[e] == "val" for e in evs):
            split[evs[0]] = "val"
        if len(evs) >= 2 and all(split[e] == "val" for e in evs):
            split[evs[-1]] = "train"
    return split


def rng(seed: int) -> random.Random:
    return random.Random(seed)


def latest_model() -> Path | None:
    best = sorted(RUNS.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return best[-1] if best else None
