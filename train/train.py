"""Train yolo11n-cls on datasets/scancar_cls, evaluate per track, export NCNN for the Pi.

Evaluation is per track like the Pi decides: track score = mean of the top-k crop scores (eval.topk).
The alert threshold favours recall (a false alarm costs a glance, a missed scan car is gone): just
below the weakest scan car val track, within eval.threshold_min..threshold_max; without held-out scan
cars (build_dataset.py --all) it is threshold_max. Output (see README):

  ../server/models/scancar_cls_ncnn_model/   NCNN model + threshold.json (CLS_MODEL=models/scancar_cls_ncnn_model)

  python build_dataset.py && python train.py
  python train.py --run cls-20261001-160231    # skip training: evaluate + export an existing run
"""
import argparse
import json
import shutil
import time
from collections import defaultdict

from scancar.common import DATASET, ROOT, RUNS, cfg, labelled_crops
from scancar.score import score_crops

EXPORT = ROOT.parent / "server" / "models" / "scancar_cls_ncnn_model"


def evaluate(model, topk: int, lo: float, hi: float) -> tuple[list, float]:
    """[(track, label, score)] for every val track, and the threshold."""
    val = {p.resolve() for p in (DATASET / "val").rglob("*.jpg")}
    tracks = defaultdict(list)
    for r in labelled_crops():
        if (ROOT / r["crop"]).resolve() in val:
            tracks[r["track_id"], "scancar" if r["label"] == "scancar" else "other"].append(ROOT / r["crop"])
    rows = []
    for (tid, label), paths in sorted(tracks.items()):
        s = sorted(score_crops(model, paths), reverse=True)[:topk]
        rows.append((tid, label, sum(s) / len(s)))
    pos = [s for _, lab, s in rows if lab == "scancar"]
    return rows, round(max(lo, min(hi, min(pos, default=hi) - 0.05)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="existing run in runs/ to evaluate and export instead of training")
    a = ap.parse_args()
    c = cfg()
    t = c["train"]
    from ultralytics import YOLO

    name = a.run or time.strftime("cls-%Y%m%d-%H%M%S")
    if not a.run:
        YOLO(t["model"]).train(
            data=str(DATASET), project=str(RUNS), name=name, device="mps", imgsz=t["imgsz"], epochs=t["epochs"],
            batch=t["batch"], patience=t["patience"], fliplr=t["fliplr"], flipud=t["flipud"],
            auto_augment=t["auto_augment"], hsv_h=t["hsv_h"], hsv_s=t["hsv_s"], hsv_v=t["hsv_v"],
            scale=t["scale"], erasing=t["erasing"], plots=False, verbose=False)
    best = RUNS / name / "weights" / "best.pt"

    e = c["eval"]
    rows, thr = evaluate(best, e["topk"], e["threshold_min"], e["threshold_max"])
    print(f"\nper-track val scores ({best}):")
    for tid, lab, s in sorted(rows, key=lambda r: -r[2]):
        print(f"  {s:.3f}  {lab:8s} {tid}")
    pos = [s for _, lab, s in rows if lab == "scancar"]
    hit = sum(s >= thr for s in pos)
    print(f"threshold {thr}: {hit}/{len(pos)} scan car tracks caught, "
          f"{sum(s >= thr for _, lab, s in rows if lab == 'other')} false alarms")

    out = YOLO(str(best)).export(format="ncnn", imgsz=t["imgsz"])
    shutil.rmtree(EXPORT, ignore_errors=True)
    shutil.copytree(out, EXPORT)
    (EXPORT / "threshold.json").write_text(json.dumps(dict(
        threshold=thr, run=name, topk=c["eval"]["topk"],
        val_tracks=dict(scancar=len(pos), other=len(rows) - len(pos), caught=hit)), indent=1))
    print(f"-> {EXPORT}")


if __name__ == "__main__":
    main()
