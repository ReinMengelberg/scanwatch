"""labels -> datasets/scancar_cls/{train,val}/{scancar,other}/ for yolo11n-cls.

Split by event (one pass of a car), never by frame, so near-identical crops of one pass never sit on
both sides. hard_neg counts as other and is always kept; plain negatives can be capped per positive in
train (split.neg_cap, 0 = all). Positives are repeated in train until the classes are balanced: there are far
fewer scan cars than other cars, and yolo-cls has no class weights. Files are symlinks to data/raw.

  python build_dataset.py
  python build_dataset.py --all    every event in train (val = the same files, only to monitor training):
                                   for the model that gets deployed, when there are too few scan cars to hold one out
"""
import argparse
import shutil
from collections import Counter

from scancar.common import CLASSES, DATASET, ROOT, assign_split, cfg, labelled_crops, rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="no held-out events: train on everything")
    a = ap.parse_args()
    c = cfg()["split"]
    rows = labelled_crops()
    split = assign_split(rows, 0 if a.all else None)
    shutil.rmtree(DATASET, ignore_errors=True)
    for s in ("train", "val"):
        for cls in CLASSES:
            (DATASET / s / cls).mkdir(parents=True)

    by = {(s, k): [r for r in rows if split[r["event_id"]] == s and r["label"] == k]
          for s in ("train", "val") for k in ("scancar", "other", "hard_neg")}
    pos = by["train", "scancar"]
    neg = by["train", "other"]
    rng(c["seed"]).shuffle(neg)
    if c["neg_cap"]:
        neg = neg[:max(1, len(pos)) * c["neg_cap"]]
    neg += by["train", "hard_neg"]
    reps = max(1, round(len(neg) / max(1, len(pos))))

    counts = Counter()
    val_pos = pos if a.all else by["val", "scancar"]
    val_neg = neg if a.all else by["val", "other"] + by["val", "hard_neg"]
    for s, cls, items, n in (("train", "scancar", pos, reps), ("train", "other", neg, 1),
                             ("val", "scancar", val_pos, 1), ("val", "other", val_neg, 1)):
        for r in items:
            src = (ROOT / r["crop"]).resolve()
            name = r["crop"].replace("/", "__")
            for i in range(n):
                (DATASET / s / cls / (f"{i}_{name}" if n > 1 else name)).symlink_to(src)
                counts[s, cls] += 1
    events = Counter((split[e], lab) for e, lab in {(r["event_id"], r["label"]) for r in rows})
    for s in ("train", "val"):
        print(f"{s:5s} scancar {counts[s, 'scancar']:4d} files ({events[s, 'scancar']} events)   "
              f"other {counts[s, 'other']:4d} files ({events[s, 'other'] + events[s, 'hard_neg']} events)")
    if reps > 1:
        print(f"train positives repeated x{reps}")
    print(f"-> {DATASET}")


if __name__ == "__main__":
    main()
