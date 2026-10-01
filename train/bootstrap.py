"""Turn the pre-sorted full frames (../datasets/{scancar,othercar}) into labelled crops.

Per frame: run the Pi's detector, crop every vehicle with crop.py. pHash near-duplicates are not\nstored again; their box is indexed with dup_of=<crop> so the frame view still shows it.
Auto labels (source=auto, fix them in label.py):
  - boxes at the same spot in most frames (parked cars)  -> other
  - frames from an 'other' folder                         -> other
  - frames from a 'scancar' folder: biggest moving box    -> scancar, the rest -> other,
    except boxes whose crop shows the scan car's pod       -> skip (pod fragments)
Re-runnable: frames already in data/index.csv are skipped.
"""
import csv
import os
from datetime import datetime

import cv2

from scancar.common import ROOT, RAW, Deduper, append_index, append_labels, cfg, detect, iou, read_index
from scancar.crop import crop, pad_box


def load_frames(c):
    """[(path, label, ts|None)] with timestamps from an index.csv next to the frames, if any."""
    out = []
    for src in c["bootstrap"]["sources"]:
        d = (ROOT / src["dir"]).resolve()
        times = {}
        if (d / "index.csv").exists():
            with (d / "index.csv").open() as f:
                times = {r["file"]: r["time"] for r in csv.DictReader(f)}
        for p in sorted(d.glob("*.jpg")):
            out.append((p, src["label"], times.get(p.name)))
    return out


def event_ids(frames, c):
    """Timestamped frames: new event after a gap. Others: config mapping, else one event per file."""
    ev, last_t, n = {}, None, 0
    for p, label, ts in sorted((f for f in frames if f[2]), key=lambda f: f[2]):
        t = datetime.fromisoformat(ts)
        if last_t is None or (t - last_t).total_seconds() > c["bootstrap"]["event_gap_s"]:
            n += 1
        last_t = t
        ev[p] = f"{label}-{t:%Y%m%d}-{n:04d}"
    for p, label, ts in frames:
        if p not in ev:
            ev[p] = next((e for k, e in c["bootstrap"]["events"].items() if p.name.startswith(k)), f"{label}-{p.stem}")
    return ev


def main():
    c = cfg()
    W, H = c["frame_size"]
    done = {r["frame"] for r in read_index()}
    frames = load_frames(c)
    ev = event_ids(frames, c)

    imgs, dets = {}, {}
    for p, _, _ in frames:
        im = cv2.imread(str(p))
        if im.shape[1] != W or im.shape[0] != H:
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        imgs[p], dets[p] = im, detect(im)

    # parked cars: same box in >= static_frac of the other frames
    n = len(frames)
    def is_static(p, b):
        hits = sum(any(iou(b, o) >= c["bootstrap"]["static_iou"] for o, _ in dets[q]) for q in dets if q != p)
        return hits >= c["bootstrap"]["static_frac"] * (n - 1)

    def shows_pod(b, scan):
        """Does b's crop contain the top 30% of the scan car (where the pod is)?"""
        x1, y1, x2, y2 = pad_box(b, W, H)
        sx1, sy1, sx2, sy2 = scan
        return min(x2, sx2) > max(x1, sx1) and min(y2, sy1 + 0.3 * (sy2 - sy1)) > max(y1, sy1)

    dd = Deduper()
    index_rows, label_rows, dups = [], [], 0
    for p, label, ts in frames:
        rel = os.path.relpath(p, ROOT)
        if rel in done:
            continue
        moving = [b for b, _ in dets[p] if not is_static(p, b)]
        for i, (b, conf) in enumerate(dets[p]):
            static = b not in moving
            lab = "scancar" if label == "scancar" and moving and b == moving[0] else "other"
            if label == "scancar" and lab == "other" and moving and shows_pod(b, moving[0]):
                lab = "skip"  # fragment of the scan car (e.g. the pod alone) or its crop shows the pod
            # track ids (no tracker here): a pass = one track; parked cars one per event;
            # other movers one track per crop (conservative: more chances for a false alarm)
            if static:
                tid = f"{ev[p]}-parked-{round(b[0] / 50)}-{round(b[1] / 50)}"
            elif lab == "scancar":
                tid = f"{ev[p]}-scancar"
            else:
                tid = f"{ev[p]}-{p.stem}-{i}"
            im = crop(imgs[p], b)
            dup_of, ph = dd.check(im)
            row = dict(crop="", track_id=tid, event_id=ev[p], ts=ts or "", source="bootstrap", synthetic=0, frame=rel,
                       x1=round(b[0]), y1=round(b[1]), x2=round(b[2]), y2=round(b[3]), conf=round(conf, 3),
                       phash=ph, dup_of=dup_of or "")
            if dup_of:
                dups += 1
            else:
                out = RAW / ev[p] / f"{p.stem}_{i}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
                row["crop"] = str(out.relative_to(ROOT))
                dd.add(ph, row["crop"])
                label_rows.append(dict(crop=row["crop"], track_id=tid, event_id=ev[p], label=lab, source="auto"))
            index_rows.append(row)
        if not dets[p]:
            # keep the frame in the index so a re-run skips it
            index_rows.append(dict(crop="", track_id="", event_id=ev[p], ts=ts or "", source="bootstrap",
                                   synthetic=0, frame=rel))

    append_index(index_rows)
    append_labels(label_rows)
    per = {}
    for r in label_rows:
        per[r["label"]] = per.get(r["label"], 0) + 1
    print(f"{len(frames)} frames, {len(label_rows)} new crops {per}, {dups} duplicates dropped -> {RAW}")


if __name__ == "__main__":
    main()
