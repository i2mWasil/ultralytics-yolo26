"""Which detection level (P3 / P4 / P5) do a model's false positives come from?

Why: the RGATE gates (head-only placement) change only the predictions of the levels they sit on: P3 in the p3
design (rgate/yolo26-rgate-p3.yaml), P3 and P4 in the p3p4 design (rgate/yolo26-rgate-p3p4.yaml). Phantoms predicted
from an ungated level (P5) cannot be removed by the gates. Run this on the baseline before any long training run.

How it works:
  1. Runs the model on every image of a dataset (sample format: data.yaml, images/, labels/).
  2. Keeps each kept box's anchor index. Anchor index -> level: with a 640x640 input, 0-6399 = P3, 6400-7999 = P4,
     8000-8399 = P5 (computed from the real feature map sizes, so other input sizes work too).
  3. Matches boxes to ground truth (same class, IoU >= 0.5, highest score first) and labels each box:
       tp      matches an object
       fp_obj  IoU >= 0.5 with some object, but not a match (duplicate or wrong class)
       fp_loc  0.1 <= best IoU with any object < 0.5 (near an object, bad box)
       fp_bg   best IoU with any object < 0.1 (away from every object: phantom candidates)
     With --regions (hand-marked reflective road areas), a phantom = a box with IoU < 0.5 to every object whose
     centre lies inside a marked area (rgate/CLAUDE.md section 7 definition).
  4. Prints a table per level and writes summary.json, boxes.csv and (optional) pictures.

Branch: by default the same one Ultralytics 8.4.212 uses for val/predict (nms=None): the one2many head + NMS.
--nms-free uses the one2one head instead (what you get with nms=False).

Works with stock checkpoints (official COCO yolo26n.pt or your BDD baseline) and RGATE checkpoints (needs this repo).
For a COCO model, class names are mapped to the dataset's names (bicycle->bike, motorcycle->motor,
stop sign->traffic sign); boxes of COCO classes with no match in the dataset are dropped and counted as "unmapped".

Examples (from the repo root):
  python rgate/phantom_levels.py --weights yolo26n.pt --data dataset/sampled_bdd100k_night_rain
  python rgate/phantom_levels.py --weights runs/rgate/base_n_bdd_train_night_s0/weights/best.pt \\
      --data ~/data/bdd_val_night_rain --regions ~/data/reflection_regions --save-vis 40
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if (REPO / "ultralytics" / "__init__.py").is_file():
    sys.path.insert(0, str(REPO))  # use this repo's patched Ultralytics (needed for RGATE checkpoints)
sys.path.insert(0, str(HERE))
import data_prep  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ALIASES = {"bicycle": "bike", "motorcycle": "motor", "stop sign": "traffic sign"}  # COCO name -> BDD name
STATUSES = ("tp", "fp_obj", "fp_loc", "fp_bg")
LEVEL_COLORS = {"P3": (0, 0, 255), "P4": (0, 200, 255), "P5": (255, 120, 0)}  # BGR: red, yellow, blue


def pick_device(req: str | None) -> str:
    if req:
        return req
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def class_map(model_names: dict, data_names: dict) -> dict[int, int]:
    """model class id -> dataset class id, by name (with COCO aliases)."""
    inv = {v: k for k, v in data_names.items()}
    out = {}
    for i, n in model_names.items():
        n2 = n if n in inv else ALIASES.get(n)
        if n2 in inv:
            out[int(i)] = inv[n2]
    return out


def read_boxes(txt: Path, w: int, h: int) -> tuple[torch.Tensor, torch.Tensor]:
    """YOLO txt ('cls cx cy w h' normalised; 'cx cy w h' also accepted) -> (xyxy pixels (N,4), cls (N,))."""
    rows = []
    if txt.is_file():
        for line in txt.read_text().splitlines():
            v = line.split()
            if len(v) in (4, 5):
                rows.append([float(x) for x in ([-1] + v if len(v) == 4 else v)])
    if not rows:
        return torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)
    a = torch.tensor(rows)
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), 1), a[:, 0].long()


def level_of(idx: torch.Tensor, sizes: list[int]) -> list[str]:
    edges = np.cumsum(sizes)
    return [f"P{3 + int(np.searchsorted(edges, int(i), side='right'))}" for i in idx]


@torch.inference_mode()
def predict(model, head, x, conf: float, iou: float, nms_free: bool):
    """Boxes (letterbox pixels, xyxy), scores, model classes and anchor indices of the kept detections (1 image)."""
    from ultralytics.utils.nms import non_max_suppression

    y, preds = model(x)
    branch = preds["one2one" if nms_free else "one2many"]
    sizes = [f.shape[2] * f.shape[3] for f in branch["feats"]]
    if nms_free:
        dec = head._inference(branch).permute(0, 2, 1)  # (1, A, 4+nc), xyxy because head.end2end is True
        sc, lab, idx = head.get_topk_index(dec[..., 4:4 + head.nc], getattr(head, "max_det", 300))
        boxes = dec[0, idx[0], :4]
        s, c, i = sc[0, :, 0], lab[0, :, 0].long(), idx[0]
        # self-check: same detections as the model's own end-to-end output
        if not torch.allclose(y[0, :, 4], s, atol=1e-5):
            raise RuntimeError("index recovery does not match the model's end-to-end output")
        keep = s > conf
        return boxes[keep], s[keep], c[keep], i[keep], sizes
    out, keepi = non_max_suppression(y, conf, iou, max_det=getattr(head, "max_det", 300), return_idxs=True)
    d = out[0]
    return d[:, :4], d[:, 4], d[:, 5].long(), keepi[0], sizes


def match(pb, ps, pc, gb, gc, thr: float = 0.5):
    """Greedy match by score. Returns (status per box, best IoU per box, number of matched objects)."""
    from ultralytics.utils.metrics import box_iou

    n = len(pb)
    if n == 0:
        return [], torch.zeros(0), 0
    if len(gb) == 0:
        return ["fp_bg"] * n, torch.zeros(n), 0
    iou = box_iou(pb, gb)
    best = iou.max(1).values
    used = torch.zeros(len(gb), dtype=torch.bool)
    status = [""] * n
    for i in ps.argsort(descending=True).tolist():
        cand = torch.where((gc == pc[i]) & ~used, iou[i], torch.tensor(-1.0))
        j = int(cand.argmax())
        if cand[j] >= thr:
            used[j] = True
            status[i] = "tp"
        elif best[i] >= thr:
            status[i] = "fp_obj"
        elif best[i] >= 0.1:
            status[i] = "fp_loc"
        else:
            status[i] = "fp_bg"
    return status, best, int(used.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="yolo26n.pt", help="checkpoint: official name, your baseline or an RGATE .pt")
    ap.add_argument("--data", required=True, help="dataset folder (with data.yaml) or a data .yaml; uses its val split")
    ap.add_argument("--regions", help="folder of hand-marked reflective areas: <image stem>.txt, YOLO box format")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="deployment confidence threshold (predict default)")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU (Ultralytics predict default)")
    ap.add_argument("--nms-free", action="store_true", help="use the one2one (NMS-free) head instead of one2many+NMS")
    ap.add_argument("--device", help="cuda:0 | mps | cpu; default: auto")
    ap.add_argument("--split", choices=["val", "train", "all"], default="val",
                    help="which images of --data: its val entry (default), its train entry, or both")
    ap.add_argument("--limit", type=int, help="only the first N images (quick look)")
    ap.add_argument("--save-vis", type=int, default=0, metavar="N",
                    help="save pictures of the N images with the most false positives (boxes coloured by level)")
    ap.add_argument("--out", help="output folder; default runs/rgate/phantom_levels/<weights>_<dataset>")
    a = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops

    device = pick_device(a.device)
    ycfg = data_prep.resolve_yaml(a.data)
    cfg = data_prep.load_cfg(ycfg)
    if a.split == "all":
        images = sorted(set(data_prep.list_images(cfg, "train")) | set(data_prep.list_images(cfg, "val")))
    else:
        images = data_prep.list_images(cfg, a.split)
    images = images[: a.limit]
    if not images:
        sys.exit(f"no images in {ycfg}")

    yolo = YOLO(a.weights)
    model = yolo.model.float().eval().to(device)
    head = model.model[-1]
    if getattr(head, "one2one_cv2", None) is None and a.nms_free:
        sys.exit("--nms-free: this model has no one2one head")
    if getattr(head, "one2one_cv2", None) is not None:
        head.end2end = a.nms_free
    cmap = class_map(model.names, cfg["names"])
    if not cmap:
        sys.exit(f"no class names in common between the model {list(model.names.values())[:5]}... and {ycfg}")
    gate_levels = sorted(m.level for m in model.modules() if type(m).__name__ == "ReflectionGate")
    gated = bool(gate_levels)
    stride = int(max(model.stride))
    letterbox = LetterBox((a.imgsz, a.imgsz), auto=True, stride=stride)  # same minimal padding as predict()

    out = Path(a.out or REPO / "runs/rgate/phantom_levels" / f"{Path(a.weights).stem}_{ycfg.parent.name}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    regions_dir = Path(a.regions).expanduser().resolve() if a.regions else None

    print(f"[levels] {a.weights} ({'RGATE, gates on ' + '+'.join(gate_levels) if gated else 'stock'}), branch: "
          f"{'one2one (NMS-free)' if a.nms_free else 'one2many + NMS (Ultralytics default)'}, "
          f"conf {a.conf}, iou {a.iou}, device {device}")
    print(f"[levels] {len(images)} images from {ycfg}")
    if len(cmap) < len(model.names):
        print(f"[levels] model has {len(model.names)} classes, {len(cmap)} map to the dataset's {len(cfg['names'])}; "
              "boxes of the others are dropped (counted as 'unmapped')")

    table = defaultdict(Counter)  # level -> counts
    per_class = defaultdict(Counter)  # level -> class name -> fp_bg count
    rows, vis_rank = [], []
    n_gt = n_matched = n_marked = 0
    done_images = []  # every image actually processed (also those without boxes), for per-image statistics
    sizes_seen = None
    for k, p in enumerate(images, 1):
        img0 = cv2.imread(str(p))
        if img0 is None:
            print(f"[levels] WARNING cannot read {p}, skipped")
            continue
        h0, w0 = img0.shape[:2]
        im = letterbox(image=img0)
        x = torch.from_numpy(np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))).float().div(255)[None].to(device)
        b, s, c, idx, sizes = predict(model, head, x, a.conf, a.iou, a.nms_free)
        sizes_seen = sizes_seen or sizes
        b = ops.scale_boxes(x.shape[2:], b.clone(), (h0, w0)).cpu()
        s, c, idx = s.cpu(), c.cpu(), idx.cpu()
        lv = level_of(idx, sizes)

        mapped = torch.tensor([int(ci) in cmap for ci in c], dtype=torch.bool)
        for li, m in zip(lv, mapped.tolist()):
            if not m:
                table[li]["unmapped"] += 1
        b, s, c, idx = b[mapped], s[mapped], c[mapped], idx[mapped]
        lv = [li for li, m in zip(lv, mapped.tolist()) if m]
        dc = torch.tensor([cmap[int(ci)] for ci in c], dtype=torch.long)

        gb, gc = read_boxes(data_prep.label_for(p), w0, h0)
        status, best, nm = match(b, s, dc, gb, gc)
        n_gt += len(gb)
        n_matched += nm

        rb = None
        if regions_dir:
            rf = regions_dir / f"{p.stem}.txt"
            if rf.is_file():
                n_marked += 1
                rb = read_boxes(rf, w0, h0)[0]
        n_fp = 0
        for i in range(len(b)):
            x1, y1, x2, y2 = b[i].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            st, li = status[i], lv[i]
            table[li]["boxes"] += 1
            table[li][st] += 1
            if st == "fp_bg":
                per_class[li][cfg["names"][int(dc[i])]] += 1
                if cy > h0 / 2:
                    table[li]["fp_bg_lower_half"] += 1
            in_region = None
            if rb is not None:
                in_region = bool(((rb[:, 0] <= cx) & (cx <= rb[:, 2]) & (rb[:, 1] <= cy) & (cy <= rb[:, 3])).any())
                if in_region and st in ("fp_loc", "fp_bg"):
                    table[li]["phantom"] += 1
            n_fp += st != "tp"
            rows.append({"image": p.name, "level": li, "anchor": int(idx[i]), "class": cfg["names"][int(dc[i])],
                         "conf": round(float(s[i]), 4), "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1),
                         "y2": round(y2, 1), "w": round(x2 - x1, 1), "h": round(y2 - y1, 1), "status": st,
                         "best_iou": round(float(best[i]), 3), "in_marked_region": in_region,
                         "lower_half": cy > h0 / 2})
        vis_rank.append((n_fp, p, b, lv, status, gb))
        done_images.append(p.name)
        if k % 200 == 0:
            print(f"[levels] {k}/{len(images)} images")

    # ---- report ----
    levels = [f"P{3 + i}" for i in range(len(sizes_seen or [0, 0, 0]))]
    cols = ["boxes", *STATUSES, "fp_bg_lower_half"] + (["phantom"] if regions_dir else []) + ["unmapped"]
    total = Counter()
    for li in levels:
        total.update(table[li])
    print(f"\n[levels] anchors per level at this input size: "
          f"{dict(zip(levels, sizes_seen))}  (index ranges follow this order)")
    print(f"{'level':<6}" + "".join(f"{c:>17}" for c in cols))
    for li in [*levels, "all"]:
        t = total if li == "all" else table[li]
        print(f"{li:<6}" + "".join(f"{t[c]:>17}" for c in cols))
    for key in ("fp_bg", "fp_loc", "phantom"):
        if total[key]:
            share = ", ".join(f"{li} {100 * table[li][key] / total[key]:.0f}%" for li in levels)
            print(f"[levels] share of {key} by level: {share}")
    print(f"[levels] objects: {n_gt}, matched: {n_matched} (recall at conf {a.conf}: "
          f"{n_matched / max(n_gt, 1):.3f})" + (f"; images with marked regions: {n_marked}" if regions_dir else ""))
    if total["fp_bg"]:
        for name, lv in (("P3 only", ["P3"]), ("P3 + P4", ["P3", "P4"])):
            share = sum(table[x]["fp_bg"] for x in lv) / total["fp_bg"]
            print(f"[levels] reading: {100 * share:.0f}% of background false positives come from {name}, "
                  f"the levels a {name} gate design can change.")
        if regions_dir is None:
            print("[levels] note: fp_bg is not the same as 'phantom from a reflection'. Look at the pictures "
                  "(--save-vis) or mark reflective areas (--regions) before deciding.")

    summary = {"weights": str(a.weights), "gated": gated, "gate_levels": gate_levels, "data": str(ycfg), "images": len(images),
               "branch": "one2one" if a.nms_free else "one2many+nms", "conf": a.conf, "iou": a.iou,
               "imgsz": a.imgsz, "anchors_per_level": dict(zip(levels, sizes_seen or [])),
               "per_level": {li: dict(table[li]) for li in levels}, "total": dict(total),
               "fp_bg_by_class": {li: dict(per_class[li]) for li in levels},
               "objects": n_gt, "matched": n_matched, "images_with_regions": n_marked if regions_dir else None}
    summary["split"] = a.split
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "images.txt").write_text("\n".join(done_images) + "\n")
    with open(out / "boxes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["image"])
        w.writeheader()
        w.writerows(rows)

    if a.save_vis:
        vdir = out / "vis"
        vdir.mkdir(exist_ok=True)
        for n_fp, p, b, lv, status, gb in sorted(vis_rank, key=lambda r: -r[0])[: a.save_vis]:
            if n_fp == 0:
                break
            img = cv2.imread(str(p))
            for g in gb.tolist():
                cv2.rectangle(img, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 255, 0), 1)
            for i, st in enumerate(status):
                if st == "tp":
                    continue
                x1, y1, x2, y2 = (int(v) for v in b[i].tolist())
                col = LEVEL_COLORS.get(lv[i], (255, 255, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                cv2.putText(img, f"{lv[i]} {st}", (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            cv2.imwrite(str(vdir / p.name), img)
        print(f"[levels] pictures: {vdir} (green = ground truth; false positives: red P3, yellow P4, blue P5)")
    print(f"[levels] outputs in {out}")


if __name__ == "__main__":
    main()
