"""Level check BEFORE training: which feature level do the phantom candidates come from? -> P3 or P3 + P4 gates.

Runs rgate/phantom_levels.py over a whole dataset (default: every image of the night training set) and applies a
decision rule that is FIXED IN ADVANCE (decided 2026-11-15, rgate/CLAUDE.md decision #21). Do not change the rule
after seeing the numbers.

  phantom candidate = a kept box (conf >= 0.25) whose best IoU with every labelled object is < 0.1 ("fp_bg")
  road area         = box centre in the lower half of the image (where wet-road reflections appear)
  score             = share of road-area phantom candidates that come from P4 = P4 / (P3 + P4 + P5)

  score >= 0.25  -> p3p4: gates on P3 and P4   (run_all.sh p3p4, --gate-levels p3p4, yolo26-rgate-p3p4.yaml)
  score <  0.25  -> p3:   gate on P3 only      (run_all.sh p3,   --gate-levels p3,   yolo26-rgate-p3.yaml)
  fewer than 100 road-area candidates -> "inconclusive": too little evidence -> p3p4

Why 0.25: the P4 gate costs about +2% compute. If at least a quarter of the road-area phantom candidates come from
P4, a P3-only gate would leave too many untouched. A 95% bootstrap interval (resampling images) is reported; if it
contains 0.25 the result is flagged as a close call (the rule still decides).

Why the training set: the decision must not look at the eval sets (rain-night val etc.), or the design would be tuned
on the data used to judge it. Default model: the official COCO yolo26n.pt (no BDD training needed yet). It is not
trained on BDD100K, so its false positives are only a stand-in for those of the trained models; say so in the thesis.

Example (repo root):
  python rgate/level_check.py --data ~/data/bdd_train_night
Outputs in runs/rgate/level_check/<weights>_<dataset>/: decision.json, summary.json, boxes.csv, images.txt, vis/
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import data_prep  # noqa: E402

LEVELS = ("P3", "P4", "P5")


def run_levels(a, out: Path):
    cmd = [sys.executable, str(HERE / "phantom_levels.py"), "--weights", a.weights, "--data", a.data,
           "--split", a.split, "--conf", str(a.conf), "--imgsz", str(a.imgsz), "--save-vis", str(a.save_vis),
           "--out", str(out)]
    if a.device:
        cmd += ["--device", a.device]
    if a.limit:
        cmd += ["--limit", str(a.limit)]
    print("[check] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def needs_run(a, out: Path, ycfg: Path) -> bool:
    s = out / "summary.json"
    if a.rerun or not s.is_file() or not (out / "images.txt").is_file():
        return True
    old = json.loads(s.read_text())
    same = (old.get("weights") == a.weights and old.get("data") == str(ycfg) and old.get("split") == a.split
            and old.get("conf") == a.conf and old.get("imgsz") == a.imgsz)
    if not same:
        print("[check] existing results were made with other settings; running again")
    return not same


def count_per_image(out: Path, region: str) -> tuple[np.ndarray, list[str]]:
    images = [x for x in (out / "images.txt").read_text().splitlines() if x]
    idx = {name: i for i, name in enumerate(images)}
    counts = np.zeros((len(images), len(LEVELS)), dtype=np.int64)
    with open(out / "boxes.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "fp_bg" or r.get("level") not in LEVELS:
                continue
            if region == "road" and r.get("lower_half") != "True":
                continue
            counts[idx[r["image"]], LEVELS.index(r["level"])] += 1
    return counts, images


def share_p4(c: np.ndarray) -> float:
    s = c.sum()
    return float(c[1] / s) if s else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset folder or data.yaml (default use: the night training set)")
    ap.add_argument("--weights", default="yolo26n.pt", help="model for the check (default: official COCO yolo26n.pt)")
    ap.add_argument("--split", choices=["val", "train", "all"], default="all", help="images of --data to use")
    ap.add_argument("--threshold", type=float, default=0.25, help="fixed rule: P4 share >= this -> P3 + P4")
    ap.add_argument("--min-count", type=int, default=100, help="fewer road-area candidates -> inconclusive")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", help="cuda:0 | mps | cpu; default: auto")
    ap.add_argument("--save-vis", type=int, default=40, help="pictures of the N images with most false positives")
    ap.add_argument("--bootstrap", type=int, default=2000, help="resamples for the 95%% interval")
    ap.add_argument("--limit", type=int, help="only the first N images (testing only; never for the decision)")
    ap.add_argument("--rerun", action="store_true", help="run the detector again even if results exist")
    ap.add_argument("--out", help="output folder; default runs/rgate/level_check/<weights>_<dataset>")
    a = ap.parse_args()

    ycfg = data_prep.resolve_yaml(a.data)
    out = Path(a.out or REPO / "runs/rgate/level_check" / f"{Path(a.weights).stem}_{ycfg.parent.name}").resolve()
    if needs_run(a, out, ycfg):
        run_levels(a, out)

    counts, images = count_per_image(out, "road")
    counts_all, _ = count_per_image(out, "all")
    tot, tot_all = counts.sum(0), counts_all.sum(0)
    n = int(tot.sum())
    score = share_p4(tot)
    rng = np.random.default_rng(0)
    boots = [share_p4(counts[rng.integers(0, len(images), len(images))].sum(0)) for _ in range(a.bootstrap)]
    lo, hi = (float(x) for x in np.nanpercentile(boots, [2.5, 97.5])) if n else (float("nan"), float("nan"))

    warnings = []
    if n < a.min_count:
        decision, reason = "p3p4", (f"inconclusive: only {n} road-area phantom candidates (< {a.min_count}); "
                                    "using p3p4")
    elif score >= a.threshold:
        decision, reason = "p3p4", f"P4 share {score:.3f} >= {a.threshold}"
    else:
        decision, reason = "p3", f"P4 share {score:.3f} < {a.threshold}"
    if n >= a.min_count and lo < a.threshold < hi:
        warnings.append(f"close call: the 95% interval [{lo:.3f}, {hi:.3f}] contains the threshold {a.threshold}")
    if n and tot[2] / n > 0.3:
        warnings.append(f"{tot[2] / n:.0%} of road-area candidates come from P5, which neither design gates")
    if a.limit:
        warnings.append(f"--limit {a.limit} was used: test only, not a valid decision")
    summ = json.loads((out / "summary.json").read_text())

    result = {
        "decision": decision,
        "pipeline": {"p3": "p3: YOLO26 + gate on P3", "p3p4": "p3p4: YOLO26 + gates on P3 and P4"}[decision],
        "next": f"bash rgate/run_all.sh base   (if not done), then   bash rgate/run_all.sh {decision}",
        "reason": reason, "rule": f"P4 share of road-area fp_bg >= {a.threshold} -> p3p4, else p3; "
                                  f"< {a.min_count} candidates -> inconclusive -> p3p4",
        "score_p4_share": score, "ci95": [lo, hi], "warnings": warnings,
        "road_area_fp_bg": dict(zip(LEVELS, map(int, tot))), "all_fp_bg": dict(zip(LEVELS, map(int, tot_all))),
        "images": len(images), "weights": a.weights, "data": str(ycfg), "split": a.split, "conf": a.conf,
        "imgsz": a.imgsz, "branch": summ.get("branch"), "objects": summ.get("objects"), "matched": summ.get("matched"),
        "outputs": str(out),
    }
    (out / "decision.json").write_text(json.dumps(result, indent=1))

    print(f"\n[check] {len(images)} images, model {a.weights}, branch {summ.get('branch')}")
    print(f"[check] phantom candidates (fp_bg) by level   road area: "
          + ", ".join(f"{k} {v}" for k, v in result["road_area_fp_bg"].items())
          + "   | whole image: " + ", ".join(f"{k} {v}" for k, v in result["all_fp_bg"].items()))
    print(f"[check] score = P4 share of road-area candidates = {score:.3f}  (95% interval {lo:.3f}-{hi:.3f})")
    print(f"[check] DECISION: {decision}  -> {result['pipeline']}   ({reason})")
    print(f"[check] next: {result['next']}")
    for w in warnings:
        print(f"[check] NOTE: {w}")
    print(f"[check] written: {out / 'decision.json'}   pictures: {out / 'vis'}")


if __name__ == "__main__":
    main()
