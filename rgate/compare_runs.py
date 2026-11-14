"""Side-by-side comparison of the finished training pipelines.

  base  YOLO26n                          run name  base_<scale>_<train set>_s<seed>
  p3    YOLO26n + gate on P3             run name  gate-p3-l<lam>_<scale>_<train set>_s<seed>
  p3p4  YOLO26n + gates on P3 and P4     run name  gate-p3p4-l<lam>_<scale>_<train set>_s<seed>
The plan trains base and ONE gate pipeline (the one the level check chose); the report shows whichever are finished.

Reads each finished run's eval_summary.csv (written by train_rgate.py) and gamma.json. With --main-eval it also runs
rgate/phantom_levels.py with each run's best.pt on that eval set (once; results are reused) and reports the phantom
candidates per level. Writes a Markdown report and prints it. Runs that are not finished (no done.json) are skipped.

Example (repo root):
  python rgate/compare_runs.py --train-set bdd_train_night --main-eval ~/data/bdd_val_night_rain
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
METRICS = (("precision", "P"), ("recall", "R"), ("mAP50", "mAP50"), ("mAP50-95", "mAP50-95"))
LEVELS = ("P3", "P4", "P5")


def fmt(v, base=None, digits=3):
    if v is None:
        return "–"
    s = f"{v:.{digits}f}"
    if base is not None:
        s += f" ({v - base:+.{digits}f})"
    return s


def phantom_summary(run: Path, main_eval: str, device: str | None) -> dict | None:
    out = run / f"phantom_levels_{Path(main_eval).expanduser().resolve().name}"
    if not (out / "summary.json").is_file():
        cmd = [sys.executable, str(HERE / "phantom_levels.py"), "--weights", str(run / "weights" / "best.pt"),
               "--data", main_eval, "--save-vis", "20", "--out", str(out)]
        if device:
            cmd += ["--device", device]
        print("[compare] " + " ".join(cmd))
        if subprocess.run(cmd).returncode != 0:
            print(f"[compare] WARNING phantom check failed for {run.name}")
            return None
    return json.loads((out / "summary.json").read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="runs/rgate", help="folder that holds the runs (train_rgate.py --project)")
    ap.add_argument("--train-set", required=True, help="training dataset folder name, e.g. bdd_train_night")
    ap.add_argument("--scale", default="n")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", default="0.05", help="aux loss weight used by the gate pipelines (as in the run names)")
    ap.add_argument("--main-eval", help="main eval dataset (e.g. ~/data/bdd_val_night_rain): adds phantom counts")
    ap.add_argument("--device", help="device for the phantom check; default: auto")
    ap.add_argument("--out", help="Markdown report; default <project>/comparison_<train set>_s<seed>.md")
    a = ap.parse_args()

    project = Path(a.project).expanduser().resolve()
    lam = str(float(a.lam))
    pipes = [("base", f"base_{a.scale}_{a.train_set}_s{a.seed}"),
             ("p3", f"gate-p3-l{lam}_{a.scale}_{a.train_set}_s{a.seed}"),
             ("p3p4", f"gate-p3p4-l{lam}_{a.scale}_{a.train_set}_s{a.seed}")]
    runs = []
    for label, name in pipes:
        d = project / name
        if (d / "done.json").is_file():
            runs.append((label, d))
        else:
            print(f"[compare] {label}: {d} not finished (no done.json), skipped")
    if not runs:
        sys.exit("[compare] no finished runs found")

    evals = {}  # run label -> {eval set -> row}
    for label, d in runs:
        with open(d / "eval_summary.csv", newline="") as f:
            evals[label] = {r["eval_set"]: r for r in csv.DictReader(f)}
    sets = list(dict.fromkeys(s for r in evals.values() for s in r))
    base_label = runs[0][0] if runs[0][0] == "base" else None
    labels = [lab for lab, _ in runs]

    L = [f"# RGATE comparison — {a.train_set}, seed {a.seed}", "",
         "Numbers in brackets: difference to base. One seed: treat differences under ~0.01 mAP "
         "(1 point) or ~0.02 precision on rain-night as inconclusive (rgate/CLAUDE.md §7).", "",
         "## Detection accuracy per eval set (best.pt, Ultralytics val)", "",
         "| eval set | metric | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    for s in sets:
        for key, name in METRICS:
            vals = [float(evals[lab][s][key]) if s in evals[lab] else None for lab in labels]
            b = vals[0] if base_label else None
            cells = [fmt(v, b if (b is not None and i > 0 and v is not None) else None) for i, v in enumerate(vals)]
            L.append(f"| {s} | {name} | " + " | ".join(cells) + " |")

    if a.main_eval:
        ph = {lab: phantom_summary(d, a.main_eval, a.device) for lab, d in runs}
        L += ["", f"## Phantom candidates on {Path(a.main_eval).name} (conf 0.25, default branch one2many + NMS)", "",
              "fp_bg = kept box with IoU < 0.1 to every labelled object; road area = box centre in the lower half.", "",
              "| | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]

        def row(name, fn, digits=3):
            vals = [fn(ph[lab]) if ph.get(lab) else None for lab in labels]
            b = vals[0] if base_label else None
            cells = [fmt(v, b if (b is not None and i > 0 and v is not None) else None, digits) for i, v in enumerate(vals)]
            L.append(f"| {name} | " + " | ".join(cells) + " |")

        row("recall at conf 0.25", lambda s: s["matched"] / max(s["objects"], 1))
        row("fp_bg per image", lambda s: s["total"].get("fp_bg", 0) / max(s["images"], 1))
        row("road-area fp_bg per image", lambda s: s["total"].get("fp_bg_lower_half", 0) / max(s["images"], 1))
        for lv in LEVELS:
            row(f"road-area fp_bg from {lv} (count)",
                lambda s, lv=lv: float(s["per_level"].get(lv, {}).get("fp_bg_lower_half", 0)), 0)

    L += ["", "## Gate strength at the last epoch (tanh(10·γ); positive = suppress, negative = boost)", "",
          "| pipeline | " + " | ".join(LEVELS[:2]) + " |", "|---|---|---|"]
    for lab, d in runs:
        g = d / "gamma.json"
        if g.is_file():
            last = json.loads(g.read_text())[-1]
            L.append(f"| {lab} | " + " | ".join(f"{last[lv]['strength']:+.4f}" if lv in last else "–"
                                                 for lv in LEVELS[:2]) + " |")
        else:
            L.append(f"| {lab} | – | – |")

    text = "\n".join(L) + "\n"
    out = Path(a.out).expanduser() if a.out else project / f"comparison_{a.train_set}_s{a.seed}.md"
    out.write_text(text)
    print(text)
    print(f"[compare] written: {out}")


if __name__ == "__main__":
    main()
