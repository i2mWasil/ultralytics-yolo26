"""Train and evaluate one experiment arm. Same script for every arm, every dataset size, every machine.

Run from the repo root (or anywhere; paths are resolved). Examples:

  # 0. check data only (fast, no training)
  python rgate/train_rgate.py --data dataset/sampled_bdd100k --check-only

  # 1. smoke test on the 150-image sample (Mac / CPU, minutes)
  python rgate/train_rgate.py --arm gate --data dataset/sampled_bdd100k \
      --eval dataset/sampled_bdd100k_night dataset/sampled_bdd100k_night_rain --smoke

  # 2. real run on the GPU PC: same command, full folders, no --smoke (plan: rgate/CLAUDE.md section 7)
  python rgate/train_rgate.py --arm gate --lam 0.05 --data ~/data/bdd_train_night \
      --eval ~/data/bdd_val_night_rain ~/data/bdd_val_night ~/data/bdd_val_all --epochs 50 --batch 32

  # 3. continue an interrupted run: the SAME command plus --resume <run>/weights/last.pt
  python rgate/train_rgate.py ...same flags... --resume runs/rgate/<run>/weights/last.pt

Every dataset (train and eval) must be in the format of dataset/sampled_* (see data_prep.py and rgate/CLAUDE.md §6b).
Arms: --arm baseline | --arm gate [--lam 0 | --lam 0.05 | --no-prior]
Gate placement (required for gate arms): --gate-levels p3    (gate on P3,        rgate/yolo26-rgate-p3.yaml)
                                         --gate-levels p3p4  (gates on P3 and P4, rgate/yolo26-rgate-p3p4.yaml)
Run names: base_<scale>_<data>_s<seed>, gate-p3-l<lam>_..., gate-p3p4-l<lam>_... (+ "-noprior" after l<lam>).
Outputs: <project>/<name>/ (Ultralytics weights + results.csv) plus run_config.json, gamma.json, eval_summary.csv,
weights/best_ep<NNN>_*.pt (a kept copy of every new best) and done.json (written only when everything finished).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Apple GPU: fall back to CPU for any missing op

try:
    import torch  # noqa: E402
    import yaml  # noqa: E402
except ImportError as e:
    sys.exit(f"[rgate] missing package: {e.name}. From the repo root run: pip install -e . -r rgate/requirements.txt")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if (REPO / "ultralytics" / "__init__.py").is_file():
    sys.path.insert(0, str(REPO))  # use this repo's patched Ultralytics, not a pip-installed copy
sys.path.insert(0, str(HERE))
import data_prep  # noqa: E402


def check_patch():
    import ultralytics
    from ultralytics.nn import tasks

    src = Path(tasks.__file__).read_text()
    if "RGATE" not in src or not Path(ultralytics.__file__).parent.joinpath("nn/modules/rgate.py").is_file():
        sys.exit(f"[rgate] Ultralytics at {Path(ultralytics.__file__).parent} is not patched for RGATE "
                 "(see rgate/CLAUDE.md, section 'Setup').")
    print(f"[rgate] using Ultralytics {ultralytics.__version__} from {Path(ultralytics.__file__).parent}")

HYP = dict(  # identical for every arm. Change here only, never per arm.
    hsv_h=0.015, hsv_s=0.3, hsv_v=0.2,  # lowered from 0.7/0.4: the gate's prior reads brightness/saturation
    # Fixed optimizer = what optimizer=auto picks for the full night run (> 10k iterations). Without this, small runs
    # (smoke, pilot, samples) silently got AdamW and the full run MuSGD, so small runs did not test the real setup.
    optimizer="MuSGD", lr0=0.01, momentum=0.9, warmup_bias_lr=0.0,
    # No class-row copy by name (decision log #20): the COCO head (80 classes) cannot be reused for the 10 BDD classes,
    # so EVERY arm starts with a fresh class-output layer (hidden layers of the class branch still transfer).
    # Ultralytics' default (True) copies rows for matching names (car, person, ...); it reached the baseline but not
    # the gate arm, which made the comparison unfair.
    cls_remap=False,
    close_mosaic=10, deterministic=True, patience=100,
)
SMOKE = dict(epochs=1, imgsz=320, batch=4, workers=0)  # only proves the pipeline runs


def pick_device(req: str | None) -> str:
    if req:
        return req
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def has_gate(model) -> bool:
    return any(type(m).__name__ == "ReflectionGate" for m in model.modules())


def gates_of(model) -> list:
    return [m for m in model.modules() if type(m).__name__ == "ReflectionGate"]


def levels_of(model) -> str:
    """Gated levels of a model as used by --gate-levels, e.g. 'p3p4' or 'p3' ('' = no gate)."""
    return "".join(sorted(g.level.lower() for g in gates_of(model)))


GATE_YAML = {"p3": "yolo26-rgate-p3.yaml", "p3p4": "yolo26-rgate-p3p4.yaml"}
GATE_DEFAULTS = [None, 0.6, 8, 10.0, True]  # ReflectionGate args after c1: hidden, v0, stride, gamma_gain, use_prior


def file_sha1(p: Path) -> str:
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()


def keep_every_best(tr):
    """on_model_save: when this epoch made a new best.pt, keep a copy that is never overwritten.

    Same test Ultralytics uses to write best.pt. The copy has the optimizer stripped (FP16, ~1/3 the size),
    so it can be evaluated or exported but not resumed from (resume always uses last.pt).
    """
    if tr.best_fitness != tr.fitness or not tr.best.exists():
        return
    from ultralytics.utils.torch_utils import strip_optimizer

    m = tr.metrics or {}
    dst = tr.wdir / (f"best_ep{tr.epoch + 1:03d}_map50-95-{m.get('metrics/mAP50-95(B)', 0):.4f}"
                     f"_map50-{m.get('metrics/mAP50(B)', 0):.4f}.pt")
    strip_optimizer(tr.best, s=str(dst))


def gamma_logger(start_log: list):
    """on_train_epoch_end: append every gate's gamma to gamma.json (gate arms only).

    One row per epoch: {"epoch": n, "P3": {"gamma_eff": .., "strength": ..}, "P4": {...}}.
    gamma_eff = gain * gamma (the raw value); strength = tanh(gamma_eff), the factor actually applied:
    F_out = F * (1 - strength * S). Positive = suppress, negative = boost.
    """
    log = list(start_log)

    def cb(tr):
        row = {"epoch": tr.epoch + 1}
        for g in gates_of(tr.model):
            row[g.level] = {"gamma_eff": g.gamma_gain * float(g.gamma.detach()), "strength": float(g.strength().detach())}
        log[:] = [r for r in log if r["epoch"] < tr.epoch + 1]  # a resumed or repeated epoch replaces its old row
        log.append(row)
        Path(tr.save_dir, "gamma.json").write_text(json.dumps(log, indent=1))

    return cb


def gate_cfg(scale: str, lam: float, no_prior: bool, levels: str, out_dir: Path) -> Path:
    cfg = yaml.safe_load((HERE / GATE_YAML[levels]).read_text())
    cfg["rgate_lambda"] = lam
    if no_prior:  # every gate: keep its own args (e.g. stride 16 for P4), only switch the prior off
        for row in (r for r in cfg["head"] if r[2] == "ReflectionGate"):
            args = list(row[3]) + GATE_DEFAULTS[len(row[3]):]
            args[4] = False
            row[3] = args
    tag = f"{levels}-l{lam}" + ("-noprior" if no_prior else "")
    p = out_dir / f"yolo26{scale}-rgate-{tag}.yaml"  # file name must start with yolo26<scale> (Ultralytics reads scale from it)
    out_dir.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def remap_into_gate(stock, cfg_path: Path):
    """Stock YOLO26 weights -> RGATE graph. Layers 0-22 keep their index; Detect moves from 23 to the last index
    (24 with one gate, 25 with the P3+P4 gates); the gate layers in between start fresh."""
    from ultralytics.nn.tasks import DetectionModel

    stock = stock.float()
    new = DetectionModel(str(cfg_path), nc=len(stock.names), verbose=False)
    od, nd = len(stock.model) - 1, len(new.model) - 1
    sd = {(k.replace(f"model.{od}.", f"model.{nd}.", 1) if k.startswith(f"model.{od}.") else k): v
          for k, v in stock.state_dict().items()}
    gate_idx = [i for i, m in enumerate(new.model) if type(m).__name__ == "ReflectionGate"]
    if gate_idx != list(range(od, nd)):
        raise RuntimeError(f"Expected the gate layers at {list(range(od, nd))} (between P5 and Detect), got {gate_idx}")
    missing, unexpected = new.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not any(k.startswith(f"model.{i}.") for i in gate_idx)]
    if bad or unexpected:
        raise RuntimeError(f"Weight remap mismatch. missing={bad[:5]} unexpected={unexpected[:5]}")
    new.names = stock.names
    print(f"[rgate] remapped {len(sd)} stock tensors into the gate graph; {len(missing)} gate tensors start fresh")
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["baseline", "gate"], default="gate")
    ap.add_argument("--data", required=True, help="training dataset folder (with data.yaml) or a data .yaml")
    ap.add_argument("--val", help="optional separate validation dataset; default: auto split of --data")
    ap.add_argument("--val-frac", type=float, default=0.1, help="share of --data used for validation when auto-splitting")
    ap.add_argument("--eval", nargs="*", default=[], help="extra datasets to evaluate best.pt on after training")
    ap.add_argument("--weights", default="yolo26n.pt", help="start checkpoint: official name or your .pt (stock or RGATE)")
    ap.add_argument("--scale", help="n/s/m/l/x; default: read from --weights")
    ap.add_argument("--lam", type=float, default=0.0, help="aux gate loss weight (0 = A1, 0.05 = A2)")
    ap.add_argument("--no-prior", action="store_true", help="control arm A3: gate without the reflection prior")
    ap.add_argument("--gate-levels", choices=sorted(GATE_YAML), help="required for --arm gate: p3 (gate on P3) or "
                    "p3p4 (gates on P3 and P4); choose it from the level check (rgate/level_check.py)")
    # epochs/imgsz/batch/workers: defaults 50 / 640 / 16 / min(8, CPUs) for a new run. On --resume only values
    # typed on the command line are passed on (Ultralytics keeps the rest from last.pt; epochs cannot change).
    ap.add_argument("--epochs", type=int, help="default 50")
    ap.add_argument("--imgsz", type=int, help="default 640")
    ap.add_argument("--batch", type=int, help="default 16; -1 = Ultralytics auto batch (CUDA only)")
    ap.add_argument("--workers", type=int, help="default min(8, CPU count)")
    ap.add_argument("--device", help="0 | 0,1 | mps | cpu; default: auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="runs/rgate")
    ap.add_argument("--name", help="run name; default built from arm/scale/lam/seed")
    ap.add_argument("--smoke", action="store_true", help=f"tiny settings {SMOKE} to test the pipeline")
    ap.add_argument("--check-only", action="store_true", help="check datasets and write the split, then stop")
    ap.add_argument("--allow-overlap", action="store_true", help="allow eval images that are also in training")
    ap.add_argument("--resume", metavar="LAST_PT", help="continue the run that saved this last.pt (same other flags)")
    a = ap.parse_args()

    if a.arm == "baseline" and (a.lam or a.no_prior):
        ap.error("--lam/--no-prior only apply to --arm gate")
    if a.no_prior and a.lam:
        ap.error("--no-prior needs --lam 0 (the aux loss uses the prior)")
    if a.arm == "baseline" and a.gate_levels:
        ap.error("--gate-levels only applies to --arm gate")
    if a.arm == "gate" and not a.gate_levels and not a.check_only:
        ap.error("--arm gate needs --gate-levels p3 or p3p4 (the level check says which: rgate/PIPELINE.md)")
    levels = a.gate_levels
    run = dict(epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers)
    if a.smoke:
        run.update(SMOKE)

    project = Path(a.project).expanduser().resolve()
    tag = "base" if a.arm == "baseline" else f"gate-{levels}-l{a.lam}" + ("-noprior" if a.no_prior else "")
    data_tag = Path(a.data).resolve().name.replace(".yaml", "")

    # ---------- data ----------
    split_dir = project / "splits" / data_tag
    data_yaml, info = data_prep.build_train_yaml(a.data, a.val, split_dir, a.val_frac)
    print(f"[rgate] data: {info['mode']}; train {info['train']['images']} imgs / {info['train']['boxes']} boxes, "
          f"val {info['val']['images']} imgs / {info['val']['boxes']} boxes -> {data_yaml}")
    for k in ("train", "val"):
        if info[k]["missing_labels"]:
            print(f"[rgate] WARNING {k}: {info[k]['missing_labels']} images have no label file "
                  f"(treated as empty background), e.g. {info[k]['missing_examples']}")
    tcfg = data_prep.load_cfg(data_yaml)
    eval_info = {}
    for e in a.eval:
        ecfg = data_prep.load_cfg(data_prep.resolve_yaml(e))
        data_prep.same_names(tcfg, ecfg, e)
        n_ov = data_prep.overlap(data_yaml, e)
        eval_info[e] = {"overlap_with_train": n_ov, **data_prep.check_dataset(ecfg, data_prep.list_images(ecfg, "val"), e)}
        msg = f"[rgate] eval set {e}: {eval_info[e]['images']} imgs, {n_ov} also in training"
        if n_ov and not (a.smoke or a.allow_overlap):
            sys.exit(msg + " -> STOP: eval images leak into training. Fix the split, or for a test run on "
                     "the samples only pass --allow-overlap (Colab: EXTRA = \"--allow-overlap\" in cell 1).")
        print(msg + (" (allowed: smoke/--allow-overlap)" if n_ov else ""))
    if a.check_only:
        print(json.dumps({"train_data": info, "eval": eval_info}, indent=1, default=str))
        return

    # ---------- model ----------
    check_patch()
    from ultralytics import YOLO, __version__
    from ultralytics.nn.tasks import torch_safe_load

    device = pick_device(a.device)
    split_hash = {k: file_sha1(split_dir / f"{k}.txt") for k in ("train", "val")}
    versions = {"ultralytics": __version__, "torch": torch.__version__, "python": platform.python_version()}
    callbacks = {"on_model_save": [keep_every_best]}

    if a.resume:
        # ----- continue an existing run -----
        last = Path(a.resume).expanduser().resolve()
        if not last.is_file():
            sys.exit(f"[rgate] --resume file not found: {last}")
        save_dir = last.parent.parent
        ckpt = torch_safe_load(str(last))[0]
        ck_model = ckpt.get("ema") or ckpt["model"]
        ck_args = ckpt.get("train_args") or {}
        is_gate = has_gate(ck_model)
        if is_gate != (a.arm == "gate"):
            sys.exit(f"[rgate] --arm {a.arm} does not match {last} (it is a {'gate' if is_gate else 'baseline'} run)")
        if is_gate:
            ck_gates = gates_of(ck_model)
            ck_lam = float((getattr(ck_model, "yaml", {}) or {}).get("rgate_lambda", 0.0))
            ck_prior = {g.use_prior for g in ck_gates}
            if ck_lam != a.lam or ck_prior != {not a.no_prior}:
                sys.exit(f"[rgate] --lam {a.lam}{' --no-prior' if a.no_prior else ''} does not match {last} "
                         f"(lam {ck_lam}{'' if ck_prior == {True} else ', no prior'})")
            if levels_of(ck_model) != levels:
                sys.exit(f"[rgate] --gate-levels {levels} does not match {last} (gates on {levels_of(ck_model)})")
        old_cfg = save_dir / "run_config.json"
        if old_cfg.is_file():
            old_hash = json.loads(old_cfg.read_text()).get("split_sha1")
            if old_hash and old_hash != split_hash:
                sys.exit("[rgate] the train/val split is not the same as when this run started (the dataset changed). "
                         "Resuming would train on different data. Point --data at the original dataset.")
        else:
            print(f"[rgate] WARNING no run_config.json in {save_dir}; cannot confirm the split is unchanged")
        imgsz = run["imgsz"] or ck_args.get("imgsz", 640)
        batch = run["batch"] or ck_args.get("batch", 16)
        config = {"args": vars(a), "resumed_from": str(last), "start_epoch": ckpt.get("epoch", -1) + 2,
                  "device": device, "split_sha1": split_hash, "data": info, "eval": eval_info, "versions": versions}

        if ckpt.get("epoch", -1) == -1:
            print(f"[rgate] training in {save_dir} already finished; running only the evaluation")
        else:
            print(f"[rgate] resuming {save_dir.name} from epoch {ckpt['epoch'] + 2}")
            gamma_file = save_dir / "gamma.json"
            if is_gate:
                callbacks["on_train_epoch_end"] = [gamma_logger(
                    json.loads(gamma_file.read_text()) if gamma_file.is_file() else [])]
            callbacks["on_train_start"] = [lambda tr: Path(tr.save_dir, "run_config_resume.json").write_text(
                json.dumps(config, indent=1, default=str))]
            model = YOLO(str(last))
            for ev, fns in callbacks.items():
                for fn in fns:
                    model.add_callback(ev, fn)
            # only memory/speed settings may change on resume; pass them only if given on this command line
            extra = {k: v for k, v in (("batch", run["batch"]), ("workers", run["workers"]), ("imgsz", run["imgsz"]))
                     if v is not None}
            model.train(resume=str(last), data=str(data_yaml), device=device, **extra)
            save_dir = Path(model.trainer.save_dir)
    else:
        # ----- new run -----
        run = {k: v for k, v in run.items() if v is not None}
        run = {"epochs": 50, "imgsz": 640, "batch": 16, "workers": min(8, os.cpu_count() or 1), **run}
        imgsz, batch = run["imgsz"], run["batch"]
        start = YOLO(a.weights)  # downloads official weights if needed
        scale = a.scale or start.model.yaml.get("scale") or "n"
        start_is_gate = has_gate(start.model)
        if a.arm == "gate" and start_is_gate and levels_of(start.model) != levels:
            sys.exit(f"[rgate] --weights {a.weights} has gates on {levels_of(start.model)}, but --gate-levels is "
                     f"{levels}; start from stock weights or use the matching --gate-levels")
        if a.arm == "baseline":
            if start_is_gate:
                sys.exit("[rgate] --arm baseline needs stock YOLO26 weights, got an RGATE checkpoint")
            model = start
        else:
            cfg_path = gate_cfg(scale, a.lam, a.no_prior, levels, project / "cfg")
            model = YOLO(str(cfg_path))
            # Hand the trainer the start weights themselves (same as the baseline arm), not model.load() into the YAML
            # model: that left placeholder class names "0".."9", so with cls_remap=True only the baseline got its class
            # rows copied by name. With cls_remap=False (HYP) both arms start with a fresh class layer either way;
            # this keeps the two arms equal even if cls_remap is ever switched back on (decision log #20).
            start_model = start.model if start_is_gate else remap_into_gate(start.model, cfg_path)
            model.model, model.ckpt = start_model, {"model": start_model}
        name = a.name or f"{tag}_{scale}_{data_tag}_s{a.seed}" + ("_smoke" if a.smoke else "")
        config = {"args": vars(a), "resolved": {**run, "device": device, "scale": scale, "name": name,
                  "start_is_gate": start_is_gate, "gate_levels": levels if a.arm == "gate" else None},
                  "hyp": HYP, "split_sha1": split_hash, "data": info, "eval": eval_info, "versions": versions}
        if a.arm == "gate":
            callbacks["on_train_epoch_end"] = [gamma_logger([])]
        # written when training starts (not at the end), so a crashed or interrupted run still records its settings
        callbacks["on_train_start"] = [lambda tr: Path(tr.save_dir, "run_config.json").write_text(
            json.dumps(config, indent=1, default=str))]
        for ev, fns in callbacks.items():
            for fn in fns:
                model.add_callback(ev, fn)
        model.train(data=str(data_yaml), project=str(project), name=name, exist_ok=False, seed=a.seed,
                    device=device, **run, **HYP)
        save_dir = Path(model.trainer.save_dir)
    print(f"[rgate] training done -> {save_dir}")

    # ---------- evaluate on extra sets ----------
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        best = save_dir / "weights" / "last.pt"
        print(f"[rgate] WARNING no best.pt; evaluating {best.name}")
    rows = []
    for e in a.eval:
        ey = data_prep.resolve_yaml(e)
        r = YOLO(str(best)).val(data=str(ey), split="val", imgsz=imgsz, batch=batch if batch > 0 else 16,
                                device=device, project=str(save_dir), name=f"eval_{Path(e).resolve().name}",
                                exist_ok=True, plots=False)
        row = {"eval_set": Path(e).resolve().name, "images": eval_info[e]["images"],
               "precision": r.box.mp, "recall": r.box.mr, "mAP50": r.box.map50, "mAP50-95": r.box.map}
        for i, c in enumerate(r.ap_class_index):
            row[f"AP50-95/{r.names[int(c)]}"] = float(r.box.maps[int(c)])
        rows.append(row)
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(save_dir / "eval_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            print(f"[rgate] eval {r['eval_set']}: P {r['precision']:.3f} R {r['recall']:.3f} "
                  f"mAP50 {r['mAP50']:.3f} mAP50-95 {r['mAP50-95']:.3f}")
    # marker read by the notebook's resume cell: this run needs nothing more
    (save_dir / "done.json").write_text(json.dumps({"best": str(best), "eval_sets": [r["eval_set"] for r in rows]},
                                                   indent=1))
    print(f"[rgate] all outputs in {save_dir}")


if __name__ == "__main__":
    main()
