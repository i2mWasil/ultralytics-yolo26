"""Dataset helpers for RGATE training. Works the same for the 150-image samples and the full dataset.

Expected dataset folder (one per dataset; this is the format of dataset/sampled_*):
    <dataset>/
        data.yaml      # nc + names; train/val keys (relative to this folder)
        images/*.jpg   # any image extension Ultralytics accepts
        labels/*.txt   # YOLO format: "class cx cy w h", values normalised 0-1, one file per image

If data.yaml uses the same folder for train and val (like the samples), the training set is split
automatically into train/val by a hash of each file name. The split is stable: adding more images
never moves an existing image from train to val or back.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_yaml(p: str | Path) -> Path:
    """Accept a dataset folder or a .yaml path; return the data.yaml path."""
    p = Path(p).expanduser().resolve()
    if p.is_dir():
        cands = [p / "data.yaml", *sorted(p.glob("*.yaml"))]
        for c in cands:
            if c.is_file():
                return c
        raise FileNotFoundError(f"No data.yaml in {p}")
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def load_cfg(yaml_path: Path) -> dict:
    cfg = yaml.safe_load(yaml_path.read_text())
    root = Path(cfg.get("path") or yaml_path.parent)
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    cfg["_root"] = root
    names = cfg["names"]
    cfg["names"] = {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    cfg["nc"] = int(cfg.get("nc", len(cfg["names"])))
    if cfg["nc"] != len(cfg["names"]):
        raise ValueError(f"{yaml_path}: nc={cfg['nc']} but {len(cfg['names'])} names")
    return cfg


def list_images(cfg: dict, split: str) -> list[Path]:
    """Images for a split. The split entry may be a folder, a .txt list, or a list of those."""
    entries = cfg[split] if isinstance(cfg[split], list) else [cfg[split]]
    out = []
    for e in entries:
        p = Path(e) if Path(e).is_absolute() else cfg["_root"] / e
        if p.is_dir():
            out += [f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT]
        elif p.suffix == ".txt":
            for line in p.read_text().splitlines():
                if line.strip():
                    q = Path(line.strip())
                    out.append(q if q.is_absolute() else (p.parent / q).resolve())
        else:
            raise FileNotFoundError(f"{split} entry not found: {p}")
    return sorted(set(out))


def label_for(img: Path) -> Path:
    """Same rule Ultralytics uses (img2label_paths): last '/images/' -> '/labels/', extension -> .txt."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    s = str(img)
    if sa not in s:
        raise ValueError(f"Image path has no 'images' folder, so Ultralytics cannot find its label: {img}")
    return Path(sb.join(s.rsplit(sa, 1))).with_suffix(".txt")


def check_dataset(cfg: dict, images: list[Path], tag: str, max_report: int = 5) -> dict:
    """Check every label file. Returns counts; raises on errors that would silently corrupt training."""
    nc, missing, bad, empty, n_boxes = cfg["nc"], [], [], 0, 0
    per_class = [0] * nc
    for img in images:
        lab = label_for(img)
        if not lab.is_file():
            missing.append(img.name)
            continue
        lines = [ln.split() for ln in lab.read_text().splitlines() if ln.strip()]
        if not lines:
            empty += 1
        for parts in lines:
            ok = len(parts) == 5
            if ok:
                try:
                    c, *xywh = int(float(parts[0])), *map(float, parts[1:])
                    ok = 0 <= c < nc and all(0.0 <= v <= 1.0 for v in xywh)
                except ValueError:
                    ok = False
            if not ok:
                bad.append(f"{lab.name}: {' '.join(parts)}")
            else:
                per_class[c] += 1
                n_boxes += 1
    if bad:
        raise ValueError(f"[{tag}] {len(bad)} bad label lines (need 'class cx cy w h', class < {nc}, values 0-1). "
                         f"First: {bad[:max_report]}")
    if not images:
        raise ValueError(f"[{tag}] no images found")
    return {"images": len(images), "boxes": n_boxes, "missing_labels": len(missing), "empty_labels": empty,
            "missing_examples": missing[:max_report], "per_class": {cfg["names"][i]: n for i, n in enumerate(per_class)}}


def is_val(name: str, frac: float) -> bool:
    h = int(hashlib.md5(Path(name).stem.encode()).hexdigest()[:8], 16)
    return (h % 10_000) < frac * 10_000


def build_train_yaml(train_src: str, val_src: str | None, out_dir: Path, val_frac: float) -> tuple[Path, dict]:
    """Write a data yaml (absolute paths, list files) for training. Returns (yaml path, info)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tcfg = load_cfg(resolve_yaml(train_src))
    train_imgs = list_images(tcfg, "train")
    if val_src:
        vcfg = load_cfg(resolve_yaml(val_src))
        same_names(tcfg, vcfg, val_src)
        val_imgs = list_images(vcfg, "val")
        mode = "separate val dataset"
    else:
        val_imgs = list_images(tcfg, "val")
        if set(val_imgs) == set(train_imgs) or set(val_imgs) & set(train_imgs):
            pool = sorted(set(train_imgs) | set(val_imgs))
            val_imgs = [p for p in pool if is_val(p.name, val_frac)]
            train_imgs = [p for p in pool if not is_val(p.name, val_frac)]
            mode = f"auto split {1 - val_frac:.0%}/{val_frac:.0%} by file-name hash (data.yaml train and val overlap)"
        else:
            mode = "train/val from data.yaml"
    if not val_imgs or not train_imgs:
        raise ValueError(f"Empty split: train={len(train_imgs)} val={len(val_imgs)}; raise --val-frac or add images")
    (out_dir / "train.txt").write_text("\n".join(map(str, train_imgs)) + "\n")
    (out_dir / "val.txt").write_text("\n".join(map(str, val_imgs)) + "\n")
    data = {"path": str(out_dir), "train": str(out_dir / "train.txt"), "val": str(out_dir / "val.txt"),
            "nc": tcfg["nc"], "names": tcfg["names"]}
    yp = out_dir / "data.yaml"
    yp.write_text(yaml.safe_dump(data, sort_keys=False))
    info = {"mode": mode, "source": str(resolve_yaml(train_src)),
            "train": check_dataset(tcfg, train_imgs, "train"), "val": check_dataset(tcfg, val_imgs, "val")}
    return yp, info


def same_names(a: dict, b: dict, label: str):
    if a["names"] != b["names"]:
        raise ValueError(f"Class names/order differ for {label}:\n  train: {a['names']}\n  other: {b['names']}\n"
                         "Every dataset must use the same class ids, or all metrics are wrong.")


def overlap(train_yaml: Path, other_src: str) -> int:
    """How many images of `other_src` (its val split) also appear in the training list, matched by file name."""
    tr = {Path(p).name for p in Path(load_cfg(train_yaml)["train"]).read_text().split()}
    ocfg = load_cfg(resolve_yaml(other_src))
    return len(tr & {p.name for p in list_images(ocfg, "val")})
