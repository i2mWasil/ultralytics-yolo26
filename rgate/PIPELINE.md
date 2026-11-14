# RGATE pipeline — how to run everything

This is the step-by-step guide for training and comparing the models on the RTX 4070 PC.
Background, design decisions and verified facts: `rgate/CLAUDE.md`. Why the plan looks like this:
`observation/2026-11-15-training-plan.md`, `observation/2026-11-15-phantom-levels-first-check.md`,
`observation/2026-11-15-class-head-start.md`.

**The plan: two training runs.** The base model, then **one** gate design (`p3` or `p3p4`), chosen by the level
check.

```
 night training set (~28k)                  eval sets (BDD100K val): rain-night (main), clear-night, all
          │
 A. check        level check, no training ──> decision: p3 or p3p4
          │
 B. base         train YOLO26n (stock)
          │
 C. p3 | p3p4    train YOLO26n + the gate design chosen in A (only that one)
          │      (B and C each: train on the night set, pick best.pt, evaluate on every eval set)
          │
 D. compare ──> comparison_<train set>_s<seed>.md (accuracy per eval set + phantom counts per level)
```

Everything runs through one script, `rgate/run_all.sh`, with the step names as arguments:

| step | what it does | time on the 4070 (estimate) |
|---|---|---|
| `check` | A. level check on the night training set; writes the decision `p3` or `p3p4` | ~10–20 min |
| `base` | B. trains the base model | ~1.2–3.3 h |
| `p3` | C. trains YOLO26n + gate on P3 (only if the check said `p3`) | ~1.3–3.5 h |
| `p3p4` | C. trains YOLO26n + gates on P3 and P4 (only if the check said `p3p4`) | ~1.3–3.5 h |
| `compare` | D. side-by-side report of the finished runs | ~5 min |
| *(no step)* | same as `check base` | |

Total: about 3–7 hours. These are estimates: the CPU (image loading) sets the speed. Read the real time per epoch
from the first epoch of the base run.

---

## 0. What you need

- The PC with the RTX 4070 (12 GB), running **Linux or WSL2** (`run_all.sh` is a bash script; native Windows is also
  slower at loading images). As many CPU cores as possible.
- This repo (patched Ultralytics 8.4.212 + `rgate/`). Copy the **current** repo to the PC, and copy it again after
  every code change. Do not `pip install ultralytics`: the scripts always use the repo's own copy.
- About 10 GB free disk for the datasets plus ~1 GB for the runs.

### YAML files: what each run needs

| run | model definition | where it comes from |
|---|---|---|
| `base` | stock YOLO26n | built into `yolo26n.pt` (downloaded automatically on first use) |
| `p3` | `rgate/yolo26-rgate-p3.yaml` | in the repo: gate on P3 (layer 23), Detect = layer 24 |
| `p3p4` | `rgate/yolo26-rgate-p3p4.yaml` | in the repo: gates on P3 (layer 23) and P4 (layer 24), Detect = layer 25 |
| every run | `<dataset>/data.yaml` | **you create it**, one in each dataset folder (section 1) |

Made automatically (never edit by hand): the per-run model config `runs/rgate/cfg/yolo26n-rgate-<p3|p3p4>-l0.05.yaml`
(the repo YAML + the run's aux-loss weight) and the split file `runs/rgate/splits/<train set>/data.yaml`.

`rgate/yolo26-rgate.yaml` is the **old name** of the p3p4 file. It now holds only a note and nothing reads it:
delete it (`rm rgate/yolo26-rgate.yaml`).

## 1. Prepare the datasets

Every dataset is a folder in **exactly the format of `dataset/sampled_*`** (see `rgate/CLAUDE.md` §6b):

```
<name>/
  data.yaml      # train: images, val: images, nc: 10, names: {0: car, 1: traffic sign, 2: traffic light, 3: person,
                 #   4: truck, 5: bus, 6: bike, 7: rider, 8: motor, 9: train}   (copy it from dataset/sampled_bdd100k)
  images/*.jpg
  labels/*.txt   # one per image, "class cx cy w h" normalised 0-1; empty file = no objects
```

Put them all in one folder, e.g. `~/data/`:

| folder | images from | used for |
|---|---|---|
| `bdd_train_night` | BDD100K **train**, time of day = **night** only (~28k; no dawn/dusk) | training (split 90/10 automatically: 90% train, 10% to pick `best.pt`) and the level check |
| `bdd_val_night_rain` | BDD100K **val**, night + rainy | **main** eval set |
| `bdd_val_night` | BDD100K **val**, night + clear | eval: "does the gate hurt clear nights?" |
| `bdd_val_all` | BDD100K **val**, all | eval: side check only (contains day; both models are night-trained) |

Rules:
- **No val image may be in the training set.** BDD100K train and val names never repeat; the script checks every eval
  set against the training list and stops on any overlap.
- The class order must be the same in every `data.yaml` (the script checks it).

## 2. Set up the PC (once)

```bash
cd ~/ultralytics-yolo26                                   # the repo on the PC
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision                             # Linux PyPI wheels include CUDA
pip install -e . -r rgate/requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # True + RTX 4070
rm -f rgate/yolo26-rgate.yaml                             # old name of yolo26-rgate-p3p4.yaml (section 0)
```
If it prints `False`, install the CUDA build of PyTorch for your driver (pytorch.org, "Get started").
Every new terminal: `cd ~/ultralytics-yolo26 && source .venv/bin/activate`.

Check the datasets (reads every label; ~1 min) and run one short test of each model (1 epoch, 320 px, ~1 min each):
```bash
python rgate/train_rgate.py --data ~/data/bdd_train_night \
    --eval ~/data/bdd_val_night_rain ~/data/bdd_val_night ~/data/bdd_val_all --check-only
python rgate/train_rgate.py --arm baseline --data ~/data/bdd_train_night --eval ~/data/bdd_val_night_rain --smoke
python rgate/train_rgate.py --arm gate --lam 0.05 --gate-levels p3p4 --data ~/data/bdd_train_night \
    --eval ~/data/bdd_val_night_rain --smoke
```
The smoke tests only prove that everything runs on this GPU. Their numbers mean nothing (after one epoch from a fresh
class layer the scores swing wildly). The p3p4 smoke test also covers p3 (same code, one gate fewer). Smoke runs end
in `_smoke` and the pipeline ignores them.

Set these once per terminal (or put them in front of every `run_all.sh` command):
```bash
export DATA=~/data TRAIN_SET=bdd_train_night EVAL_SETS="bdd_val_night_rain bdd_val_night bdd_val_all" WORKERS=7
```
`WORKERS` = number of CPU cores minus 1. The **first** name in `EVAL_SETS` is the main eval set.

## 3. Step A — level check (no training)

```bash
bash rgate/run_all.sh check
```
Runs the official COCO `yolo26n.pt` over **every image of `bdd_train_night`**, finds its false boxes and records
which feature level (P3 = small, stride 8; P4 = medium, stride 16; P5 = large, stride 32) each came from. Then it
applies a rule that was **fixed before the check was run** (`rgate/level_check.py`; decision #21 in `CLAUDE.md`):

```
phantom candidate = kept box (conf ≥ 0.25) with IoU < 0.1 to every labelled object
road area         = box centre in the lower half of the image
score             = share of road-area phantom candidates that come from P4 = P4 / (P3 + P4 + P5)

score ≥ 0.25  -> p3p4  (gates on P3 and P4)
score < 0.25  -> p3    (gate on P3 only)
< 100 road-area candidates -> inconclusive -> p3p4
```

The terminal ends with, for example:
```
[check] score = P4 share of road-area candidates = 0.412  (95% interval 0.380-0.445)
[check] DECISION: p3p4  -> p3p4: YOLO26 + gates on P3 and P4   (P4 share 0.412 >= 0.25)
[check] next: bash rgate/run_all.sh base (if not done), then bash rgate/run_all.sh p3p4
```

Output: `runs/rgate/level_check/yolo26n_bdd_train_night/`

| file | content |
|---|---|
| `decision.json` | the decision, the score, its 95% interval (bootstrap over images), counts per level, warnings |
| `summary.json`, `boxes.csv`, `images.txt` | every kept box with level, class, score, IoU, status |
| `vis/` | the 40 images with the most false boxes (red = P3, yellow = P4, blue = P5, green = labels) |

How to read it:
- **"close call"**: the 95% interval contains 0.25. The rule still decides; mention it in the thesis.
- **"... come from P5"**: a large share of candidates comes from P5, which neither design gates. Report it as a limit.
- Look at the pictures: are the false boxes really reflections (road, hood, windscreen), or unlabelled objects?
- **Do not change the rule or the threshold after seeing the score.**

Limits: the check uses the COCO model (not trained on BDD100K; 9 of 10 classes matched by name, no "rider"), so its
false boxes stand in for those of the trained models. It runs on the training set on purpose, so the design is never
tuned on the eval sets.

## 4. Steps B and C — the two training runs

| | B. `base` | C. `p3` (if the check says p3) | C. `p3p4` (if the check says p3p4) |
|---|---|---|---|
| model | YOLO26n (stock) | YOLO26n + reflection gate on P3 | YOLO26n + gates on P3 and P4 |
| model file | inside `yolo26n.pt` | `rgate/yolo26-rgate-p3.yaml` | `rgate/yolo26-rgate-p3p4.yaml` |
| aux gate loss | – | λ = 0.05 | λ = 0.05 (mean over both gates) |
| run name | `base_n_bdd_train_night_s0` | `gate-p3-l0.05_n_bdd_train_night_s0` | `gate-p3p4-l0.05_n_bdd_train_night_s0` |
| extra compute | – | ≈ +2% | ≈ +4% |

**Identical in both runs** (this is what makes the comparison fair): start weights `yolo26n.pt`, training data and its
90/10 split, 50 epochs, 640 px, batch 32, seed 0, and every setting in `HYP` (`train_rgate.py`): optimizer MuSGD
lr 0.01, lighter colour augmentation, `cls_remap=False` (both start with a fresh class-output layer), mosaic off for
the last 10 epochs. Never change any of these between the two runs.

Train them:
```bash
bash rgate/run_all.sh base        # B (also done by "bash rgate/run_all.sh" with no step, after the check)
bash rgate/run_all.sh p3p4        # C: use the design the check printed: p3 or p3p4
```
`run_all.sh` refuses to train a gate design if the check has not been run, if the check chose the other design, or if
a run of the other design already exists. To go against the check on purpose:
`OVERRIDE_DECISION=1 bash rgate/run_all.sh p3` (then report that in the thesis).

Each run trains, keeps every new best checkpoint, picks `best.pt` on the 10% split, evaluates `best.pt` on every eval
set, and writes `done.json` when finished. The same runs without the runner:
```bash
python rgate/train_rgate.py --arm baseline --data ~/data/bdd_train_night \
    --eval ~/data/bdd_val_night_rain ~/data/bdd_val_night ~/data/bdd_val_all --epochs 50 --batch 32 --seed 0
python rgate/train_rgate.py --arm gate --lam 0.05 --gate-levels p3p4   # + the same --data / --eval / --epochs / --batch / --seed
```
`--gate-levels` (`p3` or `p3p4`) is required for every gate run, including `--resume`.

While the gate run trains, look at `gamma.json` in its run folder after the first few epochs: `strength` per gate
(tanh(10·γ); positive = suppress, negative = boost). Values staying near 0 mean the gate is barely used.

## 5. Running it unattended

```bash
mkdir -p runs/rgate
nohup bash rgate/run_all.sh check base >> runs/rgate/run_all.log 2>&1 &
tail -f runs/rgate/run_all.log                     # Ctrl+C stops watching, not training
# when it has finished: read the decision at the end of the log, then
nohup bash rgate/run_all.sh p3p4 compare >> runs/rgate/run_all.log 2>&1 &     # or: p3 compare
```
- **Restart-safe:** start the same command again after a crash, reboot or Ctrl+C. Finished runs are skipped, an
  interrupted run resumes from its `last.pt`, the level check is reused.
- Only one `run_all.sh` per project folder can run at a time.
- `DATA` must stay the same path across restarts (the split files and the resume check hold full image paths).
- The last lines of every `run_all.sh` call show the finished runs and the level-check decision.

## 6. Step D — compare

```bash
bash rgate/run_all.sh compare
```
Writes and prints `runs/rgate/comparison_bdd_train_night_s0.md` with, for base and the gate run:
1. **Accuracy per eval set** (precision, recall, mAP50, mAP50-95 from `eval_summary.csv`), with the difference to the
   base model in brackets.
2. **Phantom candidates on the main eval set** (`bdd_val_night_rain`): recall at conf 0.25, false boxes away from all
   objects per image, road-area ones per image, and per level. This runs `phantom_levels.py` once per model.
3. **Gate strength** at the last epoch.

How to judge (one seed per model):
- Differences under about **0.01 mAP** (1 point) or **0.02 precision** on rain-night are **inconclusive**, not a gain.
- A real result: the gate lowers road-area phantom candidates and raises precision on rain-night, without lowering
  recall, and clear-night accuracy does not drop.
- The day side check (`bdd_val_all`) only shows whether the gate breaks anything outside night images.

## 7. Where everything is

```
runs/rgate/
  level_check/yolo26n_bdd_train_night/      decision.json, summary.json, boxes.csv, images.txt, vis/
  splits/bdd_train_night/                   the 90/10 split used by both runs
  cfg/yolo26n-rgate-<p3|p3p4>-l0.05.yaml    model config of the gate run (made automatically)
  base_n_bdd_train_night_s0/                B
  gate-p3-l0.05_n_bdd_train_night_s0/       C, if the check said p3
  gate-p3p4-l0.05_n_bdd_train_night_s0/     C, if the check said p3p4
  comparison_bdd_train_night_s0.md          D
  run_all.log                               log (if started with nohup as above)
```
Inside each run folder: `weights/best.pt`, `weights/last.pt`, `weights/best_ep<NNN>_*.pt` (every new best),
`results.csv` (per epoch), `eval_summary.csv` (per eval set), `gamma.json` (gate run), `run_config.json`
(all settings), `eval_<set>/`, `phantom_levels_<main eval>/` (after compare), `done.json`.

## 8. Troubleshooting

| problem | fix |
|---|---|
| `torch.cuda.is_available()` is False | install the CUDA build of PyTorch for your driver; check that `nvidia-smi` works |
| "--arm gate needs --gate-levels" | add `--gate-levels p3` or `--gate-levels p3p4` (the check's decision) |
| run_all refuses `p3` / `p3p4` | run `bash rgate/run_all.sh check` first, then train the design it printed |
| "eval images leak into training" | a val image is in `bdd_train_night`; rebuild the training set from BDD100K **train** only |
| class names differ | copy `data.yaml` from `dataset/sampled_bdd100k` into every dataset |
| out of GPU memory | `BATCH=16`, but then use 16 for **both** runs (start the base run again) |
| very slow epochs | CPU-bound: raise `WORKERS` (cores − 1), use Linux/WSL2, keep the data on a local SSD |
| "does not match" on restart | the restart used other flags (`--lam`, `--gate-levels`, dataset) than the run; use the same command |
| level check says inconclusive | too few road-area false boxes; the rule picks p3p4. Report it |
| code changed after a run started | the finished run used the old code; run both again with the new code so they match |

## 9. Rules that keep the result valid

1. Both runs: same data, split, epochs, image size, batch, seed, start weights, `HYP`, code version.
2. Eval sets only from BDD100K val; never train on them; never pick settings by looking at eval-set results.
3. The level-check rule and threshold are fixed; train the design it chose. Do not change them after seeing the score.
4. Report every eval set, not only the best one, and name the limits: one seed, COCO model for the level check,
   class-output layer re-initialised for both models (`cls_remap=False`), night-only training, only one gate design
   trained (the other design was not tested).
