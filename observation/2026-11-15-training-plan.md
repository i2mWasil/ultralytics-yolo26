# Observation: training plan — data, compute and scope

**Date:** 2026-11-15
**Decision:** the **minimum plan**. Two training runs, one seed, night-only data, on the RTX 4070 PC:
**A0 baseline** and **A2 gate + aux loss** (gates on P3 + P4), seed 0, 50 epochs, 640 px, batch 32.
Everything below explains why, what it costs, and what this plan can and cannot prove.

> **Update (later on 2026-11-15, decision log #21, #22):** the run order changed. The gate placement is now decided
> **before** training by `rgate/level_check.py` on the night training set (COCO model, fixed rule): design `p3`
> (gate on P3, `rgate/yolo26-rgate-p3.yaml`) or `p3p4` (gates on P3 and P4, `rgate/yolo26-rgate-p3p4.yaml`).
> Still **two runs**: base, then only the design the check chose (run names `base_...` and `gate-p3-l0.05_...` or
> `gate-p3p4-l0.05_...`).
> Steps and commands: `rgate/PIPELINE.md`. Sections 1–3 and 5 below (data, compute, arms/seeds, limits) still apply;
> section 4 (order A0 → level check on A0 → A2) is replaced.

## 1. Training data: BDD100K night images only

**What "100k" really is.** BDD100K = 70k train + 10k val + 20k test images. The test labels are not public, so the
labelled pool for training is the 70k **train** split. The 10k **val** split is kept for evaluation only.

**Night share.** About 28k train images and about 4k val images have time of day = night (from BDD100K's published
label statistics; count again from your own files, decision log #10). The "30k night images" number mixes both splits.
**Only the train-split night images may be used for training.** Any val night image in the training set would leak
into the rain-night eval set and make the results invalid. `train_rgate.py` stops on overlap only for sets passed with
`--eval`, so pass every eval set.

**Use the label "night" only** (no dawn/dusk), so the training set means the same thing as the night eval sets.

| option | images | pros | cons |
|---|---|---|---|
| **night only (chosen)** | ~28k | ~2.5x cheaper per epoch; every image teaches the gate about night lights; the baseline also specialises on night, so a gain is more convincing | no claim about day accuracy; neither model is meant for day driving |
| all train (day + night) | ~70k | standard setup; can test "does not hurt day accuracy" | ~2.5x the cost; about 60% day images, where the prior fires on sky, white buildings and sunlit road, so the aux loss teaches the gate to suppress bright day backgrounds, which has nothing to do with reflections |

**Effect on the hypothesis** (now scoped to night driving): *on rain-night images, RGATE lowers the phantom rate and
raises precision without hurting recall or clear-night accuracy, at about +4% compute.* Day images are still evaluated
(`bdd_val_all`), but only as a side check where both models are equally outside their training data.

## 2. Compute: how long on an RTX 4070

These are **estimates, not measurements**. YOLO26n is small for an RTX 4070; the limit is the CPU, which builds every
training image from 4 photos (mosaic) and must decode the JPEGs. On Colab (2 CPU cores, T4) this gave about 100 images/s.

Assumptions: ~25k training images per epoch (90% of ~28k; 10% is the val split used to pick `best.pt`), ~2.8k val images
per epoch, 640 px, batch 32. The gate arm is ~5-10% slower.

| CPU cores | images/s (guess) | per epoch | per run (50 epochs) | minimum plan (2 runs) | full plan (12 runs) |
|---|---|---|---|---|---|
| 4 | ~120 | ~4 min | ~3.3 h | ~7 h | ~40 h |
| 8 | ~250 | ~2 min | ~1.7 h | ~3.5 h | ~20 h |
| 12+ (GPU becomes the limit) | ~350 | ~1.5 min | ~1.2 h | ~2.5 h | ~15 h |

Add about 5 minutes per run for the final evaluations, and a few minutes for the level check.
**Measure, don't trust the table:** time epoch 1 of the baseline. Total hours ≈ seconds per epoch × 50 ÷ 3600.
Memory: batch 32 used 6.4 GB on the T4 with one gate; the 4070 has 12 GB, so two gates fit. Use the same batch for both runs.
Use Linux or WSL2: native Windows starts data workers slowly (often 20-40% slower). Set `WORKERS` to CPU cores − 1.

## 3. What "arms" and "seeds" mean

**Arm** = one version of the model that is trained and compared.

| arm | what it is | question it answers | in the minimum plan? |
|---|---|---|---|
| A0 baseline | stock YOLO26n, same data, epochs and settings | "is any gain just more training?" | **yes** |
| A1 gate | gates, detection loss only | "does the gate help on its own?" | no |
| A2 gate + aux | gates + the aux gate loss (λ = 0.05) | "does the gate help?" (most likely to work) | **yes** |
| A3 control | gates with the prior switched off | "is it the prior, or just extra parameters?" | no |

**Seed** = the starting number of all randomness in training (start values of new layers, image order, augmentation).
The same arm trained with two seeds gives different scores, often 0.5-1 mAP apart on a small set like rain-night.
Three seeds per arm would measure that spread, so a gain can be shown to be bigger than chance.

Full plan = 4 arms × 3 seeds = 12 runs. Minimum plan = 2 arms × 1 seed = 2 runs.

## 4. The plan, step by step

1. **A0 baseline**, seed 0, 50 epochs.
2. **Level check** on A0's `best.pt` over the full rain-night val (`rgate/phantom_levels.py`). It repeats the first level
   check (`2026-11-15-phantom-levels-first-check.md`) on a BDD-trained model. If most background false positives come
   from P3, train A2 with `--gate-levels p3`; otherwise keep P3 + P4 (now design `p3p4`).
3. **A2 gate + aux**, seed 0, same settings, gate placement from step 2.
4. Compare on `bdd_val_night_rain` (main), `bdd_val_night` (clear night) and `bdd_val_all` (side check).

`rgate/run_all.sh` does exactly this: it runs A0, runs the level check, **stops**, and when started again runs A2.

Same for both runs: data, split, epochs, image size, batch, seed, starting weights (`yolo26n.pt`) and `HYP`.
`HYP` now fixes the optimizer (MuSGD, lr 0.01, momentum 0.9), the same one `optimizer=auto` picks for a run this size.
Before, small runs (smoke tests, pilots) silently used AdamW, so they did not test the real setup.

## 5. What the minimum plan can and cannot show

**Can show:** whether gate + aux beats the baseline on rain-night images in this one pair of runs, and at what cost.

**Cannot show:**
- **Whether a small difference is real.** With one seed there is no measure of run-to-run noise. Treat rain-night
  differences under about 1 mAP (or about 2 points of precision) as **inconclusive**, not as a gain. A bootstrap over
  images (resample the eval images, recompute both models' scores, see how often A2 > A0) gives a partial answer
  from the same two runs; it measures image-sampling noise, not training noise.
- **Why it works.** Without A1 there is no test of the aux loss on its own; without A3 there is no test of the prior.
  The thesis can claim "gate + aux vs baseline", not "the prior causes the gain". Say this openly in the limits section.
- **Anything about day driving** (section 1).

**If more time appears later, add in this order:** seeds 1 and 2 for A0 and A2 (makes the main result credible),
then A3 (tests the prior), then A1.

## 6. Commands (repo root on the 4070 PC)

```bash
# once: check every dataset (labels, class order, split, overlap with eval sets)
python rgate/train_rgate.py --data ~/data/bdd_train_night \
    --eval ~/data/bdd_val_night_rain ~/data/bdd_val_night ~/data/bdd_val_all --check-only
# once: 1-epoch smoke test on this machine
python rgate/train_rgate.py --arm gate --lam 0.05 --data ~/data/bdd_train_night \
    --eval ~/data/bdd_val_night_rain --smoke
# the plan: A0 + level check (stops), then run the same line again for A2
DATA=~/data TRAIN_SET=bdd_train_night EVAL_SETS="bdd_val_night_rain bdd_val_night bdd_val_all" \
    WORKERS=<cores-1> bash rgate/run_all.sh
```
