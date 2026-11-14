# Observation: the gate arm started with an untrained class head (fixed)

**Date:** 2026-11-15
**Status:** bug found, fixed, verified on the 150-image samples. Decision log #20 in `rgate/CLAUDE.md`.
**In one line:** the baseline got 6 of its 10 class rows copied from COCO, the gate arm got none, so the comparison
was unfair to the gate. Now **neither arm** gets class rows copied (`cls_remap=False`).

## 1. What was seen

A 1-epoch smoke comparison of baseline vs gate (λ = 0.05) on the samples, same settings (MuSGD, 320 px, batch 4, seed 0):

| | baseline | gate (before fix) |
|---|---|---|
| train class loss, epoch 1 | 3.74 | 6.17 |
| val mAP50 (sample split) | 0.116 | 0.049 |
| boxes at conf ≥ 0.25 on 10 rain-night images (`phantom_levels.py`) | 33 | **2,638** (almost 300 per image, the maximum) |

The gate could not explain this: after 1 epoch its strength was about 0.01, so it changed the features by about 1%.
Recording the loss batch by batch showed the gap already at **batch 1** (class loss 4.5 vs 7.9), when γ = 0 and the
gate does nothing at all. So the two arms did not start from the same weights.

## 2. Cause

When the class count changes (COCO 80 → BDD 10), Ultralytics 8.4.212 does two things while loading the start weights:

1. It rebuilds the class branch at the source width so its hidden layers transfer. This worked in both arms.
2. With `cls_remap=True` (the default) it copies the COCO class rows into the new class layer **by class name**:
   car, person, truck, bus, train, traffic light (6 of 10; "traffic sign", "bike", "rider", "motor" have no COCO name
   match). It does this by matching the names of the model it is given against the dataset names.

The baseline arm gives the trainer the COCO checkpoint, which has real class names, so it got the 6 rows. The gate arm
gave the trainer a model built from the gate YAML and filled with `model.load()`. That model kept the YAML's
placeholder names "0", "1", … "9", so no name matched and **all 10 class rows started from scratch**.

Evidence: comparing the two arms' weights at the moment training starts, all 696 shared tensors were identical except
the class-output layers. In the baseline those layers had per-class biases (−10.18, −9.46, −11.07, −8.26, …) and
weights with spread 0.10; in the gate arm every bias was the default −9.46 and the spread was 0.065 (fresh).

**Effect:** every gate run before this fix (sample runs, the Colab 50-epoch run) started with a handicap the baseline
did not have. Their numbers cannot be compared with any baseline.

## 3. The two possible fixes (both fair)

| | A: copy for both | B: copy for neither (chosen) |
|---|---|---|
| how | gate arm hands the trainer the COCO-named model, so it also gets the 6 rows | `cls_remap=False` in `HYP`: no arm gets class rows copied |
| start of training | both arms keep the COCO knowledge for 6 classes | both arms learn all 10 class rows from scratch |
| 1-epoch val mAP50, baseline (seeds 0 / 1 / 2) | 0.116 / 0.136 / – | 0.047 / 0.050 / 0.036 |
| matches Ultralytics' default fine-tuning | yes | no |

**Chosen: B** (user decision, 2026-11-15). It is the simpler rule to explain: no class knowledge is reused for
either model. What it costs, and what to write in the thesis:
- Both models start weaker and need the first epochs to learn the class layer. Over 50 epochs on ~28k images this
  probably matters little, but it is not measured.
- The baseline is **not** a default Ultralytics fine-tune (the default would copy the 6 rows). State this in the
  method section: "class-output layers were re-initialised for both models (`cls_remap=False`)".

## 4. What changed in the code

- `rgate/train_rgate.py`, `HYP`: `cls_remap=False` (applies to every arm and every run).
- `rgate/train_rgate.py`, gate arm: the start weights are handed to the trainer directly (`model.model = start_model`)
  instead of `model.load()` into the YAML model. With `cls_remap=False` both ways give a fresh class layer; this way the
  two arms stay equal even if `cls_remap` is ever switched back on.

## 5. Verification (samples, CPU, Ultralytics 8.4.212, torch 2.15)

- **Same start:** at the start of training, both arms differ only in the class-output layers (both fresh, random,
  same spread: 0.063–0.066), with identical class biases. Neither log has a "Remapped ... cls head rows" line.
  Transferred: baseline 696/708, gate 714/726 (the 12 missing tensors = class-output layers in both).
- **`args.yaml`** of both runs shows `cls_remap: false`.
- **5-epoch run, same settings** (320 px, batch 4): the early gap closes.

  | epoch | baseline train / val class loss | gate train / val class loss |
  |---|---|---|
  | 1 | 4.90 / 3.74 | 6.44 / 4.37 |
  | 2 | 3.59 / 3.27 | 4.25 / 3.39 |
  | 3 | 2.81 / 3.34 | 3.16 / 2.76 |
  | 4 | 3.05 / 2.58 | 2.64 / 2.35 |
  | 5 | 2.58 / 2.39 | 2.44 / 2.17 |

  The epoch-1 gap is start-up noise: the 1-epoch baseline class loss alone ranges 4.96–6.46 across seeds 0–2.

## 6. Notes for the real runs

- **Do not read anything into 1-epoch smoke numbers.** With a fresh class layer, box counts after one epoch swing
  widely (1-epoch sample runs: baseline 656 boxes on 10 images, gate 1). Smoke tests only prove the pipeline runs.
- In the first epochs of the real runs both arms will look weak; compare only finished runs.
- On the samples (5 epochs, 135 images) the gate strength went slightly negative (P3 −0.034, P4 −0.003). At this size
  that means nothing; watch `gamma.json` in the real A2 run.
