# Observation: which level do the false positives come from? (first check + clear-night second reading)

**Date:** 2026-11-15
**Question:** The RGATE gate changes only the P3 (stride 8) predictions. Do the phantom boxes on rain-night images come
from P3, or from P4 / P5, where the gate cannot reach them?
**Status:** first hint only. NOT a thesis result. Must be repeated on the BDD-trained baseline (see "To confirm").
**Action taken (2026-11-15):** main design changed to two head-only gates, P3 and P4 (decision log #17 in
`rgate/CLAUDE.md`). The P3-only design stays available with `--gate-levels p3`. The check below still has to be
repeated on the BDD-trained baseline to confirm the choice.
**Later (2026-11-15):** the placement is now decided by `rgate/level_check.py` on the full night training set (decision
log #21, #22), which replaces the "To confirm" plan below. A **second reading on the clear-night sample** was added
(section "Second reading"). Its false-positive percentages differ from the first reading because it uses **different
images** (clear night, not rain night), not because anything changed or improved.

## Setup

| item | value |
|---|---|
| model | official COCO `yolo26n.pt` (stock, **not** trained on BDD100K) |
| data | `dataset/sampled_bdd100k_night_rain`: 150 images from BDD100K val `night_rainy`, 2,441 labelled objects |
| branch | one2many + NMS (Ultralytics 8.4.212 default for val/predict, `nms=None`); NMS-free branch also run |
| settings | conf 0.25, NMS IoU 0.5 (predict defaults), letterbox 640 x 384 (minimal padding, same as `predict()`) |
| classes | COCO names mapped to BDD names (bicycle->bike, motorcycle->motor, stop sign->traffic sign); 9 of 10 BDD classes covered (no rider) |
| tool | `rgate/phantom_levels.py` (box counts checked: identical to `YOLO.predict()` on 40 images) |
| command | `python rgate/phantom_levels.py --weights yolo26n.pt --data dataset/sampled_bdd100k_night_rain --save-vis 12` |

Anchors at this input size: P3 = 3,840, P4 = 960, P5 = 240.

Box labels used below (best IoU = highest IoU with any labelled object):
- `tp`: matches an object (same class, IoU >= 0.5)
- `fp_obj`: IoU >= 0.5 with some object but not a match (duplicate or wrong class)
- `fp_loc`: best IoU 0.1 to 0.5 (near an object, bad box)
- `fp_bg`: best IoU < 0.1 (away from every object) = **phantom candidates**

## Results (default branch: one2many + NMS)

| level | boxes | tp | fp_obj | fp_loc | fp_bg | fp_bg in lower half (road area) | median box height |
|---|---|---|---|---|---|---|---|
| P3 | 126 | 74 | 4 | 34 | 14 | 2 | 49 px |
| P4 | 281 | 205 | 7 | 45 | 24 | **17** | 86 px |
| P5 | 103 | 84 | 12 | 4 | 3 | 1 | 248 px |
| all | 510 | 363 | 23 | 83 | 41 | 20 | |

- Share of `fp_bg` by level: **P3 34%, P4 59%, P5 7%**.
- Road area (box centre in the lower half of the image): **17 of 20** `fp_bg` come from P4.
- `fp_bg` by class: P4 = 18 car, 5 traffic light, 1 bike; P3 = 5 traffic light, 5 person, 4 car.
- Recall at conf 0.25 is only 0.149 (363 / 2,441). The COCO model misses most small BDD objects, so these numbers
  describe a model that is not trained for this data.

NMS-free branch (one2one, `--nms-free`), for comparison: `fp_bg` = P3 16, P4 10, P5 2 (P3 57%, P4 36%, P5 7%);
road-area `fp_bg` = P3 4, P4 9. **The split between P3 and P4 depends on the branch**, so always analyse the branch
the thesis numbers come from (the default one).

## What the road-area false positives actually are (looked at by eye)

Picture: `2026-11-15-phantom-levels-first-check/road_area_false_positives.jpg` (all 20; yellow = P4, red = P3).

| what it is | count (approx.) | level |
|---|---|---|
| clear wet-road reflection of vehicle lights | ~2 | both P4 |
| reflection on the ego car's own hood, dashboard or windscreen | ~6 | mostly P4, one P5 |
| object without a label (parked car, distant car) | ~4 | P4 / P3 |
| raindrops / blur on the windscreen | ~2 | P4 / P3 |
| other clutter (pole, fence, building light) | ~6 | mixed |

So in this sample, **true wet-road reflections are a small part of the phantoms**, and the ones found come from P4.
Reflections inside the car (hood, dashboard, windscreen) are more common than road reflections. They are a different
failure: they sit at fixed places at the bottom of the frame.

## Does the prior see these boxes?

Prior R = clip((V - 0.6) / 0.4) x (1 - Sat), as in the gate. A box "fires" if max R inside it > 0.3.

| boxes | share where the prior fires |
|---|---|
| tp (real objects) | 92% |
| fp_loc | 96% |
| fp_bg (phantom candidates) | 76% |

The prior fires on real objects **more** often than on phantoms (headlights, white cars, lit signs are bright and
colourless too). At box level it cannot tell objects from phantoms. The learned conv and the aux loss (with the
object mask, decision log #14) have to do that.

## Interpretation for the design

1. **Head-only P3 gating cannot reach most of these phantoms.** 59% of `fp_bg`, and 17 of 20 in the road area, come
   from P4. Reflections of nearby vehicles are 50-200 px tall, which is P4's size range (median P4 box 86 px).
2. **P5 is not worth gating.** Only 7% of `fp_bg`. A P5 cell is 32 x 32 px, so suppressing one also suppresses large
   nearby objects, which risks clear-weather accuracy for almost no gain.
3. **Candidate design: two head-only gates, P3 and P4.** Gate layer 16 -> Detect P3 (as now) and add a second gate on
   layer 19 -> Detect P4 (prior max-pooled at stride 16). P5 and the rest of the network stay stock, and both gates
   start as identity (gamma = 0), so step 0 is still exactly stock YOLO26.
   - Preferred over "gate before PAN": gating P3 before PAN only partly cleans P4, because P4 (layer 19) also takes
     ungated top-down features (layer 13); and it changes the features of every level.
   - Cost: about +37k params and +0.12 GFLOPs more for the P4 gate (YOLO26n), so about +4% compute in total instead of +2%.
   - Implemented 2026-11-15: second gate row in the YAML (Detect at index 25), aux loss = mean over the gates,
     weight remap for two gate layers, `--no-prior` / gamma logging / resume checks for all gates.
     Measured cost (YOLO26n): 6.24 -> 6.48 GFLOPs (+3.9%), +47k params.

## Second reading: clear-night sample (2026-11-15)

Same model, branch, settings and tool as above; only the images differ.
Command: `DATA=dataset TRAIN_SET=sampled_bdd100k_night EVAL_SETS="sampled_bdd100k_night_rain" bash rgate/run_all.sh check`
(`level_check.py` -> `phantom_levels.py --split all --conf 0.25 --save-vis 40`).
Data: `dataset/sampled_bdd100k_night` = 150 images from BDD100K val `night_clear`, 2,484 labelled objects. No image is
shared with the rain-night sample.

| level | boxes | tp | fp_obj | fp_loc | fp_bg | fp_bg in lower half (road area) |
|---|---|---|---|---|---|---|
| P3 | 132 | 80 | 5 | 41 | 6 | 3 |
| P4 | 289 | 217 | 10 | 46 | 16 | 9 |
| P5 | 93 | 70 | 11 | 4 | 8 | 7 |
| all | 514 | 367 | 26 | 91 | 30 | 19 |

Side by side with the first reading (both 150 images, same model):

| | rain-night sample (first reading) | clear-night sample (second reading) |
|---|---|---|
| `fp_bg` (phantom candidates), whole image | 41 | 30 |
| share of `fp_bg` by level, P3 / P4 / P5 | 34% / 59% / 7% | 20% / 53% / 27% |
| road-area `fp_bg`, P3 / P4 / P5 | 2 / 17 / 1 (P4 = 85%) | 3 / 9 / 7 (P4 = 47%) |
| recall at conf 0.25 | 0.149 | 0.148 |
| level-check score (P4 share of road-area `fp_bg`) | 0.85 | 0.47 (95% interval 0.25-0.73) |
| decision | p3p4 | p3p4 (inconclusive: 19 < 100 candidates) |

What the 19 road-area `fp_bg` boxes are (looked at by eye; picture:
`2026-11-15-phantom-levels-first-check/clear_night_road_area_false_positives.jpg`):

| what it is | count | level |
|---|---|---|
| the ego car itself: hood across the bottom of the frame (5), dark side of the ego car (1), dashboard reflected in the windscreen (2) | 8 | 6 P5, 2 P4 |
| wet-road reflection (the "clear" images can still have wet roads) | 2 | P4 |
| one large box over lit wet road + hood | 1 | P5 |
| unlabelled object: road drums (2), distant or parked car (3), barrier (1), object at the image edge (1) | 7 | 4 P4, 3 P3 |
| unclear (dark area between cars) | 1 | P4 |

What this adds:
1. **The P4 share fell (85% -> 47% of road-area candidates) mainly because P5 rose (1 -> 7), and 6 of those 7 P5 boxes
   are the ego car, not the road.** Without the ego-car boxes the road-area split is P3 3, P4 7, P5 1 (P4 = 64%).
   So the drop does not mean fewer P4 phantoms; it means the "lower half = road" rule also counts the car's own hood.
2. Fewer phantom candidates on clear night (30) than on rain night (41), as expected if wet roads add phantoms. With
   150 images each this is a hint only.
3. Wet-road reflections show up even in `night_clear` images (BDD's weather tag is about rain falling, not about the
   road being dry). The clear-night eval set is therefore not reflection-free.
4. Both readings give p3p4 under the fixed rule. The ego-car boxes push the score **toward p3** (they add P5 boxes to
   the denominator), so they cannot cause a wrong p3p4 decision.
5. The COCO model has never seen BDD's ego-car hood as background; BDD does not label it. Models trained on BDD
   should learn to ignore it, so these boxes say little about the trained models (same limit as below).

## Limits of this observation

- COCO model, not BDD-trained. A BDD-trained baseline will have learned BDD's backgrounds and may already ignore many
  hood/dashboard reflections. Its phantom pattern can differ.
- 150 images per reading (41 and 30 `fp_bg` boxes, about 2 clear road reflections each). Far too few to decide on their own.
- "Lower half of the image" is a rough stand-in for "road"; no reflective areas were hand-marked.
- The by-eye categories were judged by one look at each picture, not by a careful labelling protocol.

## To confirm (before the long training runs)

1. Train the pilot baseline (A0, about 10k images, about 15 epochs; see the plan in chat, 2026-11-15).
2. Run `phantom_levels.py` with that `best.pt` on the **full** rain-night val, with `--save-vis 40`.
3. Hand-mark reflective road areas on 200-300 rain-night val images (marked before looking at predictions), rerun
   with `--regions`, and count road reflections and in-car reflections separately.
4. Decide: if most road-reflection phantoms are P4 (or P3 + P4), build the P3 + P4 gate. Make this decision from the
   baseline only, never from comparing gate results on the eval sets.

## Files

- `2026-11-15-phantom-levels-first-check/road_area_false_positives.jpg`: the 20 road-area `fp_bg` boxes
- `2026-11-15-phantom-levels-first-check/clear_night_road_area_false_positives.jpg`: the 19 road-area `fp_bg` boxes of
  the second reading, labelled with level and what each one is
- `2026-11-15-phantom-levels-first-check/summary.json`: per-level counts (default branch)
- `2026-11-15-phantom-levels-first-check/boxes.csv`: every kept box with level, class, score, IoU and label
