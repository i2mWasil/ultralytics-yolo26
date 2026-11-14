# RGATE project context — YOLO26 Reflection Gate for rainy-night detection

Context and working rules for this project. Read fully before changing code.
Repo: `~/Documents/GitHub/ultralytics-yolo26` = a fork of Ultralytics **8.4.212** with the RGATE patch applied.
The repo root `CLAUDE.md` is Ultralytics' own guide (a symlink to `AGENTS.md`); this file is the project guide.
Facts marked **[verified]** were checked against Ultralytics 8.4.212 source (torch 2.15) by building the models,
training, and exporting to ONNX. Re-check them if the Ultralytics version ever changes.

**Plan (2026-11-15, decision log #21, #22; run guide: `rgate/PIPELINE.md`):** on the RTX 4070 PC, one seed,
night-only data, **two training runs**. A. **Level check before training** (`rgate/level_check.py`): COCO `yolo26n.pt`
over the whole night training set, fixed rule "P4 share of road-area phantom candidates ≥ 0.25 → `p3p4`, else `p3`".
B. **base**: stock YOLO26n. C. **one gate design**, the one the check chose: `p3` (gate on P3,
`yolo26-rgate-p3.yaml`) or `p3p4` (gates on P3 and P4, `yolo26-rgate-p3p4.yaml`), λ = 0.05, otherwise identical
to base. D. **Compare** (`rgate/compare_runs.py`). All run by `rgate/run_all.sh check | base | p3 | p3p4 | compare`. 50 epochs, 640 px, batch 32, seed 0. Training set = BDD100K **train**, night only (~28k);
eval sets from BDD100K **val** (§6b). Background: `observation/2026-11-15-training-plan.md` (data, compute, seeds).

**History (2026-11-14):** the training pipeline works end to end on the 150-image samples in `dataset/`.
[verified] on the samples (Linux, CPU, Ultralytics 8.4.212, torch 2.15): every arm trains and evaluates; `--resume`
after a kill, keep-every-best, `done.json`, and the resume guards (wrong arm/λ, changed split) work.
CUDA [verified on Colab T4, torch 2.14.1, 2026-11-14]: after the AMP fix (#11) a 50-epoch gate λ=0.05 run on the
samples trained with AMP, passed the mosaic switch-off at epoch 41, kept every best, evaluated and wrote `done.json`
(6.4 GB GPU memory at batch 32, 640 px). Not yet on GPU: `--resume`, the baseline and no-prior arms.
Apple MPS [verified on M1, torch 2.15, Python 3.13, 2026-11-15]: 10-epoch gate run on the samples, batch 8, 640 px,
~14 s/epoch, 3.3 GB; finished with evals and `done.json`. AMP is off on MPS. MPS prints a warning that
`index_put_with_accumulate` has no deterministic version, so Mac runs are not bit-reproducible (harmless; real runs are CUDA).
Mac setup: `python3.13 -m venv .venv`, then `pip install -e . -r rgate/requirements.txt`.

**2026-11-15 changes (decision log #13–#16):** gate factor bounded with tanh, aux-loss object mask uses cell overlap,
prior epsilon made FP16-safe, and a new diagnostic `rgate/phantom_levels.py` (which level P3/P4/P5 the false positives
come from). [verified on Linux CPU, Ultralytics 8.4.212, torch 2.15]: unit tests of all three fixes, step-0 output
identical to stock (max diff 0.0), `remapped 708` unchanged, smoke run gate λ=0.05 trains/evaluates/writes `done.json`,
ONNX export matches PyTorch (boxes < 0.01 px, scores < 4e-6, no NaN on black pixels). **Runs made before this change
are not comparable with runs after it: train every arm with the new code.**
**2026-11-15, later (decision log #17): second design: TWO gates, on P3 and P4** (now named `p3p4`,
`yolo26-rgate-p3p4.yaml`, renamed from `yolo26-rgate.yaml` by #22: gates are layers 23 and 24, Detect is 25). P5 is
not gated. The first design (P3 only, now named `p3`) stays available with `--gate-levels p3`
(`yolo26-rgate-p3.yaml`). Which one is trained is decided by the level check (#21, #22). [verified on Linux CPU, Ultralytics 8.4.212, torch 2.15]: step-0 output identical to stock
for both designs, `remapped 708` + 16 fresh gate tensors, smoke runs (λ=0.05, `--no-prior`, `--gate-levels p3`),
kill + `--resume` mid-run, resume guard on wrong `--gate-levels`, ONNX export with both gates active (boxes < 0.01 px,
scores < 1e-5), n and s scales, 640x384 rectangular input.
**2026-11-15, later (decision log #20): class-head start fixed.** The gate arm started with an untrained class layer
while the baseline got 6 of its 10 class rows copied from COCO by name (Ultralytics' default `cls_remap=True`), so every
earlier gate-vs-baseline comparison was unfair (all gate runs before this fix, incl. the Colab and sample runs, are not
comparable). Now `HYP` sets `cls_remap=False`: **both arms start with a fresh class-output layer**; everything else
transfers. Details and evidence: `observation/2026-11-15-class-head-start.md`.
Next step: on the RTX 4070 PC follow `rgate/PIPELINE.md` (setup, smoke test, step A level check, base run, then the chosen
gate run, compare).
**2026-11-15, later (decision log #22): names and plan.** Designs are named `p3` and `p3p4` everywhere (YAML files,
`--gate-levels`, run names `gate-p3-l0.05_...` / `gate-p3p4-l0.05_...`, `run_all.sh` steps). Only two runs are trained:
base + the design the check chose. `--gate-levels` is now required for gate runs.

---

## 1. Goal in one paragraph

On wet roads at night, headlights, tail lights and street lights reflect off the road. YOLO detects some of these
reflections as extra objects ("phantom" boxes). We add one small learned block, the **Reflection Gate (RGATE)**, to YOLO26.
It looks at the P3 feature map (design `p3`) or the P3 and P4 maps (design `p3p4`, one gate each) plus a cheap "bright and colourless" map computed from the input
image, and learns to turn down features in cells that look like reflections. Everything else in YOLO26 stays stock.

**Hypothesis to test (not assume):** on rain-night images, RGATE lowers the phantom rate and raises precision,
without hurting recall or clear-night accuracy, at about +2% (`p3`) or +4% (`p3p4`) compute (YOLO26n). Both models are trained
on night images only (decision #18), so the claim is about night driving; day images are only a side check.

**Thesis-level success =** a clean comparison of the base run with the gate run where the only difference
is the gate, with a gain clearly larger than the noise the plan can measure (§7). One seed cannot show seed-to-seed noise or that
the prior (not just extra parameters) causes a gain; those need extra seeds and the A3 control (§7, "if more time").
A negative result reported honestly is still a valid thesis; a gain from an unfair comparison is not.

## 2. Hardware and software

| Item | Role |
|---|---|
| MacBook Air (Apple Silicon) | Code editing and `--smoke` tests on the samples. Device `mps` or `cpu`. |
| PC with RTX 4070 (12 GB), Linux or WSL2 | **All real training** (§6d). Same script and flags, full `--data`, no `--smoke`. |
| Google Colab (T4) | Backup only (§6c). |
| Jetson Orin Nano | Main deployment and speed tests (TensorRT FP16, later INT8). |
| Jetson Nano (original, JetPack 4.x, TensorRT 8.2) | Low-end deployment target. Build engines on the device from ONNX. |
| Ultralytics | This repo (8.4.212, patched). `train_rgate.py` always imports the repo copy, not a pip copy. |
| Dataset | BDD100K detection, 10 classes, YOLO txt labels, prepared by the user in the sample format (§6b). |

## 3. Stock YOLO26 facts [verified]

- Config: `ultralytics/cfg/models/26/yolo26.yaml`. Scales n/s/m/l/x; `end2end: True`, `reg_max: 1` (no DFL bins).
- **Detect head has two branches.** `one2many` (many boxes per object) and `one2one` (one box per object, no NMS).
  In training the one2one branch gets **detached** features (`head.py`: "detach keeps one2one out of the backbone").
  So **the backbone, neck and our gate learn only from the one2many loss (+ our aux loss).**
- **Which branch is used at val/predict** [verified 2026-11-15]: in 8.4.212 the default `nms=None` means
  "external NMS": `val()`, `predict()` and the val during training (which picks `best.pt`) all use **one2many + NMS**
  (`engine/validator.py`: `end2end = args.nms is False`). The NMS-free one2one head is used only with `nms=False`.
  Predict defaults: conf 0.25, NMS IoU 0.5; val: conf 0.001, IoU 0.7. So every number `train_rgate.py` reports comes
  from the one2many branch, the same branch the gate learns from. If you deploy NMS-free (`nms=False`), evaluate that
  way too and say so.
- **Loss = `E2ELoss`** (`utils/loss.py`): `o2m_weight * L_one2many + o2o_weight * L_one2one`. `o2m_weight` decays
  linearly from 0.8 to 0.1 over training (this is what the docs call ProgLoss). Consequence: late in training the gate's
  learning signal from detection loss shrinks to ~10%.
- **STAL** (`utils/tal.py`): during label assignment, any GT box side smaller than `stride_val = 16` px is enlarged to
  16 px so tiny objects still get assigned anchors. It does not depend on brightness.
- Total detection loss is multiplied by batch size; any extra loss must be scaled the same way.
- Inference output (stock and RGATE identical): raw `(B, 4+nc, 8400)`; in end-to-end mode `(B, 300, 6)` =
  `[x1, y1, x2, y2, score, class]`. 8400 = 80² + 40² + 20² grid cells.
- Layer 16 (P3 output) feeds **two** places: the Detect head **and** layer 17 (the PAN downsample). Layer 19 (P4
  output) also feeds two places: Detect **and** layer 20. Important for placement: head-only gates change only what
  Detect sees; PAN still reads the ungated 16 and 19.

## 4. Full data flow with RGATE (640×640 input) [verified shapes]

Shapes are `channels × height × width`, batch dim omitted. n / s = channel counts for YOLO26n / YOLO26s.

| # | from | module | input (n) | output n | output s | what it does |
|---|---|---|---|---|---|---|
| in | – | image | – | 3×640×640 | 3×640×640 | letterboxed RGB, values 0–1 |
| 0 | -1 | Conv s2 | 3×640×640 | 16×320×320 | 32×320×320 | P1, halve size |
| 1 | -1 | Conv s2 | 16×320×320 | 32×160×160 | 64×160×160 | P2 |
| 2 | -1 | C3k2 | 32×160×160 | 64×160×160 | 128×160×160 | feature mixing |
| 3 | -1 | Conv s2 | 64×160×160 | 64×80×80 | 128×80×80 | P3 resolution (stride 8) |
| 4 | -1 | C3k2 | 64×80×80 | 128×80×80 | 256×80×80 | **backbone P3** (saved) |
| 5 | -1 | Conv s2 | 128×80×80 | 128×40×40 | 256×40×40 | P4 resolution (stride 16) |
| 6 | -1 | C3k2 | 128×40×40 | 128×40×40 | 256×40×40 | **backbone P4** (saved) |
| 7 | -1 | Conv s2 | 128×40×40 | 256×20×20 | 512×20×20 | P5 resolution (stride 32) |
| 8 | -1 | C3k2 | 256×20×20 | 256×20×20 | 512×20×20 | |
| 9 | -1 | SPPF | 256×20×20 | 256×20×20 | 512×20×20 | wide context (pooling) |
| 10 | -1 | C2PSA | 256×20×20 | 256×20×20 | 512×20×20 | self-attention, **backbone P5** |
| 11 | -1 | Upsample×2 | 256×20×20 | 256×40×40 | 512×40×40 | FPN top-down starts |
| 12 | -1, 6 | Concat | 256 + 128 @40 | 384×40×40 | 768×40×40 | join with backbone P4 |
| 13 | -1 | C3k2 | 384×40×40 | 128×40×40 | 256×40×40 | top-down P4 |
| 14 | -1 | Upsample×2 | 128×40×40 | 128×80×80 | 256×80×80 | |
| 15 | -1, 4 | Concat | 128 + 128 @80 | 256×80×80 | 512×80×80 | join with backbone P3 |
| 16 | -1 | C3k2 | 256×80×80 | **64×80×80** | **128×80×80** | **neck P3 = F** (to gate and to 17) |
| 17 | -1 | Conv s2 | 64×80×80 | 64×40×40 | 128×40×40 | PAN bottom-up starts (reads **ungated** 16) |
| 18 | -1, 13 | Concat | 64 + 128 @40 | 192×40×40 | 384×40×40 | |
| 19 | -1 | C3k2 | 192×40×40 | 128×40×40 | 256×40×40 | **neck P4** (to Detect) |
| 20 | -1 | Conv s2 | 128×40×40 | 128×20×20 | 256×20×20 | |
| 21 | -1, 10 | Concat | 128 + 256 @20 | 384×20×20 | 768×20×20 | |
| 22 | -1 | C3k2 | 384×20×20 | 256×20×20 | 512×20×20 | **neck P5** (to Detect) |
| **23** | 16 (+image) | **ReflectionGate** (P3, stride 8) | 64×80×80 + image | **64×80×80** | **128×80×80** | **new**: gated P3 |
| **24** | 19 (+image) | **ReflectionGate** (P4, stride 16) | 128×40×40 + image | **128×40×40** | **256×40×40** | **new**: gated P4 |
| 25 | 23, 24, 22 | Detect | [64×80², 128×40², 256×20²] | 300×6 (e2e) | 300×6 | unchanged head |

Stock YOLO26 is identical except it has no layers 23–24 and Detect is layer 23 reading `[16, 19, 22]`.
The table shows design `p3p4` (`--gate-levels p3p4`, `yolo26-rgate-p3p4.yaml`).
Design `p3` (`--gate-levels p3`, `yolo26-rgate-p3.yaml`): only layer 23, Detect is 24 reading `[23, 19, 22]`.

### Inside one gate (layer 23 = P3, layer 24 = P4), YOLO26n numbers (P4 in brackets)

Each gate has its own weights and its own γ. YAML args: `[hidden, v0, stride, gamma_gain, use_prior]`; the P4 gate is
`[null, 0.6, 16]`. The prior stride must equal the gated layer's stride; the gate raises a clear error if not.

| step | operation | input | output | learned? |
|---|---|---|---|---|
| a | `V = max(r,g,b)`, `Sat = (max−min)/max(V, 1e-3)` | image 3×640×640 | 2 maps 1×640×640 | no |
| b | `R = clip((V−0.6)/0.4, 0, 1) · (1−Sat)` | V, Sat | 1×640×640 | no |
| c | max-pool 8×8, stride 8 [16×16, stride 16] | 1×640×640 | **R: 1×80×80** [1×40×40] | no |
| d | concat(F, R) | 64×80×80 [128×40×40] + R | 65×80×80 [129×40×40] | – |
| e | Conv3×3 + BN + SiLU | 65×80×80 | 16×80×80 [32×40×40] (`hidden = max(16, C/4)`) | yes |
| f | Conv3×3 + sigmoid | hidden map | **S: 1×80×80** [1×40×40], values 0–1 | yes |
| g | `F_out = F · (1 − tanh(10·γ)·S)` | F, S, γ | **64×80×80** [128×40×40] | γ yes |

- S near 1 = "reflection here, turn down". γ is one learnable number, starts at **0**, so at step 0 the model is
  exactly stock YOLO26 [verified: max output difference 0.0 after loading stock weights; re-verified with tanh].
- `strength = tanh(10·γ)` is in (−1, 1), so the factor stays in (0, 2): at most full suppression, at most double
  strength, never a sign flip (decision #13). tanh has slope 1 at 0, so the start of training is unchanged
  [verified: same γ gradient at γ=0 as the old formula]. After export γ is a constant, so ONNX folds tanh away.
- The factor 10 (`gamma_gain`) exists because γ's gradient is tiny (~3e-4). Without it γ moved only 0.0004 in 80 SGD
  steps; with it 0.029 [verified on toy data]. Full BDD100K training runs far more steps, but log γ every epoch anyway
  (`gamma.json`: `gamma_eff` = 10·γ and `strength` = tanh of it).
- While γ = 0 the mask convs get no gradient from the detection loss (∂F_out/∂S = −strength·F = 0); they start
  learning only as γ moves. The aux loss (λ > 0) trains them directly from step 1. So an arm without aux can stay near
  stock for a while: check `gamma.json` in the first epochs.
- Optimizer: `HYP` fixes MuSGD, lr 0.01, momentum 0.9, warmup_bias_lr 0 (decision #19) = what `optimizer=auto` picks
  when the run has > 10k iterations (counted as images / 64 × epochs; the full night run has ~20k). Before, auto gave
  small runs (smoke, samples, a pilot) AdamW, so they did not test the real setup [verified in `engine/trainer.py`,
  8.4.212; smoke runs now log `MuSGD(lr=0.01, momentum=0.9)`]. γ is 1-D, so it is not orthogonalised by Muon and sits
  in the "weight with decay" group (decay 5e-4).
- Mask bias starts at −2, so S starts ≈ 0.12 everywhere.
- Cost [verified with Ultralytics `get_flops`, 640 px]:

  | | P3 gate params | P4 gate params | GFLOPs stock → P3 only → P3+P4 |
  |---|---|---|---|
  | n | 9,538 | 37,506 | 6.24 → 6.36 (+2.0%) → 6.48 (+3.9%); +47k params (+1.8%) |
  | s | 37,506 | 148,738 | 22.78 → 23.74 (+4.2%) with P3+P4; +186k params (+1.9%) |
- Why max-pool not average: a 2–3 px glint would be averaged away inside an 8×8 (or 16×16) cell.
- Why the 0.6 threshold: letterbox padding is grey (V = 0.447, Sat = 0); plain `V·(1−Sat)` would mark all padding as
  "reflection". Dim grey road is also not a reflection.

### Optional auxiliary loss (variant B, `rgate_lambda > 0`)

```
B      = 1 where a cell (of that gate's grid) overlaps any GT box, else 0   (batch labels; decision #14)
target = R · (1 − B)                        # reflection-looking AND no object  -> S should be high
weight = 1 where B = 1 or R > tau (0.3)     # cells that are neither are ignored
L_gate = weighted mean of BCE(S, target)       computed as BCE-with-logits on the gate's pre-sigmoid output
         with two gates: computed per gate on its own grid (80×80 / 40×40), then the MEAN of the two
L_total = E2ELoss + lambda · L_gate · batch_size        lambda = 0.05 to start
```
Using the mean keeps λ = 0.05 the same total aux weight as in the P3-only design.
Logged automatically as `train/gate` and `val/gate` [verified]. The gate keeps `last_logit` (pre-sigmoid) for this loss
and `last_s` (= sigmoid) for visualisation. Plain `F.binary_cross_entropy` on S is refused by PyTorch under AMP.

## 5. Decision log — corrections to the earlier design notes

| # | Earlier note said | Problem | Decision |
|---|---|---|---|
| 1 | Add the gate with one YAML line `[-1, 1, ReflectionGate, [64]]` | Layers only receive earlier layer outputs; the **input image is not reachable** in the graph, and `parse_model` cannot size a custom layer | 4 small edits in `nn/tasks.py` (see §6). Unavoidable. |
| 2 | Insert gate right after layer 16 | Layer 16 also feeds PAN (17). Inserting in the chain shifts all indices and gates PAN too | v1 = **head-only**: gate is layer 23, only Detect reads it. "Gate before PAN" is a later variant. |
| 3 | `F + γ(F⊙G)`, aux loss penalises G open where no box | One-sided loss has a trivial solution: G = 0 everywhere. Also with γ > 0, G high means *amplify*, the opposite of the story | `F·(1 − γ·S)`, S = suppression map, two-sided aux loss (§4). |
| 4 | Prior = `V·(1−S)` + mirror-symmetry NCC around horizon | Mosaic, scale, translate augmentation break any fixed horizon; horizon unknown per image; NCC is slow on Nano; grey padding fires | v1 uses the **thresholded brightness × colourless prior only**. Mirror term = optional later ablation. |
| 5 | "Taillight stays red, reflection is white" | True for tail lights only. Headlights and many street lights are white too, so the prior also fires on **real** lights | The prior is a hint, not a label. The learned conv (and aux loss using GT boxes) must separate real from reflected. Expect this to be the hardest part. |
| 6 | "YOLO26 is NMS-free so duplicates are now worse" | NMS removes boxes that **overlap** a better box. A road reflection sits below the car with little overlap; NMS would mostly not have removed it either | Do not use this as the main motivation. Motivate with measured phantom rates on your own data. |
| 7 | Gate learns from the detection loss | One2one (inference) branch is detached; gate learns only from one2many, whose weight decays 0.8 → 0.1 | Keep variant A (no aux) and B (aux) both; aux matters more late in training. |
| 8 | `YOLO('yolo26-rgate-p3p4.yaml').load('best.pt')` | Detect moves from index 23 to 25 (24 in design `p3`), so keys don't match and **all head weights are silently dropped** | `train_rgate.py` remaps in memory (log line `remapped 708 stock tensors`, then 16 fresh gate tensors; 8 for P3 only). Never call `.load()` on a stock checkpoint by hand. |
| 9 | Default augmentation | `hsv_s=0.7`, `hsv_v=0.4` randomly change saturation/brightness a lot — exactly what the prior reads | Use `hsv_s=0.3`, `hsv_v=0.2` for **all** arms, baseline included. |
| 10 | "2,494 Night-Rainy images" | Number from another paper's split | Count from your own data. The samples were built from BDD100K `subsets/night_rainy.txt` and `night_clear.txt` (val split). |
| 11 | Aux loss = `F.binary_cross_entropy(S.clamp(...), target)` | On CUDA with AMP (`amp=True`, Ultralytics default) PyTorch raises "unsafe to autocast" on the first batch. CPU/Mac runs have no AMP, so CPU tests did not catch it (2026-11-14, first Colab smoke run) | `F.binary_cross_entropy_with_logits(logit, target)`. Same value in fp32 [verified: identical to 7 digits], AMP-safe, more stable. |
| 12 | Colab cells used `!python ...` | `!` does not stop the notebook when the command fails; a crash looked like success and the resume cell then failed confusingly | Notebook runs commands with `sh()`, which raises on a non-zero exit code. |
| 13 | `F_out = F · (1 − 10·γ·S)` | Unbounded: when 10γS > 1 the factor goes negative and flips the sign of the features; a negative γ boosts without limit | `F · (1 − tanh(10·γ)·S)`: factor in (0, 2). Same start (identity at γ=0, same gradient). 2026-11-15. |
| 14 | Aux mask B = "cell centre inside a GT box" | A box narrower than one 8 px cell can miss every centre, so small objects (distant signs, headlights) got target "suppress" [verified: a 4 px box marked 0 cells] | B = 1 for every cell the box **overlaps** (box-aligned cells only, no extra ring). Only λ > 0 arms change. 2026-11-15. |
| 15 | `Sat = (max−min)/(max + 1e-6)` | 1e-6 is below FP16's normal range; if TensorRT rounds it to 0, black pixels give 0/0 = NaN that spreads through the conv | `/ max.clamp(min=1e-3)`. Same values in fp32 (max diff 7e-7), no NaN in fp16 [verified]. 2026-11-15. |
| 16 | "one2one is used at inference" (§3, #7) | In 8.4.212 the default `nms=None` makes val/predict/best.pt selection use **one2many + NMS** [verified] | Report numbers from that default branch; `phantom_levels.py` uses it too (`--nms-free` for the other). The gate learns from and is scored on the same branch. 2026-11-15. |
| 17 | v1 = head-only gate on P3 only (#2) | First level check (`observation/2026-11-15-phantom-levels-first-check.md`, COCO model, 150 images): 59% of background false positives and 17 of 20 in the road area came from P4, which a P3 gate cannot change. P5: 7% only, and a 32 px cell would also suppress large real objects | Main design = **two head-only gates, P3 and P4** (separate weights and γ each; aux loss = mean over gates). P5 not gated. P3-only kept as `--gate-levels p3`. Preferred over gate-before-PAN because P4 also takes ungated top-down features (layer 13), so a P3 gate before PAN cleans P4 only partly, and it would change every level. Evidence is thin: confirm on the BDD-trained baseline before the long runs. 2026-11-15. |
| 18 | Train on all of BDD100K; full protocol 4 arms × 3 seeds (12 runs) | All-data training costs ~2.5× per epoch, and in day images the prior fires on sky, white buildings and sunlit road, so the aux loss would teach the gate a day-background task unrelated to reflections. The user chose the minimum plan (2 runs) | Train on BDD100K **train, night only** (~28k; no dawn/dusk; never val images). Claim scoped to night driving. **Minimum plan: A0 + A2, seed 0** (`observation/2026-11-15-training-plan.md`). Rain-night differences under ~1 mAP are reported as inconclusive. 2026-11-15. |
| 19 | `optimizer=auto` | Picks AdamW for runs ≤ 10k iterations and MuSGD above, so smoke/sample/pilot runs used a different optimizer from the real run | `HYP`: optimizer MuSGD, lr0 0.01, momentum 0.9, warmup_bias_lr 0.0 (= auto's choice for the real run), for every run and arm. 2026-11-15. |
| 20 | "Official `yolo26n.pt` works: the 12 class-output tensors restart for 10 classes" | Wrong in 8.4.212: by default (`cls_remap=True`) the trainer copies COCO class rows **by name** into the new 10-class layer (car, person, truck, bus, train, traffic light = 6/10). The baseline got them; the gate arm did not, because `train_rgate.py` passed a YAML-built model with placeholder names "0".."9" [verified: different start weights only in those rows; gate train cls loss ~2× from batch 1; 1-epoch sample run 2,638 boxes on 10 images vs 33] | **`HYP`: `cls_remap=False`** (user's choice): both arms start with a fresh class-output layer, all other tensors transfer (baseline 696/708, gate 714/726). Also: the gate arm now hands the trainer the start weights themselves (`model.model = ...`), so the arms stay equal even if `cls_remap` is switched back on. Cost: both models start weaker (sample runs: val mAP50 after 1 epoch ~0.04-0.05 instead of ~0.12-0.14); the baseline is not Ultralytics' default fine-tune, say so in the thesis. 2026-11-15. |
| 21 | Minimum plan: A0, then a level check on the trained A0 (rain-night val), then one gate run (#17, #18) | Checking on the rain-night **val** set would tune the design on the main eval set; the user wants the placement decided before training, and both gate variants available | **A.** `rgate/level_check.py` before any training: COCO `yolo26n.pt` over the whole night **training** set; score = P4 share of road-area `fp_bg` (box centre in lower half); **≥ 0.25 → P3+P4, < 0.25 → P3, < 100 candidates → inconclusive → P3+P4**; 95% bootstrap interval, "close call" flag. Rule fixed before running. **B.** Three pipelines: 1 base, 2 gate P3 (`--gate-levels p3`), 3 gates P3+P4 (`--gate-levels p3p4`), all λ = 0.05, otherwise identical. **C.** `rgate/compare_runs.py`. Runner: `run_all.sh check|base|p3|p3p4|decided|compare|all`. Limit: the check's COCO model is not BDD-trained. 2026-11-15. (Plan and names changed by #22: two runs, designs `p3` / `p3p4`.) |
| 22 | #21: three pipelines (base, P3, P3+P4); main gate file `yolo26-rgate.yaml`, run names `gate-l0.05-p3only_...` / `gate-l0.05_...`; `--gate-levels` defaulted to P3+P4; runner steps `decided` and `all` | The user will train only **two** runs; the old names were confusing (the P3+P4 file had no level in its name, and a missing `--gate-levels` silently meant P3+P4) | **Two runs: base + the design the check chose.** Names: designs `p3` / `p3p4`; files `rgate/yolo26-rgate-p3.yaml` / `rgate/yolo26-rgate-p3p4.yaml` (old `yolo26-rgate.yaml` = a stub to delete); run names `base_...`, `gate-p3-l<λ>_...`, `gate-p3p4-l<λ>_...`; generated cfg `yolo26<scale>-rgate-<p3\|p3p4>-l<λ>.yaml`. `--gate-levels p3\|p3p4` is **required** for `--arm gate` (not for `--check-only`). `run_all.sh` steps: `check \| base \| p3 \| p3p4 \| compare`, no step = `check base`; it refuses `p3`/`p3p4` without a check result, against the check's decision, or when a run of the other design exists (`OVERRIDE_DECISION=1` skips this; report it). [verified on the samples, CPU, 2026-11-15: both YAMLs build (gates `[P3]` / `[P3, P4]`), step-0 output = stock for both incl. no-prior; the guards refuse and allow as described; `p3` and `p3p4` runs train, evaluate and compare under the new names.] 2026-11-15. |

## 6. Implementation

### Repo layout (this repo)

```
CLAUDE.md -> AGENTS.md          # Ultralytics' own guide (symlink). Do not overwrite it.
rgate/
  CLAUDE.md                     # this file
  train_rgate.py                # THE entry point: data checks, split, train, eval. All arms.
  requirements.txt              # extra packages for rgate/ (PyYAML)
  data_prep.py                  # dataset loading, checks, stable split, leak check
  yolo26-rgate-p3.yaml          # design p3: stock yolo26.yaml + gate on P3 (layer 23), Detect -> 24 (--gate-levels p3)
  yolo26-rgate-p3p4.yaml        # design p3p4: + gates on P3 (layer 23) and P4 (layer 24), Detect -> 25 (--gate-levels p3p4)
  yolo26-rgate.yaml             # OLD name of yolo26-rgate-p3p4.yaml; only a note, nothing reads it. Delete it.
  colab_train.ipynb             # Google Colab runner (backup): Drive, unzip data, check, smoke, train, resume
  PIPELINE.md                   # THE run guide: data, setup, level check, base + chosen gate run, compare, troubleshooting
  run_all.sh                    # runner: steps check | base | p3 | p3p4 | compare; guards; restart-safe (§6d)
  level_check.py                # step A: level check over the night training set + fixed p3 / p3p4 decision rule
  phantom_levels.py             # which level (P3/P4/P5) a model's false positives come from (used by A and D)
  compare_runs.py               # step D: side-by-side report of base and the gate run
observation/                    # dated observation notes (what was measured, how, limits, what it means)
ultralytics/nn/modules/rgate.py # ReflectionPrior, ReflectionGate, RGateE2ELoss
ultralytics/nn/tasks.py         # patched: 4 edits, lines marked "# RGATE"
dataset/
  sampled_bdd100k/              # 150 random BDD100K val images  (all conditions)
  sampled_bdd100k_night/        # 150 from val night_clear
  sampled_bdd100k_night_rain/   # 150 from val night_rainy
  *.zip                         # zipped copies of the three samples (the full datasets use the same layout)
runs/rgate/                     # all outputs (gitignored)
rgate_reference_code/           # OLD first draft. Superseded by rgate/. Delete it.
```

### Setup

This repo already has the patch. On the RTX 4070 PC see §6d; on Google Colab use `rgate/colab_train.ipynb` (§6c). On any fresh machine:
1. Copy or clone this repo (not pip `ultralytics`). Python 3.10–3.13 recommended; then in the repo root
   `pip install -e . -r rgate/requirements.txt` (`train_rgate.py` imports the repo copy even without `-e`, but deps must
   be installed). `rgate/requirements.txt` adds PyYAML: Ultralytics 8.4.212 no longer depends on it, but the rgate
   scripts do (a fresh Mac venv failed on `import yaml`, 2026-11-15; Colab has it preinstalled).
   Mac: `python3.13 -m venv .venv && source .venv/bin/activate` first (the venv used on the Mac, 2026-11-15).
2. Check: `python rgate/train_rgate.py --data dataset/sampled_bdd100k --check-only`.
   It prints `using Ultralytics 8.4.212 from <repo>/ultralytics` on training runs and stops with an error if the
   patch is missing.

If the patch ever has to be redone (new Ultralytics version), the 4 edits in `ultralytics/nn/tasks.py` are:
- import `ReflectionGate, RGateE2ELoss` from `ultralytics.nn.modules.rgate` at the top;
- in `BaseModel._predict_once` only: `img = x` before the loop, and
  `x = m(x, img) if getattr(m, "needs_image", False) else m(x)`;
- in `parse_model`, before the final `else`: `elif m is ReflectionGate: c1 = c2 = ch[f]; args = [c1, *args]`;
- in `DetectionModel.init_criterion`: return `RGateE2ELoss(self)` when the model contains a ReflectionGate.

### Known limits of the patch

- `profile=True` per-layer profiling calls each layer with one input; it can fail on the gate layers (23, 24). Compute the gate's
  cost by hand (§4).
- Only `BaseModel._predict_once` is patched. World/YOLOE/RT-DETR models have their own copies; not used here.

## 6b. Dataset format and training workflow [verified on the samples]

### Dataset contract — every dataset must look exactly like the samples

All datasets (the training set and every eval set) are prepared by the user **in the format of `dataset/sampled_*`**.
The repo has no BDD100K JSON converter, on purpose: the script only reads this format.

```
<dataset>/
  data.yaml        # nc: 10 + names (order below); train/val paths relative to this folder; no 'path' key needed
  images/*.jpg     # 1280x720 BDD100K frames (any size works; training resizes)
  labels/*.txt     # one per image, same stem. Lines: "class cx cy w h", normalised 0-1. Empty file = no objects.
```
The samples' `data.yaml` is `train: images`, `val: images`, `nc: 10`, `names: {0: car, ...}`; copy it for new datasets.
Also accepted: `images/train` + `images/val` (with matching `labels/...`) and `train:`/`val:` pointing at them;
or `.txt` lists of image paths. Labels are found the Ultralytics way: last `/images/` in the path -> `/labels/`.
Zipped (for Colab): `<name>.zip` holding one top folder `<name>/` (as `dataset/*.zip` do).

**Class ids (must be identical in every dataset used together; the script stops if not):**
0 car, 1 traffic sign, 2 traffic light, 3 person, 4 truck, 5 bus, 6 bike, 7 rider, 8 motor, 9 train.
This is NOT the official BDD100K order, so weights fine-tuned with another order are not compatible.

**Which images go where (for thesis numbers):**

| set | source | role |
|---|---|---|
| `bdd_train_night` | BDD100K **train**, time of day = **night** only (~28k; no dawn/dusk) | `--data`; auto-split 90/10 into train/val for picking `best.pt` |
| `bdd_val_night_rain` (**main**) | BDD100K **val**, night + rainy | `--eval`, first in the list (step C reports its phantom counts) |
| `bdd_val_night` | BDD100K **val**, night + clear | `--eval`: "does not hurt clear-night accuracy" |
| `bdd_val_all` | BDD100K **val**, everything | `--eval`: side check only (includes day; both models are night-trained) |

BDD100K has 70k train, 10k val and 20k test images; the test labels are not public. About 28k train and 4k val images
are night (count from your files). Night images from **val** must never be in the training set.

BDD100K train and val file names do not repeat, so the leak check passes without `--allow-overlap`.
The 150-image samples are all from val: use them only as eval sets or for smoke tests.
Keep large datasets out of git (`.gitignore` ignores `/datasets`, not `/dataset`).

### What the script does with a dataset

1. Reads `data.yaml`, lists images, checks every label line (5 values, class < nc, values 0-1).
   Bad line -> stops with file name. Missing label file -> warning (Ultralytics treats it as background).
2. **Split.** If `train` and `val` in `data.yaml` are the same images (true for the samples), it splits them
   90/10 by a hash of the file name (`--val-frac`). Stable: adding images never moves existing ones.
   If `train` and `val` differ, they are used as given. `--val DIR` uses a separate dataset as val.
   The split is written as `<project>/splits/<dataset>/{train.txt,val.txt,data.yaml}`; the dataset folder is not changed.
   (Ultralytics writes `labels.cache` inside `<dataset>/`. Train and val share one cache file, so it is rebuilt each run.
   Harmless; on a large dataset this costs a minute or two. If the folder is read-only it only warns.)
3. **Leak check.** Every `--eval` set is compared with the training list by file name. Any overlap stops the run,
   unless `--smoke` or `--allow-overlap`. (The samples overlap by 1 image: all three came from BDD val.)
   Never use `--allow-overlap` for a real run.
4. Builds the model: baseline = the start checkpoint; gate = `yolo26-rgate-p3.yaml` (`--gate-levels p3`) or
   `yolo26-rgate-p3p4.yaml` (`--gate-levels p3p4`; the flag is required) at the checkpoint's scale, with stock weights remapped in memory. From the official
   `yolo26n.pt` (80 classes) every tensor transfers except the class-output layers (12 tensors), which start fresh
   in **both** arms because `HYP` sets `cls_remap=False` (decision #20). The log shows no "Remapped ... cls head rows" line.
5. Writes `run_config.json` when training starts, trains with the shared `HYP` (§7), keeps a copy of every new best,
   picks `best.pt` on the split's val, evaluates `best.pt` on each `--eval` set, and writes `done.json` last.

### Commands (run from the repo root)

```bash
# check data only (seconds; on the full dataset ~1 min to read every label)
python rgate/train_rgate.py --data dataset/sampled_bdd100k --check-only

# smoke test: 1 epoch, 320 px, batch 4 (~1 min on CPU). Proves the pipeline, numbers mean nothing.
python rgate/train_rgate.py --arm gate --lam 0.05 --gate-levels p3p4 --data dataset/sampled_bdd100k \
    --eval dataset/sampled_bdd100k_night dataset/sampled_bdd100k_night_rain --smoke

# real run on the GPU PC: same command, full folders, no --smoke (normally started by run_all.sh, §6d)
python rgate/train_rgate.py --arm gate --lam 0.05 --gate-levels p3p4 --data ~/data/bdd_train_night \
    --eval ~/data/bdd_val_night_rain ~/data/bdd_val_night ~/data/bdd_val_all \
    --weights yolo26n.pt --epochs 50 --batch 32 --seed 0

# continue an interrupted run: the SAME command plus --resume <run>/weights/last.pt
python rgate/train_rgate.py ...same flags... --resume runs/rgate/<run>/weights/last.pt
```
Defaults for a new run: `--epochs 50 --imgsz 640 --batch 16 --workers min(8, CPUs)`.
Device is automatic (CUDA -> MPS -> CPU); force with `--device 0|mps|cpu`. `--batch -1` = auto batch on CUDA.

**Resume rules** [verified on the samples, CPU, 2026-11-14: killed after epoch 2 of 4, resumed, finished, evaluated]:
- Ultralytics restores epochs, hyperparameters, optimizer, imgsz and batch from `last.pt`. On the command line only
  `--batch`, `--workers`, `--imgsz` (memory/speed) are passed on, and only if you type them. `--epochs` cannot change.
- The script stops if `--arm` / `--lam` / `--no-prior` / `--gate-levels` do not match the checkpoint, or if the train/val split differs
  from the one recorded in `run_config.json` (the dataset changed). `--data` must be the same dataset.
- `gamma.json` and `results.csv` continue where they stopped. The resumed session's settings go to `run_config_resume.json`.
- If training had already finished (only the evaluation was lost), `--resume` skips training and only evaluates.

### Outputs: `<project>/<run tag>_<scale>_<dataset>_s<seed>[_smoke]/`

Run tag: `base`, `gate-p3-l<λ>` or `gate-p3p4-l<λ>` (plus `-noprior` for `--no-prior`), e.g.
`gate-p3p4-l0.05_n_bdd_train_night_s0`.

| file | content |
|---|---|
| `weights/best.pt` | best epoch so far (highest val mAP50-95, the Ultralytics "fitness"); overwritten on each new best |
| `weights/best_ep<NNN>_map50-95-<x>_map50-<y>.pt` | **a kept copy of every new best**, never overwritten. NNN = epoch (1-based), x/y = val scores. Optimizer stripped when saved (FP16, can be evaluated/exported, not resumed). |
| `weights/last.pt` | latest epoch; the file to `--resume` from. RGATE checkpoints need this patched repo to load. |
| `results.csv` | per-epoch losses (gate arms add `train/gate`, `val/gate`) and val metrics |
| `eval_summary.csv` | per eval set: precision, recall, mAP50, mAP50-95, AP50-95 per class |
| `gamma.json` | per epoch and per gate: `{"epoch": n, "P3": {"gamma_eff", "strength"}, "P4": {...}}` (gate arms) |
| `run_config.json` | written when training starts: every argument, resolved settings, split hashes, data counts per class, overlap counts, versions (`run_config_resume.json` for a resumed session) |
| `eval_<set>/` | Ultralytics val output per eval set |
| `done.json` | written last; a run without it is unfinished (the notebook's resume cell uses this) |

## 6c. Google Colab [verified: smoke + 50-epoch run on the samples, T4]

**One-time upload to Google Drive** (`MyDrive/rgate/`):
```
MyDrive/rgate/
  ultralytics-yolo26.zip      # the repo, without .git / dataset / runs (command below)
  data/<name>.zip             # one zip per dataset; inside: a single folder <name>/ with data.yaml, images/, labels/
  runs/                       # created by the notebook; all training outputs land here
```
Make the repo zip on the Mac (from `~/Documents/GitHub`):
`zip -r ultralytics-yolo26.zip ultralytics-yolo26 -x "*/.git/*" "*/.venv/*" "*/dataset/*" "*/runs/*" "*/rgate_reference_code/*" "*.DS_Store" "*/__pycache__/*"`
Re-zip and re-upload after every code change. (`.venv/` = the Mac's Python environment; it must stay out of the zip.) (Alternative: push the repo to GitHub and set `REPO_GIT_URL` in cell 1.)

**Notebook cells** (`rgate/colab_train.ipynb`): 1 settings (the only cell to edit; also defines `sh()`) -> 2 GPU check
-> 3 mount Drive -> 4 get repo -> 5 `pip install -e . -r rgate/requirements.txt` (keeps Colab's PyTorch) -> 6 copy zips to `/content/data`,
unzip, `--check-only` -> 7 smoke test -> 8 train -> 9 resume -> 10 list weights and scores.
Cells 1–6 must be rerun in every new Colab session (the VM is wiped; Drive is not).

**Rules for Colab**
- Commands run through `sh()`, which raises an error when the command fails. (`!cmd` does not stop the notebook on
  failure: on 2026-11-14 a crashed training cell looked fine and the next cell failed with a confusing message.)
- `REFRESH_REPO = True` (default) deletes `/content/ultralytics-yolo26` and takes a fresh copy, so new code is used
  after a re-upload. Cell 4 also checks the copy has the 5 RGATE markers, the AMP fix and `--resume`.
- Images are read from `/content/data` (local disk), never from Drive. Drive is slow for many small files.
- `--project` points to Drive, so `last.pt` and every kept best are on Drive after each epoch; a disconnect loses
  at most the current epoch. Free sessions end after at most ~12 h and earlier when idle.
- The split files live in `RUNS_DIR/splits/` (on Drive) and hold `/content/data/...` paths, which are the same in
  every session because cell 6 always unzips to the same place.
- Free Colab has ~2 CPU cores, so image loading, not the GPU, likely limits speed. Rough guess for ~70k images
  at 640 px on a T4: 20–40 min per epoch, so 50 epochs span several sessions. Measure epoch 1 and plan resumes.
- The 150-image samples overlap by 1 image, so sample test runs need `EXTRA = "--allow-overlap"` (or the smoke cell).

## 6d. Training on the RTX 4070 PC (main) [pipeline verified end to end on the samples, CPU; not yet run on the 4070]

**The full step-by-step guide is `rgate/PIPELINE.md`** (setup, datasets, every command, outputs, troubleshooting).
Keep it in sync with this section when anything changes.

Summary: RTX 4070 (12 GB), Linux or WSL2; the CPU (JPEG decoding + mosaic) limits speed, so `WORKERS` = cores − 1;
batch 32 fits (6.4 GB on a T4 with one gate). Setup: venv, `pip install torch torchvision`, `pip install -e . -r
rgate/requirements.txt`, check `torch.cuda.is_available()`, then `--check-only` and one `--smoke` run on that PC.

```bash
export DATA=~/data TRAIN_SET=bdd_train_night EVAL_SETS="bdd_val_night_rain bdd_val_night bdd_val_all" WORKERS=7
nohup bash rgate/run_all.sh check base >> runs/rgate/run_all.log 2>&1 &   # same as no step
# read the decision at the end of the log, then ONE of:
nohup bash rgate/run_all.sh p3 compare >> runs/rgate/run_all.log 2>&1 &
nohup bash rgate/run_all.sh p3p4 compare >> runs/rgate/run_all.log 2>&1 &
```
Steps: `check` (A), `base` (B), `p3` / `p3p4` (C, only the one the check chose), `compare` (D).
Guards: `p3` / `p3p4` are refused without `decision.json`, against its decision, or when a run of the other design
exists; `OVERRIDE_DECISION=1` skips them (report it). A gate step is also refused if a base step in the same call failed.
[verified on the samples, CPU, 2026-11-15 (decision #22): no step = check + base; the guards refuse and allow as
described; `compare` works with base + one gate run; a second start reuses the check and skips finished runs.]
Restart-safe: finished runs skipped, interrupted runs resumed from `last.pt`, check reused.
One `run_all.sh` per project folder at a time (flock). `DATA` must stay the same path across restarts.

## 7. Experiment protocol

**Arms** (all start from the same `--weights`, same epochs, same `HYP` in `train_rgate.py`, same data and seed):

| arm | command flag | purpose | in the plan |
|---|---|---|---|
| A0 baseline | `--arm baseline` | stock YOLO26 fine-tuned for the same extra epochs. **Without this, any gain may just be extra training.** | **yes: run 1 (`base`)** |
| A1 gate | `--arm gate --lam 0` | gate learns from detection loss only | no |
| A2 gate + aux | `--arm gate --lam 0.05 --gate-levels p3` / `p3p4` | adds the two-sided gate loss | **yes: run 2, `p3` or `p3p4` (the check decides)** |
| A3 control | `--arm gate --no-prior` | same gate, prior = 0. If A3 ≈ A1, the prior is not doing the work | no |

**Chosen plan (decisions #21, #22): level check first, then two runs**, one seed (0), night-only training data:
run 1 `base`, run 2 the gate design the check chose, `p3` (gate on P3) or `p3p4` (gates on P3 and P4), λ = 0.05
(`rgate/PIPELINE.md`). The other design is not trained.
What it cannot show: run-to-run noise (one seed), *why* a gain happens (no A1/A3), and whether the other gate design
would have done better (the check's choice is not tested by training). Report the result as "gate + aux vs baseline"
and name these limits. **If more time appears later**, add in this order: seeds 1 and 2 for both runs, then the other
gate design, then A3, then A1.

Run names: `base_...`, `gate-p3-l0.05_...`, `gate-p3p4-l0.05_...`. The resume check stops if `--gate-levels` does not
match the checkpoint.

Later, only if time allows: gate before PAN; mirror-symmetry prior; λ sweep {0.01, 0.05, 0.1}; s scale.

**Step A, before any training: which level do the false positives come from?** Head-only gates change only the
predictions of the levels they sit on; P5 is never gated.
```bash
python rgate/level_check.py --data ~/data/bdd_train_night        # or: bash rgate/run_all.sh check
```
`level_check.py` runs `phantom_levels.py` with the COCO `yolo26n.pt` over every image of the night **training** set
(not an eval set, so the design is not tuned on the data used to judge it) and applies the fixed rule of decision #21:
P4 share of road-area `fp_bg` ≥ 0.25 → `p3p4`, else `p3`; < 100 candidates → inconclusive → `p3p4`. Output:
`runs/rgate/level_check/yolo26n_bdd_train_night/decision.json` (decision, score, 95% bootstrap interval, counts per
level, warnings for a close call or a large P5 share) + `summary.json`, `boxes.csv`, `images.txt`, `vis/`.
[verified on the samples 2026-11-15: reproduces the first reading below exactly (road area P3 2, P4 17, P5 1); reuses
earlier results when the settings match.] `phantom_levels.py` on its own (any model, any set; `--split`,
`--regions`, `--nms-free`):
```bash
python rgate/phantom_levels.py --weights <best.pt or yolo26n.pt> --data <dataset> --save-vis 40
```
It prints, per level, true positives and false positives split into `fp_obj` (overlaps an object), `fp_loc` (near an
object) and `fp_bg` (away from every object = phantom candidates), plus phantoms inside marked regions if given.
Writes `summary.json`, `boxes.csv`, and pictures (false positives: red P3, yellow P4, blue P5; green = GT).
[verified 2026-11-15: box counts identical to `YOLO.predict()` on 40 images; works on stock and RGATE checkpoints.]

First reading (2026-11-15; official COCO `yolo26n.pt`, NOT trained on BDD, only the 150 rain-night sample images, so
treat as a hint, not a result): 41 `fp_bg` boxes = P3 34%, P4 59%, P5 7%. Of the 20 in the lower half of the image
(road area), 17 came from P4. By eye, of those 20: ~2 were clear wet-road reflections (both P4), ~6 were reflections on
the ego car's hood/dashboard/windscreen, the rest unlabelled objects, raindrops on the windscreen, or clutter.
The prior fired (max R > 0.3 inside the box) on 92% of true-positive boxes and 76% of `fp_bg` boxes, so at box level
the prior does not separate objects from phantoms; the learned conv and the aux loss must do it.
This first reading led to the P3+P4 design (decision #17); the placement is now decided by step A on the full night
training set (decision #21).

Second reading (2026-11-15; same COCO model, 150 **clear-night** sample images, `run_all.sh check` on
`sampled_bdd100k_night`): 30 `fp_bg` = P3 20%, P4 53%, P5 27%; road area P3 3, P4 9, P5 7 (score 0.47, 95% interval
0.25–0.73; decision p3p4, inconclusive: 19 < 100 candidates). The percentages differ from the first reading because the
images differ, not because anything improved. By eye, 8 of the 19 road-area boxes are the **ego car** (hood across the
bottom of the frame, dashboard in the windscreen: 6 P5, 2 P4); without them the split is P3 3, P4 7, P5 1. Only ~3 are
wet-road reflections (the `night_clear` images can still have wet roads). Details:
`observation/2026-11-15-phantom-levels-first-check.md`, section "Second reading".

**Seeds:** the rain-night val subset is small, so mAP differences under ~1 point may be noise. The plan has one
seed, so treat rain-night differences under ~1 mAP (or ~2 points of precision) as **inconclusive**, and add a bootstrap
over images (resample the eval images, recompute both models' scores, report how often A2 > A0 and a 95% interval of
the difference). That measures image-sampling noise only, not training noise. With 3 seeds: report mean ± std.

**Evaluation subsets** (from BDD100K val, same folder format as the samples, passed with `--eval`):
rain-night (main, first), clear-night, full val (side check: includes day, and both models are night-trained).
Plus ACDC rain/night as an external check if available.

**Metrics per subset:** mAP50, mAP50-95, precision, recall (all in `eval_summary.csv`), and per-class AP for
car (0), traffic sign (1), traffic light (2).

**Phantom rate (custom metric):** hand-mark reflective road regions on 200–300 rain-night val images (boxes are fine).
A *phantom* = predicted box at the deployment threshold (fix it, e.g. conf ≥ 0.25) whose centre is inside a marked
region and whose IoU with every GT box is < 0.5. Report phantoms per image and the share of all false positives that
are phantoms. Mark the regions **before** looking at any model's predictions.

**Also report:** γ_eff and strength = tanh(γ_eff) per epoch and per gate (`gamma.json`), and saved S maps for ~20 images (does S light up on reflections, or on
real lights?). The sign of γ is a result: if it goes negative, the network is using the gate to *boost* those cells,
which contradicts the hypothesis and must be reported.

**Speed:** export both A0 and the best gate arm with identical export args; measure latency/FPS/power on Orin Nano and
Nano with TensorRT FP16 (then INT8). Batch 1, same input size, warm-up runs excluded.

## 8. Deployment

- Export on the PC: `YOLO(ckpt).export(format="onnx", imgsz=640, opset=17, simplify=True)` [verified: works, ONNX vs
  PyTorch max diff 6e-5; re-verified 2026-11-15 after #13/#15: boxes < 0.01 px (values up to ~900 px), scores < 4e-6,
  no NaN on black pixels]. Gate uses only standard ONNX ops (ReduceMax/Min, Div, Clip, MaxPool, Conv, Sigmoid, Mul);
  tanh(10·γ) is a constant after export and is folded away.
- Build the TensorRT engine **on each Jetson** with `trtexec --onnx=... --fp16`. Engines are not portable between devices.
- Output shape of the RGATE export must equal the stock export's shape. If not, the export args differ.
- Not yet verified: TensorRT 8.2 on the original Nano accepting this exact graph; INT8 calibration impact on the gate.

## 9. Rules for anyone (human or AI) changing this code

1. Do not change Ultralytics outside the `# RGATE`-marked edits (4 edits, 5 lines). New logic goes in `ultralytics/nn/modules/rgate.py` or `rgate/`.
2. Never compare arms trained with different hyperparameters, epochs, starting weights, data, or Ultralytics versions.
   All arms go through `train_rgate.py`; hyperparameters live only in its `HYP` dict.
3. Never load stock weights into the RGATE graph by hand; `train_rgate.py` remaps them. Check its `remapped` log line.
4. The script must keep working unchanged when only the dataset size changes. No hard-coded paths, counts, or
   machine-specific settings; anything machine-specific is a command-line flag.
5. Before any GPU run, run `--check-only` on the real folders, then one `--smoke` run on that GPU machine.
6. Keep each gate shape-preserving: its output shape must always equal the shape of the layer it gates (16 or 19).
7. Keep everything inside the gate as plain torch tensor ops (no OpenCV, no numpy, no Python loops over pixels),
   so it exports to ONNX/TensorRT and runs on the Jetson GPU and Apple MPS.
8. Report numbers per condition subset, never only full-val mAP.
9. When unsure about an Ultralytics internal, read the installed source; do not rely on docs or memory.
   Record any new verified fact in this file with the version.
10. Write in simple English in docs and comments.
11. In the notebook, run commands with `sh()`, never `!` (see decision log #12).

## 10. Open questions

- Does S separate reflections from real headlights, or does it suppress both? (Check S maps on A2 early.)
- `p3` or `p3p4`? Decided by step A (`level_check.py`, decision #21) with a COCO model; only the chosen design is
  trained (#22), so the choice itself is not tested by training. PAN still mixes ungated P3/P4 into P5.
- Many false positives at the bottom of the frame are the ego car itself (hood across the frame, dashboard reflected
  in the windscreen), not the road: ~6 of 20 road-area candidates in the rain-night sample, 8 of 19 in the clear-night
  sample (mostly P5). The level check's "lower half = road" counts them; they push the score toward p3, so they cannot
  cause a wrong p3p4 decision. Count them separately when marking regions: they are a different failure than road
  reflections, and BDD-trained models should learn to ignore them (BDD does not label the ego car).
- Best `v0` threshold for BDD100K night images (0.6 is a starting guess; check a histogram of V on rain-night images).
- Does INT8 calibration need rain-night images in the calibration set? (Probably yes.)
- Not yet run on a CUDA GPU with the current code (p3 / p3p4 gates, MuSGD fixed, AMP): do the smoke test on the 4070 first.
  Multi-GPU never tested (not needed).
