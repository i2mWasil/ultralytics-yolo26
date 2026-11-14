#!/usr/bin/env bash
# The RGATE pipeline on one GPU PC: TWO training runs, base + one gate design. Full guide: rgate/PIPELINE.md.
#
# Steps (given as arguments, run in that order):
#   check    level check on the night training set (rgate/level_check.py): which level do phantom candidates come
#            from? -> runs/rgate/level_check/.../decision.json says p3 or p3p4. No training. Fixed rule.
#   base     train YOLO26n (stock)                       run: base_<scale>_<train set>_s<seed>
#   p3       train YOLO26n + reflection gate on P3       run: gate-p3-l0.05_<scale>_<train set>_s<seed>
#                                                        model: rgate/yolo26-rgate-p3.yaml
#   p3p4     train YOLO26n + reflection gates P3 + P4    run: gate-p3p4-l0.05_<scale>_<train set>_s<seed>
#                                                        model: rgate/yolo26-rgate-p3p4.yaml
#   compare  side-by-side report of the finished runs (rgate/compare_runs.py), incl. phantom counts per level
# No arguments = check base. Then train the gate design the check chose:  bash rgate/run_all.sh p3   (or p3p4)
#
# Guards (the plan is base + ONE gate design, chosen by the check's fixed rule):
#   p3 / p3p4 stop if the check has not been run, if its decision is the other design, or if a run of the other
#   gate design already exists. OVERRIDE_DECISION=1 skips these guards (then say so in the thesis).
#
# Usage (repo root or anywhere; Linux or WSL2):
#   DATA=~/data TRAIN_SET=bdd_train_night EVAL_SETS="bdd_val_night_rain bdd_val_night bdd_val_all" bash rgate/run_all.sh
#   The FIRST name in EVAL_SETS is the main eval set (used by compare for the phantom counts).
# Optional: PROJECT=runs/rgate EPOCHS=50 BATCH=32 IMGSZ=640 SEEDS="0" WORKERS=<CPU cores - 1> WEIGHTS=yolo26n.pt
#           LAM=0.05 CHECK_WEIGHTS=yolo26n.pt CHECK_SET=<TRAIN_SET> EXTRA_FLAGS="..."
# Safe to start again at any time: finished runs (done.json) are skipped, an interrupted run resumes from last.pt,
# an existing level check is reused.
set -u
cd "$(dirname "$0")/.." || exit 1

DATA=${DATA:?set DATA to the folder that holds the dataset folders}
TRAIN_SET=${TRAIN_SET:?set TRAIN_SET to the training dataset folder name inside DATA}
EVAL_SETS=${EVAL_SETS:?set EVAL_SETS to the eval dataset folder names inside DATA, space-separated, main set first}
PROJECT=${PROJECT:-runs/rgate}
EPOCHS=${EPOCHS:-50}
BATCH=${BATCH:-32}
IMGSZ=${IMGSZ:-640}
SEEDS=${SEEDS:-"0"}
WEIGHTS=${WEIGHTS:-yolo26n.pt}
SCALE=${SCALE:-n}            # must match WEIGHTS (yolo26n.pt -> n); used only to find run folders
WORKERS=${WORKERS:-}         # empty = train_rgate.py default, min(8, CPUs)
LAM=${LAM:-0.05}
CHECK_WEIGHTS=${CHECK_WEIGHTS:-yolo26n.pt}
CHECK_SET=${CHECK_SET:-$TRAIN_SET}
OVERRIDE_DECISION=${OVERRIDE_DECISION:-0}
EXTRA_FLAGS=${EXTRA_FLAGS:-}
CHECK_EXTRA=${CHECK_EXTRA:-}  # extra flags for level_check.py (testing only, e.g. --limit)

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(check base)

EVAL_ARGS=""
for e in $EVAL_SETS; do EVAL_ARGS+=" $DATA/$e"; done
MAIN_EVAL=${EVAL_SETS%% *}
WORKER_FLAG=""
[ -n "$WORKERS" ] && WORKER_FLAG="--workers $WORKERS"
LAM_TAG=$(python3 -c "print(float('$LAM'))")
CHECK_DIR="$PROJECT/level_check/$(basename "${CHECK_WEIGHTS%.*}")_${CHECK_SET}"
mkdir -p "$PROJECT"
FAILED=0

# only one run_all.sh per PROJECT: two copies would train the same runs at the same time and mix their files
if command -v flock >/dev/null 2>&1; then
  exec 9>"$PROJECT/.run_all.lock"
  flock -n 9 || { echo "[run_all] another run_all.sh is already running for $PROJECT (see: ps aux | grep run_all). Not starting."; exit 1; }
fi

do_check() {
  echo "[run_all] $(date '+%F %T') level check: $CHECK_WEIGHTS on $DATA/$CHECK_SET"
  python rgate/level_check.py --data "$DATA/$CHECK_SET" --weights "$CHECK_WEIGHTS" --out "$CHECK_DIR" $CHECK_EXTRA \
    || { echo "[run_all] level check FAILED"; return 1; }
}

gate_allowed() {  # $1 = p3 | p3p4; returns 1 (and explains) if the plan's guards forbid training it
  local want=$1 other decided
  [ "$want" = p3 ] && other=p3p4 || other=p3
  [ "$OVERRIDE_DECISION" = 1 ] && { echo "[run_all] OVERRIDE_DECISION=1: guards skipped for $want"; return 0; }
  if [ ! -f "$CHECK_DIR/decision.json" ]; then
    echo "[run_all] no level check result ($CHECK_DIR/decision.json). Run: bash rgate/run_all.sh check"
    return 1
  fi
  decided=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['decision'])" "$CHECK_DIR/decision.json")
  if [ "$decided" != "$want" ]; then
    echo "[run_all] the level check decided '$decided', not '$want'. Run: bash rgate/run_all.sh $decided"
    echo "[run_all] (to train '$want' anyway: OVERRIDE_DECISION=1, and report it)"
    return 1
  fi
  if ls -d "$PROJECT"/gate-"$other"-l*_"${SCALE}"_"${TRAIN_SET}"_s* >/dev/null 2>&1; then
    echo "[run_all] a '$other' gate run already exists in $PROJECT; the plan trains one gate design only."
    echo "[run_all] (to train '$want' as well: OVERRIDE_DECISION=1, and report it)"
    return 1
  fi
  return 0
}

run_one() {  # $1 = run-name tag (must match how train_rgate.py names runs), $2 = arm flags, $3 = seed
  local tag=$1 flags=$2 seed=$3
  local name="${tag}_${SCALE}_${TRAIN_SET}_s${seed}"
  local dir="$PROJECT/$name"
  local base="python rgate/train_rgate.py $flags --data $DATA/$TRAIN_SET --eval$EVAL_ARGS --project $PROJECT $EXTRA_FLAGS"
  local cmd
  if [ -f "$dir/done.json" ]; then
    echo "[run_all] $name: finished, skipping"
    return 0
  elif [ -f "$dir/weights/last.pt" ]; then
    echo "[run_all] $name: unfinished, resuming"
    cmd="$base --resume $dir/weights/last.pt $WORKER_FLAG"
  else
    if [ -d "$dir" ]; then  # started but died before saving any epoch: move it aside so the name stays free
      mv "$dir" "$dir.failed-$(date +%Y%m%d-%H%M%S)"
      echo "[run_all] $name: moved an empty failed start aside"
    fi
    cmd="$base --weights $WEIGHTS --epochs $EPOCHS --imgsz $IMGSZ --batch $BATCH --seed $seed $WORKER_FLAG"
  fi
  echo "[run_all] $(date '+%F %T') start $name"
  echo "[run_all] $cmd"
  $cmd
  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "[run_all] $(date '+%F %T') done $name"
  else
    echo "[run_all] $(date '+%F %T') FAILED $name (exit $rc)"
    FAILED=1
    return 1
  fi
}

train_pipeline() {  # $1 = base | p3 | p3p4
  local seed
  for seed in $SEEDS; do
    case "$1" in
      base) run_one base "--arm baseline" "$seed" ;;
      p3|p3p4) run_one "gate-$1-l${LAM_TAG}" "--arm gate --lam $LAM --gate-levels $1" "$seed" ;;
    esac
  done
}

for step in "${STEPS[@]}"; do
  case "$step" in
    check) do_check || exit 1 ;;
    base) train_pipeline base ;;
    p3|p3p4) gate_allowed "$step" || exit 1; train_pipeline "$step" ;;
    compare)
      for seed in $SEEDS; do
        python rgate/compare_runs.py --project "$PROJECT" --train-set "$TRAIN_SET" --scale "$SCALE" --seed "$seed" \
          --lam "$LAM" --main-eval "$DATA/$MAIN_EVAL" || echo "[run_all] compare FAILED for seed $seed"
      done
      ;;
    *) echo "[run_all] unknown step '$step' (use: check base p3 p3p4 compare)"; exit 1 ;;
  esac
done
echo "[run_all] $(date '+%F %T') steps finished: ${STEPS[*]}. Finished runs:"
ls -d "$PROJECT"/*/done.json 2>/dev/null | sed 's#/done.json##'
if [ -f "$CHECK_DIR/decision.json" ]; then
  echo "[run_all] level check decision: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['decision'], '-', d['reason'])" "$CHECK_DIR/decision.json")"
fi
exit $FAILED
