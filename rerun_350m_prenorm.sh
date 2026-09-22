#!/usr/bin/env bash
#
# TEMPORARY. Replaces the OPT-350M results behind arXiv:2606.13993 (the verb+up
# paper) with results from the corrected pre-LN model. This is the only POSTED
# preprint affected by the post-LN bug.
#
# WHY THIS ONE IS THE SIMPLEST: the slug appears in exactly one file,
# Analyses/babylm/run_pipeline.sh, and paper/writeup.qmd names models only in
# prose ("OPT-350M"), never by HuggingFace id. Verified: 0 id occurrences in
# the writeup. So there are no writeup edits at all.
#
# THIS REPLACES IN PLACE. Outputs are keyed by TAG (Data/babylm/opt-350m/), not
# by model id, so the re-run overwrites the old CSVs directly. That is what you
# asked for, but it does mean the old numbers are gone once this completes --
# BACKUP=1 keeps a copy first if you want one for comparison.
#
# WHERE TO RUN: locally. ~1,000 items per class across 4 CSVs, one checkpoint,
# then per-layer logistic regressions on CPU. Small.
#
# Usage:  bash rerun_350m_prenorm.sh
#         DRY_RUN=1 bash rerun_350m_prenorm.sh
#         BACKUP=1  bash rerun_350m_prenorm.sh
#         TRAIN_REPO=/path/to/babylm-model-training bash rerun_350m_prenorm.sh
set -uo pipefail

MODEL_ID="znhoughton/opt-babylm-350m-20eps-seed964"   # unchanged: corrected model promoted into this name
TAG="opt-350m"
DRY_RUN="${DRY_RUN:-0}"
BACKUP="${BACKUP:-0}"
# babylm-model-training is a sibling repo; compute_val_loss.py lives there.
TRAIN_REPO="${TRAIN_REPO:-../babylm-model-training}"
# Override with RSCRIPT=/path/to/Rscript if needed.
RSCRIPT="${RSCRIPT:-Rscript}"

# ── Parallelism ──────────────────────────────────────────────────────────────
# BRMS_CORES is the total core budget for Stan sampling. Capped at the machine's
# real core count: oversubscribing Stan threads makes sampling slower, not faster.
DETECTED_CORES=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
BRMS_CORES="${BRMS_CORES:-24}"
if [ "$BRMS_CORES" -gt "$DETECTED_CORES" ]; then
    echo "  BRMS_CORES=$BRMS_CORES exceeds $DETECTED_CORES detected cores; capping."
    BRMS_CORES="$DETECTED_CORES"
fi

# ── RAM budget ───────────────────────────────────────────────────────────────
# Stan chains and (under multisession) each future worker hold their own copy of
# the data, so parallelism is bounded by memory as well as cores.
TOTAL_RAM_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')
[ -z "$TOTAL_RAM_GB" ] && TOTAL_RAM_GB=0
MAX_RAM_GB="${MAX_RAM_GB:-100}"
if [ "$TOTAL_RAM_GB" -gt 0 ] && [ "$MAX_RAM_GB" -gt "$TOTAL_RAM_GB" ]; then
    echo "  MAX_RAM_GB=$MAX_RAM_GB exceeds ${TOTAL_RAM_GB}GB installed; lowering."
    MAX_RAM_GB=$(( TOTAL_RAM_GB * 8 / 10 ))
fi

# multicore forks and shares the ~8 GB of loaded data frames copy-on-write, so
# only the per-worker sampling overhead scales. multisession copies the lot into
# every worker, which is far more expensive -- hence the two estimates.
export BRMS_PLAN="${BRMS_PLAN:-multicore}"
if [ "$BRMS_PLAN" = "multicore" ]; then PER_WORKER_GB="${PER_WORKER_GB:-5}"
else PER_WORKER_GB="${PER_WORKER_GB:-14}"; fi

workers_by_cores=$(( BRMS_CORES / 4 ))
workers_by_ram=$(( MAX_RAM_GB / PER_WORKER_GB ))
BRMS_WORKERS="${BRMS_WORKERS:-$workers_by_cores}"
if [ "$BRMS_WORKERS" -gt "$workers_by_ram" ]; then
    echo "  RAM-capped: ${workers_by_cores} workers would need ~$(( workers_by_cores * PER_WORKER_GB ))GB; budget is ${MAX_RAM_GB}GB"
    BRMS_WORKERS="$workers_by_ram"
fi
[ "$BRMS_WORKERS" -lt 1 ] && BRMS_WORKERS=1
export BRMS_WORKERS
echo "  brms: ${BRMS_WORKERS} workers x 4 chains = $(( BRMS_WORKERS * 4 )) cores, plan=${BRMS_PLAN}"
echo "        projected RAM ~$(( BRMS_WORKERS * PER_WORKER_GB ))GB of ${MAX_RAM_GB}GB budget"



# ── Interpreter ──────────────────────────────────────────────────────────────
# Override with PY=/path/to/python if the default is not the env you want.
PY="${PY:-}"
if [ -z "$PY" ]; then
    for c in python3 python; do
        command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }
    done
fi
[ -n "$PY" ] || { echo "FATAL: no python found. Set PY=/path/to/python." >&2; exit 1; }

"$PY" - <<'PROBE' || { echo "FATAL: $PY cannot import torch. Set PY= to the right env." >&2; exit 1; }
import sys, torch
print(f"interpreter: {sys.executable}")
print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}"
      + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
if not torch.cuda.is_available():
    print("WARNING: no CUDA. GPU stages will run on CPU and take far longer.")
PROBE

command -v "$RSCRIPT" >/dev/null 2>&1 || [ -x "$RSCRIPT" ] || {
    echo "FATAL: Rscript not found; the model-fitting stages need it." >&2
    echo "       Set RSCRIPT=/path/to/Rscript." >&2; exit 1; }

# paper/ is gitignored in this repo (.gitignore:12), so a fresh clone has none
# of it: prepare_results.R, writeup.qmd and paper/results/ live only where the
# paper is actually worked on. Fail here rather than three stages in.
for need in paper/prepare_results.R paper/writeup.qmd; do
    [ -f "$need" ] || { echo "FATAL: $need missing." >&2
        echo "       paper/ is gitignored, so it is absent from a fresh clone." >&2
        echo "       Run this where the paper lives, or copy paper/ across." >&2; exit 1; }
done

say () { echo; echo "=== $* ==="; }

# run(): a failing step must stop the run. Without this the script would sail on
# to re-render the paper after a failed extraction and produce confidently wrong
# output. (set -e is deliberately not used: the upstream runners here return
# non-zero for benign reasons, so failures are checked explicitly instead.)
run () {
    if [ "$DRY_RUN" = "1" ]; then echo "  [dry-run] $*"; return 0; fi
    "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo >&2
        echo "FAILED (exit $rc): $*" >&2
        echo "Stopping before anything downstream consumes a half-finished result." >&2
        exit $rc
    fi
}

# run_soft(): for steps whose failure genuinely does not invalidate the run
# (cache purge, optional backup).
run_soft () {
    if [ "$DRY_RUN" = "1" ]; then echo "  [dry-run] $*"; return 0; fi
    "$@" || echo "  WARNING: non-fatal step failed: $*" >&2
}

# Verification below compares file mtimes against this, not a fixed window: a
# long run would otherwise report its own early outputs as stale.
RUN_STARTED_AT=$(date +%s)

[ -f Analyses/babylm/run_pipeline.sh ] || { echo "Run me from the repo root."; exit 1; }

say "0. Preconditions"
echo "  writeup id references (expected 0): $(grep -c "$MODEL_ID" paper/writeup.qmd 2>/dev/null)"

say "1. Confirm the replacement model is pre-LN"
if [ "$DRY_RUN" != "1" ]; then
"$PY" - "$MODEL_ID" <<'PY'
import sys
from transformers import AutoConfig
c = AutoConfig.from_pretrained(sys.argv[1])
print(f"  do_layer_norm_before={c.do_layer_norm_before} hidden={c.hidden_size} layers={c.num_hidden_layers}")
if c.do_layer_norm_before is not True:
    print("  FATAL: not the corrected model."); sys.exit(1)
print("  OK: pre-LN.")
PY
[ $? -ne 0 ] && exit 1
fi

if [ "$BACKUP" = "1" ]; then
    say "1b. Backing up the old 350M outputs"
    stamp=$(date +%Y%m%d_%H%M%S)
    run_soft cp -r "Data/babylm/${TAG}" "Data/babylm/${TAG}.pre-prenorm.${stamp}"
    echo "  -> Data/babylm/${TAG}.pre-prenorm.${stamp}"
fi

say "2. Purge the stale HF cache"
# Cache is keyed by repo name; the corrected model was promoted into the
# original name, so the cached post-LN weights would be used silently.
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--znhoughton--opt-babylm-350m-20eps-seed964"
if [ -d "$CACHE_DIR" ]; then
    echo "  removing $CACHE_DIR"
    run_soft rm -rf "$CACHE_DIR"
else
    echo "  no cached copy at $CACHE_DIR"
fi

say "3. Re-run the pipeline for the 350M only (overwrites Data/babylm/${TAG})"
# Steps: build train/val/test CSVs -> standalone-up classifier by layer
#        -> up-subword classifier by layer
run bash Analyses/babylm/run_pipeline.sh "$TAG"

say "3b. Recompute validation perplexity"
# tbl-val-perplexity in the paper hardcodes OPT-350M = 2.94 / 18.86, which came
# from the post-LN model. compute_val_loss.py reads the canonical names, so
# after the rename it picks up the corrected 350M on its own. It recomputes all
# three models; the 125M and 1.3B should come back unchanged, which doubles as
# a check that the rename did not disturb anything else.
if [ -f "$TRAIN_REPO/compute_val_loss.py" ]; then
    run "$PY" "$TRAIN_REPO/compute_val_loss.py"
    if [ "$DRY_RUN" != "1" ] && [ -f "$TRAIN_REPO/babylm_val_results.csv" ]; then
        echo
        echo "  new values for tbl-val-perplexity (update the tribble by hand):"
        cat "$TRAIN_REPO/babylm_val_results.csv" | sed 's/^/    /'
    fi
else
    echo "  SKIPPED: $TRAIN_REPO/compute_val_loss.py not found."
    echo "  Set TRAIN_REPO=/path/to/babylm-model-training and re-run this step."
fi

say "3c. Move the cached 350M model fits aside"
# Analyses/run_models_parallel.R skips any model whose .rds already exists
# (pending <- Filter(!file.exists, ...)), and the GAM helpers do the same. The
# cache is gitignored, so these are moved, not deleted.
FIT_ARCHIVE="model_cache/babylm_pre_prenorm_$(date +%Y%m%d_%H%M%S)"
n_fits=$(ls model_cache/babylm/ 2>/dev/null | grep -c "350m")
if [ "$n_fits" -gt 0 ]; then
    echo "  moving $n_fits cached 350M fits -> $FIT_ARCHIVE"
    run mkdir -p "$FIT_ARCHIVE"
    for f in model_cache/babylm/*350m*; do
        [ -e "$f" ] || continue
        run mv "$f" "$FIT_ARCHIVE/"
    done
else
    echo "  no cached 350M fits found"
fi

say "3d. Re-fit the Bayesian and GAM models for the 350M"
# Must run from Analyses/: its cache paths are ../model_cache/*.
# Only the 350M models were evicted above, so the 125M/1.3B/OLMo/Whisper fits
# are still cached and are skipped -- this refits just what changed.
run bash -c 'cd Analyses && "$0" run_models_parallel.R' "$RSCRIPT"

say "3e. Rebuild paper/results from the refreshed fits"
# prepare_results.R reads model_cache/*.rds and writes the CSVs that
# writeup.qmd consumes via RES="results". Without this the render still sees
# the old numbers even after refitting.
run "$RSCRIPT" paper/prepare_results.R

say "4. Re-render the paper (no .qmd edits were needed)"
run quarto render paper/writeup.qmd

say "5. Verify the 350M outputs were rewritten"
if [ "$DRY_RUN" != "1" ]; then
    for d in "Data/babylm/${TAG}/Data_up" "Data/babylm/${TAG}/Data_upsubword"; do
        if [ -d "$d" ]; then
            n=$(find "$d" -name "*.csv" -newermt "@$RUN_STARTED_AT" 2>/dev/null | wc -l)
            echo "  $d: $n csv(s) written by this run"
        else
            echo "  $d: MISSING"
        fi
    done
fi

say "Done"
echo "arXiv:2606.13993 will need a revised version once you are happy with these numbers."
echo "No code edits were made: the corrected model carries the original name."
