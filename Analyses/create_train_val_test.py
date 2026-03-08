"""
Build Datasets
==============
Run this once before either analysis script. Produces train.csv, val.csv,
and test.csv in two output directories:

  Data_upsubword/   -- full dataset:
                        label=1: standalone "up" + up-within-word tokens (2000)
                        label=0: random other token from standalone "up" sentences
                                 + random other token from up-within-word sentences (2000)

  Data_up/          -- subset of Data_upsubword, dropping up_within_word rows:
                        label=1: standalone "up" tokens only (1000)
                        label=0: random other token from standalone "up" sentences (1000)

Both directories share the same sentences and token_position values for all
overlapping examples, so results are directly comparable across the two
analysis scripts.

Columns in train.csv and val.csv:
    word            -- decoded target token:
                        "up"         -> standalone particle (label=1)
                        e.g. "cup"   -> token containing "up" subword (label=1)
                        e.g. "the"   -> random other token (label=0)
    sentence        -- source sentence
    label           -- 1 = "up" subword, 0 = other token
    source          -- "standalone_up" | "up_within_word" |
                       "other_token_from_up" | "other_token_from_upword"
    token_position  -- BPE token index in the truncated sequence

Columns in test.csv (identical in both directories):
    verb_up         -- V+up type (e.g. "pick up")
    frequency       -- corpus frequency of that type
    word            -- always "up"
    sentence        -- source sentence
    token_position  -- BPE token index of the "up" token

Usage:
    python build_datasets.py

Expects:
    corpus_results.pkl         (from create_datasets.py)
    corpus_results_upwords.pkl (from create_datasets.py)
"""

import argparse
import os
import pickle
import logging
import random

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# CONFIG (defaults; all overridable via CLI — see --help)
# ---------------------------------------------------------------------------
BATCH_SIZE      = 350
RANDOM_SEED     = 964
MAX_SEQ_LEN     = 128
N_TRAIN         = 1000
N_VAL           = 1000
N_TEST_PER_TYPE = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build train/val/test CSVs from corpus pickle files."
    )
    parser.add_argument(
        "--model",
        default="allenai/Olmo-3-1025-7B",
        help="HuggingFace model ID (used to load the tokenizer). "
             "Default: allenai/Olmo-3-1025-7B",
    )
    parser.add_argument(
        "--data-dir-up",
        default="../Data/olmo-3-7b/Data_up",
        help="Output directory for the standalone-up dataset. "
             "Default: ../Data/olmo-3-7b/Data_up",
    )
    parser.add_argument(
        "--data-dir-upsubword",
        default="../Data/olmo-3-7b/Data_upsubword",
        help="Output directory for the up-subword dataset. "
             "Default: ../Data/olmo-3-7b/Data_upsubword",
    )
    parser.add_argument(
        "--vup-pkl",
        default="../Data/corpus_results.pkl",
        help="Path to corpus_results.pkl (produced by create_dataset.py). "
             "Default: ../Data/corpus_results.pkl",
    )
    parser.add_argument(
        "--upword-pkl",
        default="../Data/corpus_results_upwords.pkl",
        help="Path to corpus_results_upwords.pkl (produced by create_dataset.py). "
             "Default: ../Data/corpus_results_upwords.pkl",
    )
    parser.add_argument(
        "--corpus-stats-pkl",
        default=None,
        help="Path to corpus stats pkl produced by get_babylm_corpus_stats.py or "
             "get_olmo_corpus_stats.py. Contains (vup_freq, verb_freq, ftp). "
             "If provided, an 'ftp' column (P(up|V)) is added to test.csv. "
             "Default: None (ftp column omitted)",
    )
    return parser.parse_args()


args = parse_args()
MODEL_NAME         = args.model
DATA_DIR_UP        = args.data_dir_up
DATA_DIR_UPSUBWORD = args.data_dir_upsubword
VUP_PKL_PATH       = args.vup_pkl
UPWORD_PKL_PATH    = args.upword_pkl
CORPUS_STATS_PKL   = args.corpus_stats_pkl

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

os.makedirs(DATA_DIR_UPSUBWORD, exist_ok=True)
os.makedirs(DATA_DIR_UP,        exist_ok=True)


# ---------------------------------------------------------------------------
# LOAD CORPUS
# ---------------------------------------------------------------------------

def load_corpus():
    log.info("Loading V+up corpus from %s ...", VUP_PKL_PATH)
    with open(VUP_PKL_PATH, "rb") as f:
        vup_sentences, vup_freq, up_sentences, up_freq = pickle.load(f)

    log.info("Loading up-word corpus from %s ...", UPWORD_PKL_PATH)
    with open(UPWORD_PKL_PATH, "rb") as f:
        upword_sentences, upword_freq, _, _ = pickle.load(f)

    log.info(
        "V+up types: %d | standalone 'up' sentences: %d | up-word types: %d",
        len(vup_sentences), len(up_sentences), len(upword_sentences),
    )
    assert len(up_sentences) >= N_TRAIN + N_VAL, (
        f"Need at least {N_TRAIN + N_VAL} standalone 'up' sentences, "
        f"got {len(up_sentences)}."
    )

    all_upword_pairs = [
        (word, sent)
        for word, sents in upword_sentences.items()
        for sent in sents
    ]
    log.info("Total up-word (sentence, word) pairs: %d", len(all_upword_pairs))
    assert len(all_upword_pairs) >= N_TRAIN + N_VAL, (
        f"Need at least {N_TRAIN + N_VAL} up-word pairs, got {len(all_upword_pairs)}."
    )

    return vup_sentences, vup_freq, up_sentences, all_upword_pairs


# ---------------------------------------------------------------------------
# LOAD TOKENIZER
# ---------------------------------------------------------------------------

def load_tokenizer():
    log.info("Loading tokenizer: %s", MODEL_NAME)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    except Exception:
        log.warning("Fast tokenizer failed, falling back to use_fast=False")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, use_fast=False, trust_remote_code=True
        )
    return tokenizer


# ---------------------------------------------------------------------------
# TOKEN-POSITION RESOLUTION
# ---------------------------------------------------------------------------

def _decode(tokenizer, token_id):
    return tokenizer.decode([token_id]).strip().lower()


def resolve_up_standalone_positions(sentences, tokenizer):
    """Find BPE position of the last 'up' token in each sentence."""
    results = []
    encoded = tokenizer(
        sentences, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_SEQ_LEN,
    )
    for i, (ids, mask) in enumerate(zip(encoded["input_ids"], encoded["attention_mask"])):
        actual_len = mask.sum().item()
        ids_list   = ids[:actual_len].tolist()
        pos = None
        for k in range(len(ids_list) - 1, -1, -1):
            if _decode(tokenizer, ids_list[k]) == "up":
                pos = k
                break
        if pos is not None:
            results.append((sentences[i], pos, "up"))
    return results


def resolve_upword_positions(upword_pairs, tokenizer):
    """Find BPE position of the last token containing 'up' as a substring."""
    sentences = [sent for _, sent in upword_pairs]
    words     = [word for word, _ in upword_pairs]
    results   = []
    encoded   = tokenizer(
        sentences, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_SEQ_LEN,
    )
    for i, (ids, mask) in enumerate(zip(encoded["input_ids"], encoded["attention_mask"])):
        actual_len = mask.sum().item()
        ids_list   = ids[:actual_len].tolist()
        pos = None
        for k in range(len(ids_list) - 1, -1, -1):
            if "up" in _decode(tokenizer, ids_list[k]):
                pos = k
                break
        if pos is not None:
            results.append((sentences[i], pos, words[i]))
    return results


def resolve_other_token_positions(sentences, tokenizer, rng, exclude_subword_up=False):
    """
    For each sentence, randomly select a non-'up', non-special token position.
    Processed in BATCH_SIZE chunks.

    If exclude_subword_up=True, also excludes any token containing 'up' as a
    substring -- used when drawing negatives from upword sentences so the
    target up-subword token (e.g. "cup") is not accidentally selected.
    """
    results = []
    for batch_start in range(0, len(sentences), BATCH_SIZE):
        batch   = sentences[batch_start : batch_start + BATCH_SIZE]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_SEQ_LEN,
        )
        for i, (ids, mask) in enumerate(zip(encoded["input_ids"], encoded["attention_mask"])):
            actual_len     = mask.sum().item()
            ids_list       = ids[:actual_len].tolist()
            tokens_decoded = [_decode(tokenizer, tid) for tid in ids_list]

            # Find the last exact "up" position to exclude
            up_pos = None
            for k in range(len(tokens_decoded) - 1, -1, -1):
                if tokens_decoded[k] == "up":
                    up_pos = k
                    break

            non_up_positions = [
                j for j in range(actual_len)
                if tokens_decoded[j] not in ("", "up")
                and ids_list[j] not in tokenizer.all_special_ids
                and j != up_pos
                and (not exclude_subword_up or "up" not in tokens_decoded[j])
            ]
            if non_up_positions:
                chosen_pos  = rng.choice(non_up_positions)
                chosen_word = tokenizer.decode([ids_list[chosen_pos]]).strip()
                if chosen_word:
                    results.append((batch[i], chosen_pos, chosen_word))
    return results


def resolve_vup_positions(vup_sentences_filtered, tokenizer):
    """Find BPE position of the last 'up' token for each V+up test sentence."""
    vup_positions = {}
    for vup_type, sents in vup_sentences_filtered.items():
        sents_subset = sents[:N_TEST_PER_TYPE]
        encoded = tokenizer(
            sents_subset, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_SEQ_LEN,
        )
        type_results = []
        for i, (ids, mask) in enumerate(zip(encoded["input_ids"], encoded["attention_mask"])):
            actual_len = mask.sum().item()
            ids_list   = ids[:actual_len].tolist()
            pos = None
            for k in range(len(ids_list) - 1, -1, -1):
                if _decode(tokenizer, ids_list[k]) == "up":
                    pos = k
                    break
            if pos is not None:
                type_results.append((sents_subset[i], pos, "up"))
        if type_results:
            vup_positions[vup_type] = type_results
    return vup_positions


# ---------------------------------------------------------------------------
# BUILD & SAVE
# ---------------------------------------------------------------------------

def build_and_save(vup_sentences, vup_freq, up_sentences, all_upword_pairs, tokenizer):
    log.info("=" * 60)
    log.info("RESOLVING TOKEN POSITIONS")
    log.info("=" * 60)

    # --- Positive A: standalone "up" ---
    log.info("  Standalone 'up' ...")
    up_resolved           = resolve_up_standalone_positions(up_sentences[:N_TRAIN + N_VAL], tokenizer)
    up_train_resolved     = up_resolved[:N_TRAIN]
    up_val_resolved       = up_resolved[N_TRAIN : N_TRAIN + N_VAL]
    log.info("    Train: %d | Val: %d", len(up_train_resolved), len(up_val_resolved))

    # --- Positive B: up-within-word ---
    log.info("  Up-within-word ...")
    upword_resolved       = resolve_upword_positions(all_upword_pairs[:N_TRAIN + N_VAL], tokenizer)
    upword_train_resolved = upword_resolved[:N_TRAIN]
    upword_val_resolved   = upword_resolved[N_TRAIN : N_TRAIN + N_VAL]
    log.info("    Train: %d | Val: %d", len(upword_train_resolved), len(upword_val_resolved))

    # --- Negative A: random other token from standalone "up" sentences ---
    log.info("  Other tokens (from standalone 'up' sentences) ...")
    rng                       = np.random.default_rng(RANDOM_SEED)
    neg_up_resolved           = resolve_other_token_positions(
        up_sentences[:N_TRAIN + N_VAL], tokenizer, rng
    )
    neg_up_train_resolved     = neg_up_resolved[:N_TRAIN]
    neg_up_val_resolved       = neg_up_resolved[N_TRAIN : N_TRAIN + N_VAL]
    log.info("    Train: %d | Val: %d", len(neg_up_train_resolved), len(neg_up_val_resolved))

    # --- Negative B: random other token from up-within-word sentences ---
    # exclude_subword_up=True ensures the "up"-containing token (e.g. "cup")
    # is not accidentally selected as a negative example
    log.info("  Other tokens (from up-within-word sentences) ...")
    upword_sentences_only     = [sent for _, sent in all_upword_pairs[:N_TRAIN + N_VAL]]
    neg_upword_resolved       = resolve_other_token_positions(
        upword_sentences_only, tokenizer, rng, exclude_subword_up=True
    )
    neg_upword_train_resolved = neg_upword_resolved[:N_TRAIN]
    neg_upword_val_resolved   = neg_upword_resolved[N_TRAIN : N_TRAIN + N_VAL]
    log.info("    Train: %d | Val: %d", len(neg_upword_train_resolved), len(neg_upword_val_resolved))

    log.info(
        "  Class balance -- Train pos: %d | Train neg: %d | Val pos: %d | Val neg: %d",
        len(up_train_resolved) + len(upword_train_resolved),
        len(neg_up_train_resolved) + len(neg_upword_train_resolved),
        len(up_val_resolved) + len(upword_val_resolved),
        len(neg_up_val_resolved) + len(neg_upword_val_resolved),
    )

    # --- Test set ---
    log.info("  V+up test set ...")
    excluded = [vt for vt, sents in vup_sentences.items() if len(sents) < N_TEST_PER_TYPE]
    vup_sentences_filtered = {
        vt: sents for vt, sents in vup_sentences.items() if len(sents) >= N_TEST_PER_TYPE
    }
    log.info(
        "    %d types included, %d excluded (< %d sentences)",
        len(vup_sentences_filtered), len(excluded), N_TEST_PER_TYPE,
    )
    vup_positions = resolve_vup_positions(vup_sentences_filtered, tokenizer)

    # --- Assemble DataFrames ---
    def make_rows(resolved, label, source):
        return [
            {"word": word, "sentence": sent, "label": label,
             "source": source, "token_position": pos}
            for sent, pos, word in resolved
        ]

    train_full = pd.DataFrame(
        make_rows(up_train_resolved,          label=1, source="standalone_up")           +
        make_rows(upword_train_resolved,      label=1, source="up_within_word")           +
        make_rows(neg_up_train_resolved,      label=0, source="other_token_from_up")      +
        make_rows(neg_upword_train_resolved,  label=0, source="other_token_from_upword")
    )
    val_full = pd.DataFrame(
        make_rows(up_val_resolved,            label=1, source="standalone_up")           +
        make_rows(upword_val_resolved,        label=1, source="up_within_word")           +
        make_rows(neg_up_val_resolved,        label=0, source="other_token_from_up")      +
        make_rows(neg_upword_val_resolved,    label=0, source="other_token_from_upword")
    )

    # Load FTP values if corpus stats pkl was provided
    ftp = {}
    if CORPUS_STATS_PKL is not None:
        log.info("Loading corpus stats (FTP) from %s ...", CORPUS_STATS_PKL)
        with open(CORPUS_STATS_PKL, "rb") as f:
            _, _, ftp = pickle.load(f)
        log.info("  FTP values loaded for %d V+up types", len(ftp))

    test_rows = []
    for vup_type, type_records in vup_positions.items():
        ftp_val = ftp.get(vup_type, float("nan")) if ftp else float("nan")
        for sent, pos, word in type_records:
            row = {
                "verb_up":        vup_type,
                "frequency":      vup_freq[vup_type],
                "word":           word,
                "sentence":       sent,
                "token_position": pos,
            }
            if CORPUS_STATS_PKL is not None:
                row["ftp"] = ftp_val
            test_rows.append(row)
    test_df = pd.DataFrame(test_rows).sort_values(
        ["frequency", "verb_up"], ascending=[False, True]
    ).reset_index(drop=True)

    # --- Save Data_upsubword (full: standalone + upword + both negatives) ---
    log.info("=" * 60)
    log.info("SAVING Data_upsubword")
    log.info("=" * 60)
    train_full.to_csv(os.path.join(DATA_DIR_UPSUBWORD, "train.csv"), index=False)
    val_full.to_csv(  os.path.join(DATA_DIR_UPSUBWORD, "val.csv"),   index=False)
    test_df.to_csv(   os.path.join(DATA_DIR_UPSUBWORD, "test.csv"),  index=False)
    log.info("  train.csv: %d rows (label=1: %d, label=0: %d)",
             len(train_full), (train_full.label==1).sum(), (train_full.label==0).sum())
    log.info("  val.csv:   %d rows (label=1: %d, label=0: %d)",
             len(val_full), (val_full.label==1).sum(), (val_full.label==0).sum())
    log.info("  test.csv:  %d rows, %d V+up types", len(test_df), test_df["verb_up"].nunique())

    # --- Save Data_up (standalone + other_token_from_up only) ---
    log.info("=" * 60)
    log.info("SAVING Data_up")
    log.info("=" * 60)
    keep     = {"standalone_up", "other_token_from_up"}
    train_up = train_full[train_full["source"].isin(keep)].reset_index(drop=True)
    val_up   = val_full[val_full["source"].isin(keep)].reset_index(drop=True)
    train_up.to_csv(os.path.join(DATA_DIR_UP, "train.csv"), index=False)
    val_up.to_csv(  os.path.join(DATA_DIR_UP, "val.csv"),   index=False)
    test_df.to_csv( os.path.join(DATA_DIR_UP, "test.csv"),  index=False)
    log.info("  train.csv: %d rows (label=1: %d, label=0: %d)",
             len(train_up), (train_up.label==1).sum(), (train_up.label==0).sum())
    log.info("  val.csv:   %d rows (label=1: %d, label=0: %d)",
             len(val_up), (val_up.label==1).sum(), (val_up.label==0).sum())
    log.info("  test.csv:  %d rows, %d V+up types (copied)", len(test_df), test_df["verb_up"].nunique())

    log.info("=" * 60)
    log.info("Done.")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    vup_sentences, vup_freq, up_sentences, all_upword_pairs = load_corpus()
    tokenizer = load_tokenizer()
    build_and_save(vup_sentences, vup_freq, up_sentences, all_upword_pairs, tokenizer)
    if CORPUS_STATS_PKL is None:
        log.warning(
            "No --corpus-stats-pkl provided. The 'ftp' column will be absent from test.csv. "
            "Run get_olmo_corpus_stats.py or get_babylm_corpus_stats.py first, "
            "then re-run with --corpus-stats-pkl."
        )


if __name__ == "__main__":
    main()