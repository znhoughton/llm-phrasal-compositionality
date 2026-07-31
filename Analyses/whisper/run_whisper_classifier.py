"""
Whisper Layer-by-Layer Classifier
===================================
Trains a logistic regression classifier layer-by-layer on Whisper-small's
hidden states (encoder AND decoder) to distinguish standalone "up" tokens
from random other words.

  Positive (label=1): "up" token embeddings (encoder or decoder)
  Negative (label=0): random other word from the same utterance

Encoder embedding: mean-pool hidden states over audio frames corresponding
    to the "up" word's time span (20ms per encoder output frame).
Decoder embedding: hidden state at the "up" token position (teacher-forced).

For efficiency, each utterance is run through a SINGLE Whisper forward pass
that returns all encoder and decoder hidden states simultaneously.

Reads:
    DATA_DIR/dataset.csv  (built by build_audio_dataset.py)
    --subword-dataset PATH (optional; dataset_subword.csv built by
        build_subword_audio_dataset.py). If given, up-within-word instances
        (e.g. "update", "upon") are combined with standalone-"up" instances
        for classifier TRAINING/VALIDATION only -- Experiment 2 replication.
        The V+up test set, model, and layers are completely unchanged from
        Experiment 1; only the classifier's training data differs.

Outputs:
    DATA_DIR/encoder/layer_XX.csv, all_layers_results.csv, layer_metadata.json
    DATA_DIR/decoder/layer_XX.csv, all_layers_results.csv, layer_metadata.json

Usage:
    python run_whisper_classifier.py [--data-dir DIR] [--model MODEL] [--device DEVICE]
    python run_whisper_classifier.py --subword-dataset ../../Data/whisper/dataset_subword.csv

Requires:
    pip install transformers torch soundfile scikit-learn pandas numpy tqdm
"""

import argparse
import json
import logging
import os
import pickle
import random
import re
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RANDOM_SEED     = 964
N_TRAIN         = 1000   # balanced pos+neg each for train
N_VAL           = 1000   # balanced pos+neg each for val (matches OLMo/BabyLM)
N_TEST_PER_TYPE = 20     # max test sentences per V+up type
MIN_FREQ_VUP    = 5      # min occurrences to include a V+up type in test
MIN_FREQ_UPWORD = 5      # min occurrences (in dataset_subword.csv) to include an
                          # up-word type in train/val -- mirrors MIN_FREQ_VUP's
                          # role for V+up types, applied here since this is where
                          # that filter actually has effect (build_audio_dataset.py's
                          # own copy of MIN_FREQ_VUP is diagnostic-only, not a filter)
MAX_INSTANCES_PER_UPWORD_TYPE = 5  # max instances sampled per up-word type for
                          # train/val (audio only -- the text-side design in
                          # create_train_val_test.py uses exactly 1, which is
                          # fine there because ~8,662 qualifying types are
                          # available; the audio corpus only yields a few
                          # hundred, so capping at exactly 1 each produced an
                          # unworkably small training set. Set equal to
                          # MIN_FREQ_UPWORD so every qualifying type can supply
                          # the full cap without shortfall. Still caps any
                          # single frequent type from dominating the training
                          # signal -- just a looser cap than 1.

# Whisper-small encoder: Conv1d stride=2 on 10ms frames → 20ms per output token
ENCODER_FRAME_SEC = 0.02

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logging.getLogger("transformers").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# ARGS
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer-by-layer Whisper encoder+decoder classifier."
    )
    parser.add_argument(
        "--data-dir", default="../../Data/whisper",
        help="Directory containing dataset.csv. Default: ../../Data/whisper "
             "(fixed from a stale '../../Data/whisper_audio' default that predates "
             "this project's Data/ directory being renamed -- that path doesn't "
             "exist and doesn't match where the real dataset.csv/encoder/decoder "
             "results actually live).",
    )
    parser.add_argument(
        "--model", default="openai/whisper-small",
        help="HuggingFace Whisper model ID. Default: openai/whisper-small",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device. Default: cuda",
    )
    parser.add_argument(
        "--corpus-stats-pkl", default=None,
        help="Path to olmo_corpus_stats.pkl (vup_freq, verb_freq, predic) produced "
             "by get_olmo_corpus_stats.py.  If provided, the 'frequency' column "
             "uses Dolma corpus counts instead of audio occurrence counts. "
             "Default: None",
    )
    parser.add_argument(
        "--subword-dataset", default=None,
        help="Path to dataset_subword.csv (produced by build_subword_audio_dataset.py). "
             "If provided, up-within-word instances are combined with standalone-'up' "
             "instances for classifier training/validation only (Experiment 2 "
             "replication) -- the V+up test set is unaffected. Default: None "
             "(Experiment 1 behavior, unchanged).",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory to save encoder/decoder layer results to (separate from "
             "--data-dir, which is only read from -- dataset.csv is never written "
             "back to). Defaults to --data-dir itself UNLESS --subword-dataset is "
             "set, in which case it defaults to '<data-dir>_subword' instead, since "
             "writing a subword-combined run's results into the same directory as "
             "the primary (non-subword) results would silently overwrite them -- "
             "the layer_XX.csv / all_layers_results.csv / layer_metadata.json files "
             "the rest of this project's analysis already depends on. Pass this "
             "explicitly to choose a different location.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

def load_model(model_name, device):
    log.info("Loading %s ...", model_name)
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    cfg = model.config
    log.info(
        "Encoder layers: %d | Decoder layers: %d | d_model: %d",
        cfg.encoder_layers, cfg.decoder_layers, cfg.d_model,
    )
    return processor, model


# ---------------------------------------------------------------------------
# DATA-QUALITY FILTERS
# ---------------------------------------------------------------------------
# Both filters below are Whisper/audio-specific. The text pipeline
# (create_train_val_test.py's resolve_vup_positions / resolve_upword_positions)
# already scans BACKWARD and takes the LAST matching token, so it never faces
# the ambiguity these filters remove; only this script's target_val==1 branch
# (up_ids, forward scan, take [0]) is affected. See docstrings below for why
# each filter exists.

_UP_WORD_RE = re.compile(r'\b([a-z]*up[a-z]+|[a-z]+up[a-z]*)\b', re.IGNORECASE)


def drop_ambiguous_up_position_rows(df, label_col="label"):
    """
    Drop any row whose transcript contains more than one occurrence of the
    literal word "up" (for label in {word_up, vup, standalone_up}) -- the
    decoder embedding for these labels is extracted by scanning the
    tokenized transcript FORWARD and taking the FIRST token matching "up"
    (see extract_all_layers()), with no check that this is actually the
    token whose timing matches the row's own up_start/up_end. When a
    transcript has 2+ "up"s, the row's OWN occurrence may not be the first
    one, silently pointing the decoder embedding at the wrong instance.
    Confirmed empirically: among rows sharing an identical transcript with
    distinct up_start values (i.e. separately-extracted occurrences of the
    same segment), ~10% turned out to not be the earliest occurrence and
    would therefore get another occurrence's decoder embedding.

    For label == subword_up, the analogous ambiguity is having more than one
    up-CONTAINING word in the transcript (find_subword_up_position scans
    backward and takes the LAST match, which is likewise not guaranteed to
    be this row's own occurrence when there are multiple candidates).
    """
    transcripts = df["transcript"].astype(str).str.lower()
    is_subword  = df[label_col] == "subword_up"

    up_word_count  = transcripts.apply(lambda t: len(_UP_WORD_RE.findall(t)))
    up_token_count = transcripts.apply(lambda t: len(re.findall(r"\bup\b", t)))

    ambiguous = np.where(is_subword, up_word_count > 1, up_token_count > 1)
    n_dropped = int(ambiguous.sum())
    if n_dropped:
        log.info(
            "  Dropping %d/%d rows with an ambiguous multi-'up' transcript "
            "(unresolvable decoder token position)", n_dropped, len(df),
        )
    return df.loc[~ambiguous].copy(), n_dropped


def drop_nonadjacent_vup_rows(df):
    """
    Drop V+up (label == "vup") rows where the reconstructed verb_up string
    (e.g. "picked up") does not appear as a literal contiguous substring of
    the transcript -- i.e. cases where classify_ups_from_doc()'s dep_=='prt'
    allowance (build_audio_dataset.py) matched a particle-shifted
    construction like "picked it up", where a word sits between the verb
    and "up". The text pipeline (create_dataset.py's is_verb_up_context)
    never allows this: it requires "up" to immediately follow a VERB token.
    Confirmed empirically: ~14% of vup rows are non-adjacent by this check.
    Rows for other labels are returned unchanged.
    """
    is_vup = df["label"] == "vup"
    verb_up_l    = df["verb_up"].astype(str).str.lower()
    transcript_l = df["transcript"].astype(str).str.lower()
    adjacent = pd.Series(True, index=df.index)
    adjacent[is_vup] = [
        vu in tr for vu, tr in zip(verb_up_l[is_vup], transcript_l[is_vup])
    ]
    n_dropped = int((is_vup & ~adjacent).sum())
    if n_dropped:
        log.info(
            "  Dropping %d/%d V+up rows where a word sits between the verb "
            "and 'up' (non-adjacent particle construction)",
            n_dropped, int(is_vup.sum()),
        )
    return df.loc[adjacent].copy(), n_dropped


# ---------------------------------------------------------------------------
# SPLITS
# ---------------------------------------------------------------------------

def build_splits(df, subword_df=None):
    """
    subword_df: optional DataFrame from dataset_subword.csv (label="subword_up",
    upword_type=<word>). If provided, up-word types are first restricted to
    those with >= MIN_FREQ_UPWORD occurrences (mirroring MIN_FREQ_VUP's role
    for V+up types below), then up to N_TRAIN + N_VAL unique qualifying types
    are combined with the standalone-"up" positives for train/val only,
    mirroring create_train_val_test.py's design for the text models (1,000
    standalone + 1,000 unique up-within-word types). The V+up test set is
    built from df alone in either case and is completely unaffected.
    """
    # "word_up"      : non-V+up rows — used for train/val
    # "standalone_up" : legacy label — also accepted for backward compatibility
    standalone = df[df.label.isin(["word_up", "standalone_up"])].copy()
    vup_df     = df[df.label == "vup"].copy()

    vup_counts = vup_df["verb_up"].value_counts()
    qualifying = vup_counts[vup_counts >= MIN_FREQ_VUP].index.tolist()
    log.info(
        "Qualifying V+up types (>=%d occurrences): %d", MIN_FREQ_VUP, len(qualifying)
    )

    # Test: up to N_TEST_PER_TYPE per qualifying type
    test_df = pd.concat([
        vup_df[vup_df.verb_up == vt].head(N_TEST_PER_TYPE)
        for vt in qualifying
    ]) if qualifying else pd.DataFrame()

    # Train/val from word_up positives — mirrors OLMo/BabyLM: take first N_TRAIN
    # rows as train, next N_VAL rows as val (after shuffling with fixed seed).
    standalone = standalone.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    if len(standalone) < N_TRAIN + N_VAL:
        log.error(
            "Not enough prepositional 'up' rows after filtering: "
            "need %d (N_TRAIN=%d + N_VAL=%d), got %d. "
            "Check that build_audio_dataset.py found sufficient prep uses in the audio metadata.",
            N_TRAIN + N_VAL, N_TRAIN, N_VAL, len(standalone),
        )
        sys.exit(1)

    train_pos = standalone.iloc[:N_TRAIN].copy()
    val_pos   = standalone.iloc[N_TRAIN : N_TRAIN + N_VAL].copy()
    train_pos["target"] = 1
    val_pos["target"]   = 1

    if subword_df is not None and len(subword_df) > 0:
        upword_counts     = subword_df["upword_type"].value_counts()
        qualifying_upword = upword_counts[upword_counts >= MIN_FREQ_UPWORD].index.tolist()
        log.info(
            "Qualifying up-word types (>=%d occurrences): %d / %d",
            MIN_FREQ_UPWORD, len(qualifying_upword), len(upword_counts),
        )
        subword_df = subword_df[subword_df["upword_type"].isin(qualifying_upword)]

        # Split TYPES (not rows) into train/val first, so the same up-word type
        # never appears in both -- otherwise the classifier could be validated
        # on a type it already saw (different audio instance, same word) during
        # training, inflating validation accuracy as an estimate of general
        # "up"-within-word recognition rather than type generalization.
        rng = np.random.RandomState(RANDOM_SEED)
        shuffled_types = list(rng.permutation(qualifying_upword))
        n_types = len(shuffled_types)
        log.info("Unique up-word types available: %d", n_types)
        if n_types < N_TRAIN + N_VAL:
            log.warning(
                "Fewer unique up-word types (%d) than N_TRAIN+N_VAL (%d); "
                "using all available, split proportionally.",
                n_types, N_TRAIN + N_VAL,
            )
            n_train_types = int(n_types * N_TRAIN / (N_TRAIN + N_VAL))
        else:
            n_train_types = N_TRAIN
            shuffled_types = shuffled_types[: N_TRAIN + N_VAL]
            n_types = len(shuffled_types)

        train_types = set(shuffled_types[:n_train_types])
        val_types   = set(shuffled_types[n_train_types:n_types])

        def sample_upword_instances(types):
            # Up to MAX_INSTANCES_PER_UPWORD_TYPE instances per type, not just
            # 1 -- see MAX_INSTANCES_PER_UPWORD_TYPE's own comment for why.
            return (
                subword_df[subword_df["upword_type"].isin(types)]
                .groupby("upword_type", group_keys=False)
                .apply(lambda g: g.sample(
                    n=min(len(g), MAX_INSTANCES_PER_UPWORD_TYPE), random_state=RANDOM_SEED
                ))
                .sample(frac=1, random_state=RANDOM_SEED)
                .reset_index(drop=True)
            )

        subword_train = sample_upword_instances(train_types)
        subword_val   = sample_upword_instances(val_types)
        subword_train["target"] = 1
        subword_val["target"]   = 1

        train_pos = pd.concat([train_pos, subword_train], ignore_index=True)
        val_pos   = pd.concat([val_pos, subword_val], ignore_index=True)
        log.info(
            "Combined positives — Train: %d (%d standalone + %d subword from %d types) | "
            "Val: %d (%d standalone + %d subword from %d types)",
            len(train_pos), N_TRAIN, len(subword_train), len(train_types),
            len(val_pos), N_VAL, len(subword_val), len(val_types),
        )

    def make_neg(rows):
        neg = rows.copy()
        neg["up_start"] = neg["neg_start"]
        neg["up_end"]   = neg["neg_end"]
        neg["target"]   = 0
        return neg

    train_df = pd.concat([train_pos, make_neg(train_pos)]).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)
    val_df = pd.concat([val_pos, make_neg(val_pos)]).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    log.info(
        "Train: %d | Val: %d | Test types: %d (%d rows)",
        len(train_df), len(val_df), len(qualifying), len(test_df),
    )
    return train_df, val_df, test_df, qualifying, dict(vup_counts)


# ---------------------------------------------------------------------------
# EMBEDDING EXTRACTION — single forward pass per utterance, all layers
# ---------------------------------------------------------------------------

def find_word_token_ids(processor, word):
    """Return set of token ids for common surface forms of word."""
    word = word.strip(".,!?;:\"'").strip()
    if not word:
        return set()
    ids = set()
    for candidate in [f" {word}", word, f" {word.capitalize()}", word.capitalize(),
                      f" {word.upper()}", word.upper()]:
        ids.update(processor.tokenizer.encode(candidate, add_special_tokens=False))
    return ids


def find_subword_up_position(processor, tokens):
    """
    For subword_up rows: find the position of the LAST token in the decoder
    sequence whose decoded string contains "up" as a substring. Mirrors
    resolve_upword_positions() in create_train_val_test.py exactly (same
    "scan backward, take first hit" rule), since there's no fixed token-id
    set to look up -- unlike standalone "up", the up-containing word (e.g.
    "update") may tokenize into a completely different BPE piece depending
    on the specific word and its context.
    """
    for k in range(len(tokens) - 1, -1, -1):
        decoded = processor.tokenizer.decode([tokens[k]])
        if "up" in decoded.lower():
            return k
    return None


def extract_all_layers(df, processor, model, device, n_enc, n_dec, desc=""):
    """
    Run one Whisper forward pass per row (audio file) and collect hidden states
    at the 'up' position for all encoder and decoder layers simultaneously.

    Returns:
        enc[layer_idx] : list of np.ndarray  (one per row, or None if skipped)
        dec[layer_idx] : list of np.ndarray  (one per row, or None if skipped)
        targets        : list of int labels (from df["target"] if present, else 1)
        sources        : list of str, df["label"] if present else "" -- lets
            callers separate subword_up rows from standalone rows (e.g. to
            report validation accuracy on the subword condition specifically,
            not just combined with standalone -- see train_classifier()).
    """
    enc  = [[] for _ in range(n_enc)]
    dec  = [[] for _ in range(n_dec)]
    targets = []
    sources = []
    up_ids           = find_word_token_ids(processor, "up")
    neg_word_id_cache = {}   # word string -> set of token ids

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc, unit="utt"):
        try:
            audio, sr = sf.read(row["audio_path"])
            audio = np.array(audio, dtype=np.float32)
            if sr != 16000:
                raise ValueError(f"Expected 16kHz, got {sr}")

            # Audio features
            input_features = processor(
                audio, sampling_rate=16000, return_tensors="pt",
            ).input_features.to(device, dtype=torch.float16)

            # Decoder: encode transcript with special tokens
            decoder_input_ids = processor.tokenizer.encode(
                row["transcript"], return_tensors="pt", add_special_tokens=True,
            ).to(device)

            with torch.no_grad():
                outputs = model.model(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids,
                    output_hidden_states=True,
                )

            # ---- Encoder: mean-pool over "up" audio frames ----
            start_frame = int(row["up_start"] / ENCODER_FRAME_SEC)
            end_frame   = max(start_frame + 1, int(row["up_end"] / ENCODER_FRAME_SEC))

            for li in range(n_enc):
                h = outputs.encoder_hidden_states[li + 1]   # (1, T, d)
                ef = min(end_frame, h.shape[1])
                sf_ = min(start_frame, ef - 1)
                emb = h[0, sf_:ef, :].mean(dim=0).float().cpu().numpy()
                enc[li].append(emb)

            # ---- Decoder: hidden state at the target token position ----
            # Positives (target=1): find the "up" token -- for standalone
            #   "up"/V+up rows this is a fixed token-id lookup (up_ids); for
            #   subword_up rows the up-containing word (e.g. "update") may
            #   tokenize to a different BPE piece each time, so its position
            #   is found dynamically per-row instead (mirrors
            #   resolve_upword_positions() in create_train_val_test.py).
            # Negatives (target=0): find the neg_word token so the classifier
            #   sees genuinely different embeddings for the two classes.
            tokens     = decoder_input_ids[0].tolist()
            target_val = int(row["target"]) if "target" in row else 1
            is_subword = row.get("label") == "subword_up"

            target_positions = None
            if target_val == 1 and is_subword:
                pos = find_subword_up_position(processor, tokens)
                target_positions = [pos] if pos is not None else []
            elif target_val == 1:
                target_positions = [j for j, t in enumerate(tokens) if t in up_ids]
            else:
                nw = str(row.get("neg_word", "")).strip()
                if nw not in neg_word_id_cache:
                    neg_word_id_cache[nw] = find_word_token_ids(processor, nw)
                target_ids = neg_word_id_cache[nw]
                target_positions = [j for j, t in enumerate(tokens) if t in target_ids]

            if not target_positions:
                for li in range(n_dec):
                    dec[li].append(None)
            else:
                dec_pos = target_positions[0]
                for li in range(n_dec):
                    h = outputs.decoder_hidden_states[li + 1]   # (1, S, d)
                    emb = h[0, dec_pos, :].float().cpu().numpy()
                    dec[li].append(emb)

            targets.append(int(row["target"]) if "target" in row else 1)
            sources.append(str(row.get("label", "")))

        except Exception as e:
            log.debug("Skipped: %s", e)
            for li in range(n_enc):
                enc[li].append(None)
            for li in range(n_dec):
                dec[li].append(None)
            targets.append(int(row["target"]) if "target" in row else 1)
            sources.append(str(row.get("label", "")))

    return enc, dec, targets, sources


def layer_arrays(layer_embs, targets, sources=None):
    """Filter out None entries, return (X, y) numpy arrays, or (X, y, src) if
    sources is given (see extract_all_layers() for what sources holds)."""
    X, y, src = [], [], []
    for i, (emb, lbl) in enumerate(zip(layer_embs, targets)):
        if emb is not None:
            X.append(emb)
            y.append(lbl)
            if sources is not None:
                src.append(sources[i])
    if not X:
        X_arr, y_arr = np.zeros((0, 1)), np.zeros(0, dtype=int)
        return (X_arr, y_arr, np.array([], dtype=object)) if sources is not None else (X_arr, y_arr)
    X_arr, y_arr = np.vstack(X), np.array(y)
    return (X_arr, y_arr, np.array(src, dtype=object)) if sources is not None else (X_arr, y_arr)


# ---------------------------------------------------------------------------
# CLASSIFIER
# ---------------------------------------------------------------------------

def train_classifier(X_train, y_train, X_val, y_val, src_val=None):
    """Logistic regression with class balancing by majority truncation.

    src_val: optional array (from layer_arrays' sources output) aligned with
    X_val/y_val, giving each row's original df["label"]. If provided, and any
    validation positives are labelled "subword_up", also reports validation
    accuracy restricted to those rows -- the subword condition specifically,
    not blended in with standalone-up validation performance.
    """
    pos_tr = np.where(y_train == 1)[0]
    neg_tr = np.where(y_train == 0)[0]
    n_tr   = min(len(pos_tr), len(neg_tr))
    idx_tr = np.concatenate([pos_tr[:n_tr], neg_tr[:n_tr]])
    X_tr, y_tr = X_train[idx_tr], y_train[idx_tr]

    pos_va = np.where(y_val == 1)[0]
    neg_va = np.where(y_val == 0)[0]
    n_va   = min(len(pos_va), len(neg_va))
    idx_va = np.concatenate([pos_va[:n_va], neg_va[:n_va]])
    X_va, y_va = X_val[idx_va], y_val[idx_va]
    src_va = src_val[idx_va] if src_val is not None else None

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_va_sc  = scaler.transform(X_va)

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
    clf.fit(X_tr_sc, y_tr)

    cv        = cross_val_score(clf, X_tr_sc, y_tr, cv=5, scoring="accuracy")
    val_preds = clf.predict(X_va_sc)
    val_acc   = (val_preds == y_va).mean()
    up_acc    = (val_preds[y_va == 1] == 1).mean() if (y_va == 1).any() else float("nan")
    oth_acc   = (val_preds[y_va == 0] == 0).mean() if (y_va == 0).any() else float("nan")

    subword_acc = float("nan")
    n_subword_va = 0
    if src_va is not None:
        is_sub_pos = (y_va == 1) & (src_va == "subword_up")
        n_subword_va = int(is_sub_pos.sum())
        if n_subword_va > 0:
            subword_acc = (val_preds[is_sub_pos] == 1).mean()

    log.info(
        "  CV: %.3f±%.3f | Val: %.3f (up=%.3f, other=%.3f, subword=%.3f n=%d)",
        cv.mean(), cv.std(), val_acc, up_acc, oth_acc, subword_acc, n_subword_va,
    )
    return clf, scaler, {
        "cv_mean": float(cv.mean()), "cv_std": float(cv.std()),
        "val_acc": float(val_acc),   "up_acc": float(up_acc),   "other_acc": float(oth_acc),
        "subword_acc": float(subword_acc), "n_subword_va": n_subword_va,
        "n_train_pos": int(n_tr), "n_train_neg": int(n_tr),
        "n_val_pos":   int(n_va), "n_val_neg":   int(n_va),
    }


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate_vup(clf, scaler, vup_embs_by_type, vup_counts, layer_idx, component):
    rows = []
    for vup_type, embs in vup_embs_by_type.items():
        valid = [e for e in embs if e is not None]
        if not valid:
            continue
        X = np.vstack(valid)
        X_sc  = scaler.transform(X)
        preds = clf.predict(X_sc)
        probs = clf.predict_proba(X_sc)[:, 1]
        lgts  = clf.decision_function(X_sc)
        for pred, prob, logit in zip(preds, probs, lgts):
            rows.append({
                "layer":           layer_idx,
                "component":       component,
                "verb_up":         vup_type,
                "frequency":       vup_counts.get(vup_type, 0),
                "classifier_pred": int(pred),
                "up_probability":  round(float(prob),  4),
                "logit":           round(float(logit), 4),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.subword_dataset is not None:
        out_dir = args.data_dir.rstrip("/\\") + "_subword"
        log.warning(
            "--subword-dataset set without --out-dir -- writing results to %s "
            "instead of %s, so the primary (non-subword) layer results there are "
            "not overwritten. Pass --out-dir explicitly to choose a different "
            "location.", out_dir, args.data_dir,
        )
    else:
        out_dir = args.data_dir

    if os.path.abspath(out_dir) == os.path.abspath(args.data_dir) and args.subword_dataset is not None:
        raise ValueError(
            f"--out-dir resolved to the same directory as --data-dir ({args.data_dir}) "
            "while --subword-dataset is set. Refusing to proceed: this would overwrite "
            "the primary (non-subword) encoder/decoder layer results with the "
            "subword-combined run's results. Pass a different --out-dir if this is "
            "really what you want."
        )

    processor, model = load_model(args.model, args.device)
    n_enc = model.config.encoder_layers   # 12 for whisper-small
    n_dec = model.config.decoder_layers   # 12 for whisper-small

    csv_path = os.path.join(args.data_dir, "dataset.csv")
    assert os.path.exists(csv_path), f"{csv_path} not found — run build_audio_dataset.py first."
    df = pd.read_csv(csv_path)
    log.info(
        "Loaded dataset: %d rows | %d V+up types | %d train/val positives",
        len(df),
        df[df.label == "vup"]["verb_up"].nunique(),
        df.label.isin(["word_up", "standalone_up"]).sum(),
    )

    # Data-quality filters (Whisper-specific; see docstrings). Applied here,
    # before build_splits(), so train/val/test are all built from the
    # cleaned pool -- no ambiguous-position or non-adjacent V+up row can
    # enter any split.
    log.info("Applying data-quality filters...")
    df, n_dropped_ambiguous = drop_ambiguous_up_position_rows(df)
    df, n_dropped_nonadjacent = drop_nonadjacent_vup_rows(df)
    filter_counts = {
        "ambiguous_multi_up_dropped": n_dropped_ambiguous,
        "nonadjacent_vup_dropped": n_dropped_nonadjacent,
    }

    subword_df = None
    if args.subword_dataset:
        assert os.path.exists(args.subword_dataset), (
            f"{args.subword_dataset} not found — run build_subword_audio_dataset.py first."
        )
        subword_df = pd.read_csv(args.subword_dataset)
        log.info(
            "Loaded subword-up dataset: %d rows | %d unique up-word types",
            len(subword_df), subword_df["upword_type"].nunique(),
        )
        subword_df, n_dropped_ambiguous_sub = drop_ambiguous_up_position_rows(subword_df)
        filter_counts["ambiguous_multi_up_dropped_subword"] = n_dropped_ambiguous_sub

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "data_quality_filter_counts.json"), "w") as f:
        json.dump(filter_counts, f, indent=2)
    log.info("Filter counts: %s", filter_counts)

    train_df, val_df, test_df, qualifying, vup_counts = build_splits(df, subword_df=subword_df)

    # Save train/val/test splits (mirrors create_train_val_test.py's convention
    # for OLMo/BabyLM), so downstream analyses can recover exactly which rows
    # -- and which up-word types -- fed the classifier, without needing to
    # reconstruct the split from scratch.
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(  os.path.join(out_dir, "val.csv"),   index=False)
    test_df.to_csv( os.path.join(out_dir, "test.csv"),  index=False)

    # Use Dolma corpus frequencies if a pkl is provided; fall back to audio occurrence counts
    if args.corpus_stats_pkl:
        with open(args.corpus_stats_pkl, "rb") as f:
            vup_freq, verb_freq, predic = pickle.load(f)
        corpus_freq = dict(vup_freq)
        log.info("Loaded Dolma V+up frequencies from %s (%d types)", args.corpus_stats_pkl, len(vup_freq))
        freqs = sorted(corpus_freq.get(vt, 0) for vt in qualifying)
        if freqs:
            import numpy as np
            log.info(
                "Dolma frequency spread across %d qualifying V+up types: "
                "min=%d, median=%d, max=%d",
                len(freqs), freqs[0], int(np.median(freqs)), freqs[-1],
            )
    else:
        corpus_freq = vup_counts
        log.info("No --corpus-stats-pkl provided; using audio occurrence counts as frequency.")

    # ----------------------------------------------------------------
    # Extract embeddings for train, val, test in one pass each
    # ----------------------------------------------------------------
    log.info("Extracting train embeddings ...")
    enc_train, dec_train, y_train, src_train = extract_all_layers(
        train_df, processor, model, args.device, n_enc, n_dec, desc="Train"
    )

    log.info("Extracting val embeddings ...")
    enc_val, dec_val, y_val, src_val = extract_all_layers(
        val_df, processor, model, args.device, n_enc, n_dec, desc="Val"
    )

    # Re-save train/val with per-component survival flags now that extraction
    # has run: a row can succeed for the encoder (pooled over audio frames,
    # rarely fails) but fail for the decoder (no token cleanly containing
    # "up" in this row's tokenization), or vice versa -- see
    # extract_all_layers()'s docstring. All decoder layers share the same
    # target position per row (found once, reused for every layer), so
    # checking layer 0 is sufficient to know whether every layer succeeded.
    train_df["encoder_survived"] = [e is not None for e in enc_train[0]]
    train_df["decoder_survived"] = [e is not None for e in dec_train[0]]
    val_df["encoder_survived"]   = [e is not None for e in enc_val[0]]
    val_df["decoder_survived"]   = [e is not None for e in dec_val[0]]
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(  os.path.join(out_dir, "val.csv"),   index=False)

    # Test: grouped by V+up type, keep as list of (embs_enc, embs_dec) per type
    log.info("Extracting test (V+up) embeddings ...")
    test_enc_by_type = {vt: [] for vt in qualifying}
    test_dec_by_type = {vt: [] for vt in qualifying}

    for vt in qualifying:
        rows = test_df[test_df.verb_up == vt]
        enc_t, dec_t, _, _ = extract_all_layers(
            rows, processor, model, args.device, n_enc, n_dec,
            desc=f"Test {vt}",
        )
        for li in range(n_enc):
            test_enc_by_type[vt].append(enc_t[li])   # list of embs for this layer
        for li in range(n_dec):
            test_dec_by_type[vt].append(dec_t[li])

    # Reshape test: test_enc_per_layer[layer_idx][vup_type] = list of embs
    test_enc_per_layer = []
    for li in range(n_enc):
        d = {}
        for vt in qualifying:
            d[vt] = test_enc_by_type[vt][li]   # list of embs at this layer for this type
        test_enc_per_layer.append(d)

    test_dec_per_layer = []
    for li in range(n_dec):
        d = {}
        for vt in qualifying:
            d[vt] = test_dec_by_type[vt][li]
        test_dec_per_layer.append(d)

    # ----------------------------------------------------------------
    # Layer loop
    # ----------------------------------------------------------------
    for component, n_layers, enc_tr, dec_tr_or_enc_tr, enc_va, test_per_layer in [
        ("encoder", n_enc, enc_train, None, enc_val, test_enc_per_layer),
        ("decoder", n_dec, dec_train, None, dec_val, test_dec_per_layer),
    ]:
        # Alias to unify
        layer_train = enc_train if component == "encoder" else dec_train
        layer_val   = enc_val   if component == "encoder" else dec_val

        comp_dir = os.path.join(out_dir, component)
        os.makedirs(comp_dir, exist_ok=True)

        all_dfs    = []
        layer_meta = []

        log.info("=" * 60)
        log.info("COMPONENT: %s (%d layers)", component.upper(), n_layers)
        log.info("=" * 60)

        for li in range(n_layers):
            log.info("--- %s layer %d / %d ---", component.upper(), li, n_layers - 1)

            X_tr, y_tr = layer_arrays(layer_train[li], y_train)
            X_va, y_va, src_va = layer_arrays(layer_val[li], y_val, src_val)

            if len(X_tr) == 0:
                log.warning("No valid embeddings at %s layer %d — skipping", component, li)
                continue

            clf, scaler, metrics = train_classifier(X_tr, y_tr, X_va, y_va, src_val=src_va)

            layer_meta.append({
                "layer":            li,
                "component":        component,
                "train_n_positive": metrics["n_train_pos"],
                "train_n_negative": metrics["n_train_neg"],
                "val_n_positive":   metrics["n_val_pos"],
                "val_n_negative":   metrics["n_val_neg"],
                "cv_mean":          round(metrics["cv_mean"],   6),
                "cv_std":           round(metrics["cv_std"],    6),
                "val_acc":          round(metrics["val_acc"],   6),
                "val_up_acc":       round(metrics["up_acc"],    6),
                "val_other_acc":    round(metrics["other_acc"], 6),
                "val_subword_acc":  round(metrics["subword_acc"], 6) if metrics["n_subword_va"] > 0 else None,
                "val_subword_n":    metrics["n_subword_va"],
            })

            layer_df = evaluate_vup(
                clf, scaler, test_per_layer[li], corpus_freq, li, component
            )
            csv_out = os.path.join(comp_dir, f"layer_{li:02d}.csv")
            layer_df.to_csv(csv_out, index=False)
            log.info("  Saved: %s (%d rows)", csv_out, len(layer_df))
            all_dfs.append(layer_df)

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(os.path.join(comp_dir, "all_layers_results.csv"), index=False)
            log.info("Combined CSV saved for %s.", component)

        with open(os.path.join(comp_dir, "layer_metadata.json"), "w") as f:
            json.dump({
                "model": args.model, "component": component,
                "n_layers": n_layers, "random_seed": RANDOM_SEED,
                "n_test_vup_types": len(qualifying),
                "layers": layer_meta,
            }, f, indent=2)

    log.info("All done.")


if __name__ == "__main__":
    main()
