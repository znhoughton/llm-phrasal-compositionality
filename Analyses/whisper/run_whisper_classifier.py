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
    DATA_DIR/dataset.csv  (built by build_whisper_dataset.py)

Outputs:
    DATA_DIR/encoder/layer_XX.csv, all_layers_results.csv, layer_metadata.json
    DATA_DIR/decoder/layer_XX.csv, all_layers_results.csv, layer_metadata.json

Usage:
    python run_whisper_classifier.py [--data-dir DIR] [--model MODEL] [--device DEVICE]

Requires:
    pip install transformers torch soundfile scikit-learn pandas numpy tqdm
"""

import argparse
import json
import logging
import os
import pickle
import random

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
N_VAL           = 200    # balanced pos+neg each for val
N_TEST_PER_TYPE = 20     # max test sentences per V+up type
MIN_FREQ_VUP    = 5      # min occurrences to include a V+up type in test

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
        help="Directory containing dataset.csv and where outputs will be saved. "
             "Default: ../../Data/whisper",
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
        "--vup-pkl", default=None,
        help="Path to corpus_results.pkl (from create_dataset.py / C4). "
             "If provided, the 'frequency' column uses C4 counts instead of "
             "LibriSpeech occurrence counts. Default: None",
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
# SPLITS
# ---------------------------------------------------------------------------

def build_splits(df):
    standalone = df[df.label == "standalone_up"].copy()
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

    # Train/val from standalone (positive) and same rows shifted to neg timestamps
    standalone = standalone.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    n_pos = len(standalone)
    n_train_pos = min(N_TRAIN, int(n_pos * 0.8))
    n_val_pos   = min(N_VAL, n_pos - n_train_pos)

    train_pos = standalone.iloc[:n_train_pos].copy()
    val_pos   = standalone.iloc[n_train_pos : n_train_pos + n_val_pos].copy()

    def make_neg(rows):
        neg = rows.copy()
        neg["up_start"] = neg["neg_start"]
        neg["up_end"]   = neg["neg_end"]
        neg["target"]   = 0
        return neg

    train_pos["target"] = 1
    val_pos["target"]   = 1
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


def extract_all_layers(df, processor, model, device, n_enc, n_dec, desc=""):
    """
    Run one Whisper forward pass per row (audio file) and collect hidden states
    at the 'up' position for all encoder and decoder layers simultaneously.

    Returns:
        enc[layer_idx] : list of np.ndarray  (one per row, or None if skipped)
        dec[layer_idx] : list of np.ndarray  (one per row, or None if skipped)
        targets        : list of int labels (from df["target"] if present, else 1)
    """
    enc  = [[] for _ in range(n_enc)]
    dec  = [[] for _ in range(n_dec)]
    targets = []
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
            # Positives (target=1): find the "up" token.
            # Negatives (target=0): find the neg_word token so the classifier
            #   sees genuinely different embeddings for the two classes.
            tokens     = decoder_input_ids[0].tolist()
            target_val = int(row["target"]) if "target" in row else 1

            if target_val == 1:
                target_ids = up_ids
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

        except Exception as e:
            log.debug("Skipped: %s", e)
            for li in range(n_enc):
                enc[li].append(None)
            for li in range(n_dec):
                dec[li].append(None)
            targets.append(int(row["target"]) if "target" in row else 1)

    return enc, dec, targets


def layer_arrays(layer_embs, targets):
    """Filter out None entries, return (X, y) numpy arrays."""
    X, y = [], []
    for emb, lbl in zip(layer_embs, targets):
        if emb is not None:
            X.append(emb)
            y.append(lbl)
    if not X:
        return np.zeros((0, 1)), np.zeros(0, dtype=int)
    return np.vstack(X), np.array(y)


# ---------------------------------------------------------------------------
# CLASSIFIER
# ---------------------------------------------------------------------------

def train_classifier(X_train, y_train, X_val, y_val):
    """Logistic regression with class balancing by majority truncation."""
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

    log.info(
        "  CV: %.3f±%.3f | Val: %.3f (up=%.3f, other=%.3f)",
        cv.mean(), cv.std(), val_acc, up_acc, oth_acc,
    )
    return clf, scaler, {
        "cv_mean": float(cv.mean()), "cv_std": float(cv.std()),
        "val_acc": float(val_acc),   "up_acc": float(up_acc),   "other_acc": float(oth_acc),
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

    processor, model = load_model(args.model, args.device)
    n_enc = model.config.encoder_layers   # 12 for whisper-small
    n_dec = model.config.decoder_layers   # 12 for whisper-small

    csv_path = os.path.join(args.data_dir, "dataset.csv")
    assert os.path.exists(csv_path), f"{csv_path} not found — run build_whisper_dataset.py first."
    df = pd.read_csv(csv_path)
    log.info(
        "Loaded dataset: %d rows | %d V+up types | %d standalone",
        len(df),
        df[df.label == "vup"]["verb_up"].nunique(),
        (df.label == "standalone_up").sum(),
    )

    train_df, val_df, test_df, qualifying, vup_counts = build_splits(df)

    # Use C4 corpus frequencies if a pkl is provided; fall back to LibriSpeech counts
    if args.vup_pkl:
        with open(args.vup_pkl, "rb") as f:
            vup_sentences, vup_freq, up_sentences, up_freq = pickle.load(f)
        c4_freq = dict(vup_freq)
        log.info("Loaded C4 V+up frequencies from %s (%d types)", args.vup_pkl, len(vup_freq))
        # Report frequency spread for qualifying types
        freqs = sorted(c4_freq.get(vt, 0) for vt in qualifying)
        if freqs:
            import numpy as np
            log.info(
                "C4 frequency spread across %d qualifying V+up types: "
                "min=%d, median=%d, max=%d",
                len(freqs), freqs[0], int(np.median(freqs)), freqs[-1],
            )
    else:
        c4_freq = vup_counts
        log.info("No --vup-pkl provided; using LibriSpeech occurrence counts as frequency.")

    # ----------------------------------------------------------------
    # Extract embeddings for train, val, test in one pass each
    # ----------------------------------------------------------------
    log.info("Extracting train embeddings ...")
    enc_train, dec_train, y_train = extract_all_layers(
        train_df, processor, model, args.device, n_enc, n_dec, desc="Train"
    )

    log.info("Extracting val embeddings ...")
    enc_val, dec_val, y_val = extract_all_layers(
        val_df, processor, model, args.device, n_enc, n_dec, desc="Val"
    )

    # Test: grouped by V+up type, keep as list of (embs_enc, embs_dec) per type
    log.info("Extracting test (V+up) embeddings ...")
    test_enc_by_type = {vt: [] for vt in qualifying}
    test_dec_by_type = {vt: [] for vt in qualifying}

    for vt in qualifying:
        rows = test_df[test_df.verb_up == vt]
        enc_t, dec_t, _ = extract_all_layers(
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

        comp_dir = os.path.join(args.data_dir, component)
        os.makedirs(comp_dir, exist_ok=True)

        all_dfs    = []
        layer_meta = []

        log.info("=" * 60)
        log.info("COMPONENT: %s (%d layers)", component.upper(), n_layers)
        log.info("=" * 60)

        for li in range(n_layers):
            log.info("--- %s layer %d / %d ---", component.upper(), li, n_layers - 1)

            X_tr, y_tr = layer_arrays(layer_train[li], y_train)
            X_va, y_va = layer_arrays(layer_val[li],   y_val)

            if len(X_tr) == 0:
                log.warning("No valid embeddings at %s layer %d — skipping", component, li)
                continue

            clf, scaler, metrics = train_classifier(X_tr, y_tr, X_va, y_va)

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
            })

            layer_df = evaluate_vup(
                clf, scaler, test_per_layer[li], c4_freq, li, component
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
