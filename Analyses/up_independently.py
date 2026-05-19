"""
V+up Layer-by-Layer Analysis — standalone "up" classifier
===========================================================
Trains a logistic regression classifier layer-by-layer on a transformer
model's hidden states to distinguish standalone "up" particles from random
other tokens.

  Positive (label=1): standalone "up" token embeddings
  Negative (label=0): random other token from the same sentences

Reads pre-built datasets from DATA_DIR (produced by create_train_val_test.py):
    train.csv, val.csv, test.csv

Per-layer outputs saved to DATA_DIR:
    layer_XX.csv, layer_XX_plot.png, all_layers_results.csv, layer_metadata.json

Usage:
    python up_independently.py [--model MODEL] [--data-dir DIR] [--vup-pkl PATH]

Requires create_train_val_test.py to have been run first.
"""

import argparse
import json
import os
import logging
import random

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG (defaults; all overridable via CLI — see --help)
# ---------------------------------------------------------------------------
BATCH_SIZE      = 64    # increased from 50; safe with multiple BabyLM models on 80 GB GPU
RANDOM_SEED     = 964
MAX_SEQ_LEN     = 128
LOAD_IN_8BIT    = False
N_TRAIN         = 1000
N_VAL           = 1000
N_TEST_PER_TYPE = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer-by-layer standalone-up classifier."
    )
    parser.add_argument(
        "--model",
        default="allenai/Olmo-3-1025-7B",
        help="HuggingFace model ID. Default: allenai/Olmo-3-1025-7B",
    )
    parser.add_argument(
        "--data-dir",
        default="../Data/olmo-3-7b/Data_up",
        help="Directory containing train/val/test CSVs and where layer outputs "
             "will be saved. Default: ../Data/olmo-3-7b/Data_up",
    )
    parser.add_argument(
        "--vup-pkl",
        default="../Data/corpus_results.pkl",
        help="Path to corpus_results.pkl (produced by create_dataset.py). "
             "Default: ../Data/corpus_results.pkl",
    )
    return parser.parse_args()


args         = parse_args()
MODEL_NAME   = args.model
DATA_DIR     = args.data_dir
VUP_PKL_PATH = args.vup_pkl

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------

def load_model():
    log.info("Loading model: %s", MODEL_NAME)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    except Exception:
        log.warning("Fast tokenizer failed, falling back to use_fast=False")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, use_fast=False, trust_remote_code=True
        )
    if LOAD_IN_8BIT:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, load_in_8bit=True, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
        )
    model.eval()
    log.info(
        "Model loaded. Layers: %d | hidden size: %d",
        model.config.num_hidden_layers, model.config.hidden_size,
    )
    return tokenizer, model


# ---------------------------------------------------------------------------
# LOAD DATASETS
# ---------------------------------------------------------------------------

def load_datasets():
    """
    Load train.csv, val.csv, and test.csv from DATA_DIR and reconstruct
    position records for the layer loop.

    Returns:
        train_records          : list of (sentence, token_pos, label)
        val_records            : list of (sentence, token_pos, label)
        vup_positions          : dict {vup_type: [(sentence, token_pos, word), ...]}
        vup_sentences_filtered : dict {vup_type: [sentence, ...]}
        vup_predic             : dict {vup_type: float}
        vup_freq               : dict {vup_type: int}  — from test.csv frequency column
    """
    for fname in ("train.csv", "val.csv", "test.csv"):
        path = os.path.join(DATA_DIR, fname)
        assert os.path.exists(path), (
            f"{path} not found — run build_datasets.py first."
        )

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    log.info("Loaded train.csv (%d rows) | val.csv (%d rows) | test.csv (%d rows, %d types)",
             len(train_df), len(val_df), len(test_df), test_df["verb_up"].nunique())
    log.info("  Train — label=1: %d | label=0: %d",
             (train_df.label == 1).sum(), (train_df.label == 0).sum())
    log.info("  Val   — label=1: %d | label=0: %d",
             (val_df.label == 1).sum(), (val_df.label == 0).sum())

    train_records = list(zip(
        train_df["sentence"],
        train_df["token_position"].astype(int),
        train_df["label"],
    ))
    val_records = list(zip(
        val_df["sentence"],
        val_df["token_position"].astype(int),
        val_df["label"],
    ))

    vup_positions          = {}
    vup_sentences_filtered = {}
    for vup_type, group in test_df.groupby("verb_up"):
        vup_positions[vup_type]          = list(zip(
            group["sentence"],
            group["token_position"].astype(int),
            group["word"],
        ))
        vup_sentences_filtered[vup_type] = group["sentence"].tolist()

    vup_predic = {}
    if "predic" in test_df.columns:
        vup_predic = (
            test_df.drop_duplicates("verb_up")
            .set_index("verb_up")["predic"]
            .to_dict()
        )
        log.info("  Predictability loaded for %d V+up types.", len(vup_predic))
    else:
        log.info("  No 'predic' column in test.csv — predictability will be NaN in outputs.")

    vup_freq = (
        test_df.drop_duplicates("verb_up")
        .set_index("verb_up")["frequency"]
        .to_dict()
    )
    log.info("  Frequency loaded for %d V+up types.", len(vup_freq))

    return train_records, val_records, vup_positions, vup_sentences_filtered, vup_predic, vup_freq


# ---------------------------------------------------------------------------
# EMBEDDING EXTRACTION — single forward pass, all layers simultaneously
# ---------------------------------------------------------------------------

def extract_all_layers_records(records, model, tokenizer, n_layers, desc=""):
    """
    One forward pass through records, collecting the target-token hidden state
    for every layer at once.

    Returns dict {layer_idx: (X, y)} where X is np.ndarray (n, hidden_size)
    and y is np.ndarray (n,).
    """
    sentences = [s for s, _, _ in records]
    positions = [p for _, p, _ in records]
    labels    = [l for _, _, l in records]

    layer_X = [[] for _ in range(n_layers)]
    layer_y = [[] for _ in range(n_layers)]

    device = next(model.parameters()).device

    with tqdm(total=len(sentences), desc=desc, unit="sent", leave=False) as pbar:
        for batch_start in range(0, len(sentences), BATCH_SIZE):
            batch_sents = sentences[batch_start : batch_start + BATCH_SIZE]
            batch_pos   = positions[batch_start : batch_start + BATCH_SIZE]
            batch_label = labels[batch_start   : batch_start + BATCH_SIZE]

            encoded = tokenizer(
                batch_sents, return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_SEQ_LEN,
            )
            input_ids      = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            actual_lens = attention_mask.sum(dim=1).tolist()

            for layer_idx in range(n_layers):
                hidden = outputs.hidden_states[layer_idx + 1]
                for i, (pos, lbl, alen) in enumerate(zip(batch_pos, batch_label, actual_lens)):
                    if pos < alen:
                        layer_X[layer_idx].append(hidden[i, pos, :].float().cpu().numpy())
                        layer_y[layer_idx].append(lbl)

            del outputs, input_ids, attention_mask
            pbar.update(len(batch_sents))

    return {li: (np.vstack(layer_X[li]), np.array(layer_y[li])) for li in range(n_layers)}


def extract_all_layers_vup(vup_positions, model, tokenizer, n_layers):
    """
    One forward pass through all V+up test sentences, collecting the 'up'
    token hidden state for every layer at once.

    Returns dict {layer_idx: {vup_type: np.ndarray (n_sentences, hidden_size)}}.
    Memory is freed per-layer during reorganisation so peak usage is bounded.
    """
    all_sents, all_pos, type_index = [], [], []
    for vup_type, records in vup_positions.items():
        for sent, pos, _ in records:
            all_sents.append(sent)
            all_pos.append(pos)
            type_index.append(vup_type)

    n_total = len(all_sents)
    log.info("  Extracting V+up embeddings (all layers): %d sentences, %d types ...",
             n_total, len(vup_positions))

    # Build per-layer, per-type lists directly during the forward pass
    layer_vup_embs = [{vt: [] for vt in vup_positions} for _ in range(n_layers)]
    skipped = 0

    device = next(model.parameters()).device

    with tqdm(total=n_total, desc="Test (all layers)", unit="sent", leave=False) as pbar:
        for batch_start in range(0, n_total, BATCH_SIZE):
            batch_sents = all_sents[batch_start : batch_start + BATCH_SIZE]
            batch_pos   = all_pos[batch_start  : batch_start + BATCH_SIZE]
            batch_types = type_index[batch_start : batch_start + BATCH_SIZE]

            encoded = tokenizer(
                batch_sents, return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_SEQ_LEN,
            )
            input_ids      = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            actual_lens = attention_mask.sum(dim=1).tolist()

            for layer_idx in range(n_layers):
                hidden = outputs.hidden_states[layer_idx + 1]
                for i, (pos, alen, vt) in enumerate(zip(batch_pos, actual_lens, batch_types)):
                    if pos < alen:
                        layer_vup_embs[layer_idx][vt].append(
                            hidden[i, pos, :].float().cpu().numpy()
                        )
                    elif layer_idx == 0:
                        skipped += 1

            del outputs, input_ids, attention_mask
            pbar.update(len(batch_sents))

    if skipped:
        log.warning("  %d test sentences skipped (position beyond truncation)", skipped)

    result = {}
    for layer_idx in range(n_layers):
        result[layer_idx] = {
            k: np.vstack(v) for k, v in layer_vup_embs[layer_idx].items() if v
        }
        layer_vup_embs[layer_idx] = None  # free raw lists immediately
        log.info("  Layer %d V+up done: %d types", layer_idx, len(result[layer_idx]))

    return result


# ---------------------------------------------------------------------------
# CLASSIFIER
# ---------------------------------------------------------------------------

def train_classifier(X_train, y_train, X_val, y_val):
    """Logistic regression with class balancing by majority truncation."""
    pos_idx_tr = np.where(y_train == 1)[0]
    neg_idx_tr = np.where(y_train == 0)[0]
    n_tr       = min(len(pos_idx_tr), len(neg_idx_tr))
    idx_tr     = np.concatenate([pos_idx_tr[:n_tr], neg_idx_tr[:n_tr]])
    X_tr, y_tr = X_train[idx_tr], y_train[idx_tr]

    pos_idx_va = np.where(y_val == 1)[0]
    neg_idx_va = np.where(y_val == 0)[0]
    n_va       = min(len(pos_idx_va), len(neg_idx_va))
    idx_va     = np.concatenate([pos_idx_va[:n_va], neg_idx_va[:n_va]])
    X_va, y_va = X_val[idx_va], y_val[idx_va]

    log.info("  Train: %d pos + %d neg | Val: %d pos + %d neg", n_tr, n_tr, n_va, n_va)

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
    clf.fit(X_tr_sc, y_tr)

    cv      = cross_val_score(clf, X_tr_sc, y_tr, cv=5, scoring="accuracy")
    cv_mean, cv_std = cv.mean(), cv.std()

    val_preds = clf.predict(X_va_sc)
    val_acc   = (val_preds == y_va).mean()
    up_acc    = (val_preds[y_va == 1] == 1).mean()
    other_acc = (val_preds[y_va == 0] == 0).mean()

    log.info("  Train CV: %.3f±%.3f | Val: %.3f (up=%.3f, other=%.3f)",
             cv_mean, cv_std, val_acc, up_acc, other_acc)

    return clf, scaler, {
        "cv_mean": cv_mean, "cv_std": cv_std,
        "val_acc": val_acc, "up_acc": up_acc, "other_acc": other_acc,
        "n_train_pos": int(n_tr), "n_train_neg": int(n_tr),
        "n_val_pos":   int(n_va), "n_val_neg":   int(n_va),
    }


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate_vup(clf, scaler, vup_positions, vup_embeddings, vup_freq, vup_predic, layer_idx):
    rows = []
    for vup_type, embs in vup_embeddings.items():
        X_scaled  = scaler.transform(embs)
        preds     = clf.predict(X_scaled)
        probs     = clf.predict_proba(X_scaled)[:, 1]
        logits    = clf.decision_function(X_scaled)
        sents     = [sent for sent, _, _ in vup_positions[vup_type]]
        predic_val = vup_predic.get(vup_type, float("nan"))

        for j, (pred, prob, logit) in enumerate(zip(preds, probs, logits)):
            rows.append({
                "layer":           layer_idx,
                "verb_up":         vup_type,
                "frequency":       vup_freq[vup_type],
                "predic":          predic_val,
                "sentence":        sents[j] if j < len(sents) else "",
                "classifier_pred": int(pred),
                "up_probability":  round(float(prob), 4),
                "logit":           round(float(logit), 4),
            })

    return pd.DataFrame(rows).sort_values(
        ["frequency", "verb_up"], ascending=[False, True]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def make_plot(results_df, layer_idx, val_metrics, save_path):
    summary_df = (
        results_df
        .groupby(["verb_up", "frequency"])
        .agg(
            mean_up_prob   = ("up_probability", "mean"),
            classifier_acc = ("classifier_pred", "mean"),
            mean_logit     = ("logit", "mean"),
            n_sentences    = ("classifier_pred", "count"),
        )
        .reset_index()
        .sort_values("frequency", ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"Layer {layer_idx}  |  Val acc={val_metrics['val_acc']:.3f}  "
        f"(up={val_metrics['up_acc']:.3f}, other={val_metrics['other_acc']:.3f})  "
        f"|  Train CV={val_metrics['cv_mean']:.3f}±{val_metrics['cv_std']:.3f}\n"
        f"Classifier: standalone 'up' (1) vs. random other tokens (0)",
        fontsize=10,
    )

    ax       = axes[0]
    log_freq = np.log10(summary_df["frequency"])
    logits   = summary_df["mean_logit"].values

    sc = ax.scatter(
        log_freq, logits,
        c=summary_df["mean_up_prob"],
        cmap="RdYlGn", alpha=0.7, edgecolors="grey", linewidths=0.3, s=60,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Mean P(standalone-up-like)", fontsize=10)

    poly_deg = 2
    try:
        coeffs   = np.polyfit(log_freq, logits, poly_deg)
        x_smooth = np.linspace(log_freq.min(), log_freq.max(), 300)
        y_smooth = np.polyval(coeffs, x_smooth)
        ax.plot(x_smooth, y_smooth, color="steelblue", linewidth=2,
                label=f"Poly fit (deg={poly_deg})")
    except np.linalg.LinAlgError:
        log.warning("polyfit SVD did not converge at layer %d — skipping trend line", layer_idx)

    for _, row in summary_df.nlargest(5, "mean_logit").iterrows():
        ax.annotate(row["verb_up"], (np.log10(row["frequency"]), row["mean_logit"]),
                    fontsize=7, alpha=0.85, xytext=(4, 2), textcoords="offset points")
    for _, row in summary_df.nsmallest(5, "mean_logit").iterrows():
        ax.annotate(row["verb_up"], (np.log10(row["frequency"]), row["mean_logit"]),
                    fontsize=7, alpha=0.85, color="firebrick",
                    xytext=(4, -8), textcoords="offset points")

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8,
               label="Decision boundary (logit=0)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(10**x):,}"))
    ax.set_xlabel("Corpus frequency (log scale)", fontsize=11)
    ax.set_ylabel("Mean logit (higher = more standalone-'up'-like)", fontsize=11)
    ax.set_title(f"Compositionality of V+up — Layer {layer_idx}", fontsize=12)
    ax.legend(fontsize=9)

    corr = summary_df["frequency"].apply(np.log10).corr(
        summary_df["mean_logit"], method="spearman"
    )
    ax.text(0.02, 0.02, f"Spearman r = {corr:.3f}", transform=ax.transAxes,
            fontsize=9, color="dimgray")

    ax2    = axes[1]
    n_show = 20
    bar_df = pd.concat([
        summary_df.nlargest(n_show, "mean_logit"),
        summary_df.nsmallest(n_show, "mean_logit"),
    ]).drop_duplicates("verb_up").sort_values("mean_logit", ascending=True)

    colors = ["#d73027" if l < 0 else "#1a9850" for l in bar_df["mean_logit"]]
    bars   = ax2.barh(bar_df["verb_up"], bar_df["mean_logit"], color=colors, alpha=0.8)

    for bar, (_, row) in zip(bars, bar_df.iterrows()):
        x_pos = bar.get_width() + 0.05 if bar.get_width() >= 0 else bar.get_width() - 0.05
        ha    = "left" if bar.get_width() >= 0 else "right"
        ax2.text(x_pos, bar.get_y() + bar.get_height() / 2,
                 f"n={row['frequency']:,}", va="center", ha=ha, fontsize=7, color="dimgray")

    ax2.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Mean logit", fontsize=11)
    ax2.set_title(f"Most & least compositional V+up — Layer {layer_idx}\n(top/bottom {n_show})",
                  fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot saved: %s", save_path)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    tokenizer, model = load_model()
    n_layers = model.config.num_hidden_layers
    log.info("Will iterate over %d transformer layers (0 to %d)", n_layers, n_layers - 1)

    train_records, val_records, vup_positions, vup_sentences_filtered, vup_predic, vup_freq = load_datasets()

    log.info("Extracting train embeddings (single pass, all layers) ...")
    train_all = extract_all_layers_records(
        train_records, model, tokenizer, n_layers, desc="Train"
    )
    log.info("Extracting val embeddings (single pass, all layers) ...")
    val_all = extract_all_layers_records(
        val_records, model, tokenizer, n_layers, desc="Val"
    )
    log.info("Extracting test (V+up) embeddings (single pass, all layers) ...")
    vup_all = extract_all_layers_vup(vup_positions, model, tokenizer, n_layers)

    all_layer_dfs  = []
    layer_metadata = []

    for layer_idx in range(n_layers):
        log.info("=" * 60)
        log.info("LAYER %d / %d", layer_idx, n_layers - 1)
        log.info("=" * 60)

        X_train, y_train = train_all[layer_idx]
        X_val,   y_val   = val_all[layer_idx]
        vup_embeddings   = vup_all[layer_idx]

        clf, scaler, metrics = train_classifier(X_train, y_train, X_val, y_val)

        layer_metadata.append({
            "layer":            layer_idx,
            "train_n_positive": metrics["n_train_pos"],
            "train_n_negative": metrics["n_train_neg"],
            "train_n_total":    metrics["n_train_pos"] + metrics["n_train_neg"],
            "val_n_positive":   metrics["n_val_pos"],
            "val_n_negative":   metrics["n_val_neg"],
            "val_n_total":      metrics["n_val_pos"] + metrics["n_val_neg"],
            "cv_mean":          round(metrics["cv_mean"],   6),
            "cv_std":           round(metrics["cv_std"],    6),
            "val_acc":          round(metrics["val_acc"],   6),
            "val_up_acc":       round(metrics["up_acc"],    6),
            "val_other_acc":    round(metrics["other_acc"], 6),
        })

        layer_df = evaluate_vup(clf, scaler, vup_positions, vup_embeddings, vup_freq, vup_predic, layer_idx)

        csv_path = os.path.join(DATA_DIR, f"layer_{layer_idx:02d}.csv")
        layer_df.to_csv(csv_path, index=False)
        log.info("  Saved: %s (%d rows)", csv_path, len(layer_df))

        plot_path = os.path.join(DATA_DIR, f"layer_{layer_idx:02d}_plot.png")
        make_plot(layer_df, layer_idx, metrics, plot_path)

        all_layer_dfs.append(layer_df)

        del train_all[layer_idx], val_all[layer_idx], vup_all[layer_idx]
        del X_train, y_train, X_val, y_val, vup_embeddings
        torch.cuda.empty_cache()

    combined_df = pd.concat(all_layer_dfs, ignore_index=True)
    combined_df.to_csv(os.path.join(DATA_DIR, "all_layers_results.csv"), index=False)
    log.info("=" * 60)
    log.info("Done. Combined CSV saved.")

    metadata = {
        "model":            MODEL_NAME,
        "n_layers":         n_layers,
        "random_seed":      RANDOM_SEED,
        "n_train":          N_TRAIN,
        "n_val":            N_VAL,
        "n_test_per_type":  N_TEST_PER_TYPE,
        "n_test_vup_types": len(vup_sentences_filtered),
        "classifier":       "LogisticRegression(C=1.0, max_iter=1000)",
        "layers":           layer_metadata,
    }
    with open(os.path.join(DATA_DIR, "layer_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Layer metadata saved.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()