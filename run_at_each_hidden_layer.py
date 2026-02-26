"""
V+up Layer-by-Layer Analysis
==============================
Loads pre-collected corpus results from corpus_results.pkl, then for each
hidden layer in OLMo-3 7B:
  1. Extracts the "up" token embedding at that layer
  2. Trains a logistic regression (standalone "up" vs. other tokens)
  3. Evaluates on V+up phrases
  4. Saves per-layer DataFrame to Data/layer_{i}.csv
  5. Saves per-layer plot to Data/layer_{i}_plot.png

At the end, merges all layers into a single Data/all_layers_results.csv
with a `layer` column.

Usage:
    python vup_layer_analysis.py

Expects corpus_results.pkl in the current directory (produced by the
notebook corpus collection cells).
"""

import os
import pickle
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
# CONFIG
# ---------------------------------------------------------------------------
MODEL_NAME      = "allenai/Olmo-3-1025-7B"
BATCH_SIZE      = 32
RANDOM_SEED     = 42
MAX_SEQ_LEN     = 128
LOAD_IN_8BIT    = False
DATA_DIR        = "Data"
PKL_PATH        = "corpus_results.pkl"

# Train/val split sizes (corpus_results.pkl must have >= 2000 up_sentences)
N_TRAIN         = 1000
N_VAL           = 1000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# LOAD CORPUS RESULTS
# ---------------------------------------------------------------------------

def load_corpus_results(pkl_path):
    log.info("Loading corpus results from %s ...", pkl_path)
    with open(pkl_path, "rb") as f:
        vup_sentences, vup_freq, up_sentences, up_freq = pickle.load(f)
    log.info(
        "Loaded: %d V+up types | %d standalone 'up' sentences | up_freq=%d",
        len(vup_sentences), len(up_sentences), up_freq,
    )
    assert len(up_sentences) >= N_TRAIN + N_VAL, (
        f"Need at least {N_TRAIN + N_VAL} standalone 'up' sentences, "
        f"got {len(up_sentences)}. Re-run corpus collection with N_STANDALONE_UP=2000."
    )
    return vup_sentences, vup_freq, up_sentences, up_freq


# ---------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------

def load_olmo():
    log.info("Loading OLMo model: %s", MODEL_NAME)
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
        model.config.num_hidden_layers,
        model.config.hidden_size,
    )
    return tokenizer, model


# ---------------------------------------------------------------------------
# EMBEDDING EXTRACTION
# ---------------------------------------------------------------------------

def find_up_token_index(tokenizer, input_ids):
    tokens = [tokenizer.decode([tid]).strip().lower() for tid in input_ids]
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] == "up":
            return i
    return None


def extract_embeddings_at_layer(
    sentences,
    tokenizer,
    model,
    layer_idx,
    desc="Extracting",
    also_extract_random_other=False,
):
    """
    Extract hidden states at `layer_idx` (0 = first transformer layer,
    model.config.num_hidden_layers - 1 = last transformer layer).
    hidden_states tuple from HF: index 0 = embedding layer, index 1..N = transformer layers.
    So hidden_states[layer_idx + 1] gives transformer layer `layer_idx`.
    """
    all_embeddings   = []
    other_embeddings = []
    valid_indices    = []
    skipped          = 0
    device           = next(model.parameters()).device
    rng              = np.random.default_rng(RANDOM_SEED)

    with tqdm(total=len(sentences), desc=desc, unit="sent", leave=False) as pbar:
        for batch_start in range(0, len(sentences), BATCH_SIZE):
            batch   = sentences[batch_start : batch_start + BATCH_SIZE]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN,
            )
            input_ids      = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            # hidden_states[0] = embedding layer
            # hidden_states[layer_idx + 1] = transformer layer layer_idx
            hidden = outputs.hidden_states[layer_idx + 1]

            for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
                actual_len = mask.sum().item()
                ids_list   = ids[:actual_len].tolist()
                up_pos     = find_up_token_index(tokenizer, ids_list)

                if up_pos is None:
                    skipped += 1
                    continue

                all_embeddings.append(hidden[i, up_pos, :].float().cpu().numpy())
                valid_indices.append(batch_start + i)

                if also_extract_random_other:
                    non_up_positions = [
                        j for j in range(actual_len)
                        if j != up_pos
                        and tokenizer.decode([ids_list[j]]).strip().lower() not in ("", "up")
                        and ids_list[j] not in tokenizer.all_special_ids
                    ]
                    if non_up_positions:
                        rand_pos = rng.choice(non_up_positions)
                        other_embeddings.append(
                            hidden[i, rand_pos, :].float().cpu().numpy()
                        )
                    else:
                        other_embeddings.append(None)

            pbar.update(len(batch))
            pbar.set_postfix({"ok": len(all_embeddings), "skip": skipped})

    if not all_embeddings:
        raise ValueError(f"No 'up' tokens found for layer {layer_idx}: {desc}")

    up_embs = np.vstack(all_embeddings)

    if also_extract_random_other:
        valid_other = [(i, e) for i, e in enumerate(other_embeddings) if e is not None]
        if not valid_other:
            raise ValueError(f"No valid other-token embeddings at layer {layer_idx}")
        _, other_embs = zip(*valid_other)
        other_embs = np.vstack(other_embs)
        return up_embs, valid_indices, other_embs

    return up_embs, valid_indices


# ---------------------------------------------------------------------------
# CLASSIFIER
# ---------------------------------------------------------------------------

def train_classifier(X_pos_train, X_neg_train, X_pos_val, X_neg_val):
    min_train = min(len(X_pos_train), len(X_neg_train))
    min_val   = min(len(X_pos_val),   len(X_neg_val))

    X_pos_train = X_pos_train[:min_train]
    X_neg_train = X_neg_train[:min_train]
    X_pos_val   = X_pos_val[:min_val]
    X_neg_val   = X_neg_val[:min_val]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(np.vstack([X_pos_train, X_neg_train]))
    y_train = np.array([1] * min_train + [0] * min_train)

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
    clf.fit(X_train, y_train)

    cv      = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
    cv_mean = cv.mean()
    cv_std  = cv.std()

    X_val     = scaler.transform(np.vstack([X_pos_val, X_neg_val]))
    y_val     = np.array([1] * min_val + [0] * min_val)
    val_preds = clf.predict(X_val)
    val_acc   = (val_preds == y_val).mean()
    up_acc    = (val_preds[:min_val] == 1).mean()
    other_acc = (val_preds[min_val:] == 0).mean()

    return clf, scaler, {
        "cv_mean": cv_mean, "cv_std": cv_std,
        "val_acc": val_acc, "up_acc": up_acc, "other_acc": other_acc,
    }


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate_vup(clf, scaler, vup_sentences, vup_embeddings, vup_freq, layer_idx):
    rows = []
    for vup_type, embs in vup_embeddings.items():
        X_scaled = scaler.transform(embs)
        preds    = clf.predict(X_scaled)
        probs    = clf.predict_proba(X_scaled)[:, 1]
        logits   = clf.decision_function(X_scaled)

        for j, (pred, prob, logit) in enumerate(zip(preds, probs, logits)):
            rows.append({
                "layer":           layer_idx,
                "verb_up":         vup_type,
                "frequency":       vup_freq[vup_type],
                "sentence":        vup_sentences[vup_type][j],
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
        f"|  Train CV={val_metrics['cv_mean']:.3f}±{val_metrics['cv_std']:.3f}",
        fontsize=11,
    )

    # --- Scatter ---
    ax = axes[0]
    log_freq = np.log10(summary_df["frequency"])
    logits   = summary_df["mean_logit"].values

    sc = ax.scatter(
        log_freq, logits,
        c=summary_df["mean_up_prob"],
        cmap="RdYlGn", alpha=0.7, edgecolors="grey", linewidths=0.3, s=60,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Mean P(up-like)", fontsize=10)

    # Polynomial fit
    poly_deg = 2
    coeffs   = np.polyfit(log_freq, logits, poly_deg)
    x_smooth = np.linspace(log_freq.min(), log_freq.max(), 300)
    y_smooth = np.polyval(coeffs, x_smooth)
    ax.plot(x_smooth, y_smooth, color="steelblue", linewidth=2,
            label=f"Poly fit (deg={poly_deg})")

    for _, row in summary_df.nlargest(5, "mean_logit").iterrows():
        ax.annotate(row["verb_up"], (np.log10(row["frequency"]), row["mean_logit"]),
                    fontsize=7, alpha=0.85, xytext=(4, 2), textcoords="offset points")
    for _, row in summary_df.nsmallest(5, "mean_logit").iterrows():
        ax.annotate(row["verb_up"], (np.log10(row["frequency"]), row["mean_logit"]),
                    fontsize=7, alpha=0.85, color="firebrick",
                    xytext=(4, -8), textcoords="offset points")

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8,
               label="Decision boundary (logit=0)")
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(10**x):,}")
    )
    ax.set_xlabel("Corpus frequency (log scale)", fontsize=11)
    ax.set_ylabel("Mean logit (higher = more 'up-like')", fontsize=11)
    ax.set_title(f"Compositionality of V+up — Layer {layer_idx}", fontsize=12)
    ax.legend(fontsize=9)

    # Spearman correlation
    corr = summary_df["frequency"].apply(np.log10).corr(
        summary_df["mean_logit"], method="spearman"
    )
    ax.text(0.02, 0.02, f"Spearman r = {corr:.3f}", transform=ax.transAxes,
            fontsize=9, color="dimgray")

    # --- Bar chart ---
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
                 f"n={row['frequency']:,}", va="center", ha=ha,
                 fontsize=7, color="dimgray")

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
    # Load corpus
    vup_sentences, vup_freq, up_sentences, up_freq = load_corpus_results(PKL_PATH)

    # Load model
    tokenizer, model = load_olmo()
    n_layers = model.config.num_hidden_layers
    log.info("Will iterate over %d transformer layers (0 to %d)", n_layers, n_layers - 1)

    all_layer_dfs = []

    for layer_idx in range(n_layers):
        log.info("=" * 60)
        log.info("LAYER %d / %d", layer_idx, n_layers - 1)
        log.info("=" * 60)

        # --- Extract standalone 'up' + other token embeddings ---
        log.info("  Extracting standalone 'up' embeddings at layer %d ...", layer_idx)
        up_embs, _, other_embs = extract_embeddings_at_layer(
            up_sentences, tokenizer, model, layer_idx,
            desc=f"Layer {layer_idx}: standalone 'up'",
            also_extract_random_other=True,
        )

        # Train/val split
        up_train,    up_val    = up_embs[:N_TRAIN],    up_embs[N_TRAIN:N_TRAIN + N_VAL]
        other_train, other_val = other_embs[:N_TRAIN], other_embs[N_TRAIN:N_TRAIN + N_VAL]

        # --- Train classifier ---
        clf, scaler, metrics = train_classifier(up_train, other_train, up_val, other_val)
        log.info(
            "  Layer %d — Train CV: %.3f±%.3f | Val: %.3f (up=%.3f, other=%.3f)",
            layer_idx, metrics["cv_mean"], metrics["cv_std"],
            metrics["val_acc"], metrics["up_acc"], metrics["other_acc"],
        )

        # --- Extract V+up embeddings ---
        log.info("  Extracting V+up embeddings at layer %d ...", layer_idx)
        vup_embeddings = {}
        for vup_type, sents in tqdm(
            vup_sentences.items(), desc=f"Layer {layer_idx}: V+up", unit="type", leave=False
        ):
            embs, _ = extract_embeddings_at_layer(
                sents, tokenizer, model, layer_idx,
                desc=f"  {vup_type}",
            )
            if len(embs) > 0:
                vup_embeddings[vup_type] = embs

        # --- Evaluate ---
        layer_df = evaluate_vup(clf, scaler, vup_sentences, vup_embeddings, vup_freq, layer_idx)

        # --- Save per-layer CSV ---
        csv_path = os.path.join(DATA_DIR, f"layer_{layer_idx:02d}.csv")
        layer_df.to_csv(csv_path, index=False)
        log.info("  Saved: %s (%d rows)", csv_path, len(layer_df))

        # --- Save per-layer plot ---
        plot_path = os.path.join(DATA_DIR, f"layer_{layer_idx:02d}_plot.png")
        make_plot(layer_df, layer_idx, metrics, plot_path)

        all_layer_dfs.append(layer_df)

        # --- Free memory ---
        del up_embs, other_embs, up_train, up_val, other_train, other_val, vup_embeddings
        torch.cuda.empty_cache()
        
    # --- Merge all layers ---
    combined_df = pd.concat(all_layer_dfs, ignore_index=True)
    combined_path = os.path.join(DATA_DIR, "all_layers_results.csv")
    combined_df.to_csv(combined_path, index=False)
    log.info("=" * 60)
    log.info("All layers done. Combined CSV saved: %s (%d rows)", combined_path, len(combined_df))
    log.info("=" * 60)


if __name__ == "__main__":
    main()