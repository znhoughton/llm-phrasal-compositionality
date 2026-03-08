# llm-phrasal-compositionality

Probing the internal representations of language models to measure the
compositionality of English V+up phrasal verbs (e.g. *pick up*, *set up*).

A logistic regression classifier is trained layer-by-layer on hidden states to
distinguish standalone "up" tokens from random other tokens. The classifier's
decision logit is used as a proxy for compositionality: high-frequency, idiomatic
types (e.g. *end up*) should look less like standalone "up", while low-frequency,
transparent types should look more like it. Analyses examine how this signal
varies as a function of corpus frequency and Forward Transitional Probability
(FTP = P(up|V)) across layers.

---

## Repository structure

```
llm-phrasal-compositionality/
├── Analyses/
│   ├── create_dataset.py              # Mine C4 for V+up sentences → .pkl files
│   ├── create_train_val_test.py       # Tokenize + build train/val/test CSVs
│   ├── up_independently.py            # Layer-by-layer classifier (standalone up)
│   ├── subwords_containing_up.py      # Layer-by-layer classifier (up morpheme)
│   ├── get_olmo_corpus_stats.py       # Compute verb frequencies + FTP from C4
│   ├── get_babylm_corpus_stats.py     # Compute verb frequencies + FTP from BabyLM corpus
│   ├── check_whisper_corpus.py        # Check speech corpus data sufficiency
│   ├── olmo-3-7b/
│   │   ├── run_pipeline.sh            # Run full OLMo-3 7B Python pipeline
│   │   └── analysis-script.Rmd       # R analysis for OLMo-3 7B results
│   ├── babylm/
│   │   ├── run_pipeline.sh            # Run full BabyLM Python pipeline
│   │   └── analysis-script.Rmd       # R analysis comparing all three BabyLM OPT models
│   └── whisper/
│       ├── build_whisper_dataset.py   # LibriSpeech → WhisperX alignment → dataset.csv
│       ├── run_whisper_classifier.py  # Layer-by-layer encoder+decoder classifier
│       ├── run_pipeline.sh            # Run full Whisper pipeline
│       └── analysis-script.Rmd       # R analysis: encoder vs decoder compositionality
├── Data/
│   ├── corpus_results.pkl             # V+up + standalone-up sentences (from C4)
│   ├── corpus_results_upwords.pkl     # up-within-word sentences (from C4)
│   ├── olmo_corpus_stats.pkl          # OLMo verb frequencies + FTP (from C4)
│   ├── babylm_corpus_stats.pkl        # BabyLM verb frequencies + FTP (from BabyLM corpus)
│   ├── olmo-3-7b/
│   │   ├── Data_up/                   # OLMo results: standalone-up classifier
│   │   └── Data_upsubword/            # OLMo results: up-morpheme classifier
│   ├── babylm/
│   │   ├── opt-125m/{Data_up,Data_upsubword}/
│   │   ├── opt-350m/{Data_up,Data_upsubword}/
│   │   └── opt-1.3b/{Data_up,Data_upsubword}/
│   └── whisper/
│       ├── dataset.csv                # utterance metadata + timestamps
│       ├── audio/<utt_id>.wav         # 16kHz utterance audio clips
│       ├── encoder/                   # layer_XX.csv, all_layers_results.csv, layer_metadata.json
│       └── decoder/                   # same structure as encoder/
└── README.md
```

---

## Models

| Model | HuggingFace ID | Layers |
|---|---|---|
| OLMo-3 7B | `allenai/Olmo-3-1025-7B` | 32 |
| BabyLM OPT-125m | `znhoughton/opt-babylm-125m-64eps-seed964` | 12 |
| BabyLM OPT-350m | `znhoughton/opt-babylm-350m-64eps-seed964` | 24 |
| BabyLM OPT-1.3b | `znhoughton/opt-babylm-1.3b-64eps-seed964` | 24 |

---

## Full pipeline — OLMo-3 7B

Run these steps in order from the repo root. Steps 1 and 2 are one-time setup;
steps 3–5 must run in sequence.

### Step 1 — Build the C4 corpus dataset *(one-time)*

Reads local C4 arrow files, produces shared sentence datasets used by all models.

```bash
cd Analyses/
python create_dataset.py
# → Data/corpus_results.pkl
# → Data/corpus_results_upwords.pkl
```

### Step 2 — Compute OLMo corpus stats + FTP *(one-time)*

Counts verb surface-form frequencies in C4 and computes Forward Transitional
Probability FTP = count(V+up) / count(V) for each V+up type.

```bash
cd Analyses/
python get_olmo_corpus_stats.py
# → Data/olmo_corpus_stats.pkl
```

> Requires the local C4 arrow files (default path: `../c4_10B_2B_local/train`).
> Pass `--data-dir` to override.

### Steps 3–5 — Run classifiers *(GPU required)*

```bash
cd Analyses/olmo-3-7b/
bash run_pipeline.sh
```

This runs three steps in sequence:
1. `create_train_val_test.py` — tokenizes sentences and writes `train.csv`,
   `val.csv`, `test.csv` (with `ftp` column) to `Data/olmo-3-7b/Data_up/` and
   `Data/olmo-3-7b/Data_upsubword/`
2. `up_independently.py` — layer-by-layer standalone-up classifier; writes
   `layer_XX.csv`, `layer_XX_plot.png`, `all_layers_results.csv` to `Data_up/`
3. `subwords_containing_up.py` — layer-by-layer up-morpheme classifier; same
   outputs to `Data_upsubword/`

### Step 6 — R analysis

Open `Analyses/olmo-3-7b/analysis-script.Rmd` in RStudio and knit (or run
chunk by chunk). Fitted model objects are cached as `.rds` files alongside the
Rmd so re-runs skip refitting.

The Rmd covers:
- Effect of **frequency** on logit at first/final layer (brms linear + bam non-linear)
- Effect of **FTP** (predictability) on logit at first/final layer
- **Joint** frequency + FTP effects
- All three above **across all layers** using `te()` tensor-product smooths
- 3D surface plots of predicted logit as a function of frequency × layer, FTP × layer, and frequency × FTP

---

## Full pipeline — BabyLM OPT models

### Step 1 — Compute BabyLM corpus stats + FTP *(one-time)*

Downloads `znhoughton/babylm-150m-v3` from HuggingFace (11.5M documents),
runs two passes: spaCy POS-tagging for V+up counts, then fast regex for verb
surface-form frequencies.

```bash
cd Analyses/
python get_babylm_corpus_stats.py
# → Data/babylm_corpus_stats.pkl
```

> This is the slowest step (~hours depending on hardware). The output is reused
> across all three BabyLM models.

### Steps 2–4 — Run classifiers *(GPU required)*

```bash
cd Analyses/babylm/
bash run_pipeline.sh          # runs all three models sequentially
bash run_pipeline.sh opt-125m # or run a single model
bash run_pipeline.sh opt-350m
bash run_pipeline.sh opt-1.3b
```

Each model runs the same three steps as the OLMo pipeline, writing results to
`Data/babylm/{opt-125m,opt-350m,opt-1.3b}/{Data_up,Data_upsubword}/`.

### Step 5 — R analysis

Open `Analyses/babylm/analysis-script.Rmd` in RStudio and knit. All three
models are loaded and compared in a single document. Layers are normalized to
[0, 1] to allow cross-model comparison (OPT-125m has 12 layers; 350m and 1.3b
have 24). The Rmd covers the same analyses as the OLMo Rmd, with `model` as an
additional factor throughout.

---

## Manual CLI reference

All scripts accept `--help`. The bash pipeline scripts call these with the
correct arguments automatically, but you can also run steps individually:

```bash
# create_train_val_test.py
python create_train_val_test.py \
  --model              znhoughton/opt-babylm-125m-64eps-seed964 \
  --data-dir-up        ../Data/babylm/opt-125m/Data_up \
  --data-dir-upsubword ../Data/babylm/opt-125m/Data_upsubword \
  --vup-pkl            ../Data/corpus_results.pkl \
  --upword-pkl         ../Data/corpus_results_upwords.pkl \
  --corpus-stats-pkl   ../Data/babylm_corpus_stats.pkl

# up_independently.py
python up_independently.py \
  --model    znhoughton/opt-babylm-125m-64eps-seed964 \
  --data-dir ../Data/babylm/opt-125m/Data_up \
  --vup-pkl  ../Data/corpus_results.pkl

# subwords_containing_up.py
python subwords_containing_up.py \
  --model    znhoughton/opt-babylm-125m-64eps-seed964 \
  --data-dir ../Data/babylm/opt-125m/Data_upsubword \
  --vup-pkl  ../Data/corpus_results.pkl
```

---

## Output files (per model, per classifier)

| File | Description |
|---|---|
| `train.csv`, `val.csv` | Training/validation sets (sentence, token_position, label) |
| `test.csv` | Test set with `verb_up`, `frequency`, `ftp`, `sentence`, `token_position` |
| `layer_XX.csv` | Per-sentence classifier outputs at layer XX (logit, probability, ftp) |
| `layer_XX_plot.png` | Scatter + bar plot of compositionality at layer XX |
| `all_layers_results.csv` | Concatenation of all `layer_XX.csv` files — input to R |
| `layer_metadata.json` | Model info, layer count, train/val sample sizes |

---

---

## Full pipeline — Whisper-small (speech)

Probes Whisper-small's **encoder** (audio) and **decoder** (text) representations
using LibriSpeech as the speech corpus. Requires WhisperX for forced word-level
alignment. Best run on a GPU machine.

### Step 1 — Build the LibriSpeech dataset *(one-time)*

Scans LibriSpeech for utterances containing "up", uses spaCy to classify each
"up" as V+up or standalone, and runs WhisperX forced alignment to get word-level
timestamps. Saves `dataset.csv` and per-utterance `.wav` files.

```bash
cd Analyses/whisper/
python build_whisper_dataset.py --split train.clean.100   # fast (~100h)
python build_whisper_dataset.py --split train.clean.360   # more V+up coverage
# → ../../Data/whisper/dataset.csv
# → ../../Data/whisper/audio/<utt_id>.wav
```

Prints a data-sufficiency summary at the end (target: ≥2000 standalone "up",
≥20 qualifying V+up types with ≥5 occurrences). Use `train.clean.360` if
coverage is insufficient.

### Step 2 — Layer-by-layer encoder + decoder classifier

Extracts hidden states from all 12 encoder and 12 decoder layers in a single
Whisper forward pass per utterance, trains a logistic regression classifier at
each layer, and evaluates on V+up test types.

```bash
python run_whisper_classifier.py   # uses ../../Data/whisper/ by default
# → ../../Data/whisper/encoder/layer_XX.csv, all_layers_results.csv, layer_metadata.json
# → ../../Data/whisper/decoder/layer_XX.csv, all_layers_results.csv, layer_metadata.json
```

Or run both steps with the pipeline script:

```bash
bash run_pipeline.sh                   # uses train.clean.100
bash run_pipeline.sh train.clean.360   # more coverage
```

### Step 3 — R analysis

Open `Analyses/whisper/analysis-script.Rmd` in RStudio and knit. Compares
encoder vs. decoder representations across all 12 layers. Covers:
- Effect of frequency on logit at first/final layer (brms + bam)
- Non-linear frequency effect across layers (`te(log_freq, layer, by=component)`)
- Validation accuracy per layer per component
- Spearman correlation between log(frequency) and mean logit across layers
- Most/least compositional V+up types at the final layer

---

## Dependencies

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib tqdm
pip install pyarrow datasets spacy
python -m spacy download en_core_web_sm
```

**Whisper pipeline only:**
```bash
pip install whisperx soundfile
```

R packages: `tidyverse`, `brms`, `mgcv`, `tidybayes`, `patchwork`, `viridis`, `plotly`
