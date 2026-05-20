# llm-phrasal-compositionality

Probing the internal representations of language models to measure holistic storage of English V+*up* phrasal verbs (e.g. *pick up*, *set up*).

> **Paper**: *The holistic storage of verb+up phrases in text-based and audio-based language models* — Zachary Houghton, Yu Zhou, Dan Pluth, Vijay Gurbani

A logistic regression classifier is trained layer-by-layer on hidden states to distinguish standalone *up* tokens from random other tokens. The classifier's decision logit serves as a proxy for compositionality: high-frequency, idiomatic types (e.g. *end up*) should look less like standalone *up*, while low-frequency, transparent types should look more like it. We examine how this signal varies as a function of corpus frequency and predictability (P(up|V)) across layers in five models: OLMo-3 7B, three BabyLM OPT variants, and Whisper-small.

---

## Reproduce paper results

The classifier outputs and corpus statistics are already committed. No GPU required to regenerate figures.

```r
# 1. Fit statistical models (~2 hrs on 24 cores; cached on re-run)
source("Analyses/run_models_parallel.R")

# 2. Generate result CSVs from model cache
source("paper/prepare_results.R")

# 3. Render the paper
quarto::quarto_render("paper/writeup.qmd")
```

To rerun the full pipeline from scratch (GPU + infini-gram API access required), see [Full pipeline — OLMo-3 7B](#full-pipeline--olmo-3-7b) and following sections.

---

## Repository layout

```
├── Analyses/
│   ├── Python scripts         # dataset creation, classifiers, corpus stats
│   ├── analysis-script.Rmd   # interactive R analysis (brms + GAM)
│   ├── run_models_parallel.R  # parallel model fitting (run before paper/)
│   └── {olmo-3-7b,babylm,whisper}/run_pipeline.sh
├── Data/
│   ├── *.pkl                  # corpus statistics (committed)
│   ├── ftp_lookup.csv         # predictability values (committed)
│   ├── olmo-3-7b/             # classifier outputs: Data_up/, Data_upsubword/
│   ├── babylm/{opt-125m,opt-350m,opt-1.3b}/
│   └── whisper/{encoder,decoder}/
└── model_cache/               # fitted .rds model objects — gitignored
```

---

## Models

| Model | HuggingFace ID | Layers |
|---|---|---|
| OLMo-3 7B | `allenai/Olmo-3-1025-7B` | 32 |
| BabyLM OPT-125m | `znhoughton/opt-babylm-125m-64eps-seed964` | 12 |
| BabyLM OPT-350m | `znhoughton/opt-babylm-350m-64eps-seed964` | 24 |
| BabyLM OPT-1.3b | `znhoughton/opt-babylm-1.3b-64eps-seed964` | 24 |
| Whisper-small | `openai/whisper-small` | 12 enc + 12 dec |

---

## Corpus statistics and predictor definitions

Corpus statistics are matched to each model's training distribution:

| Model family | Frequency source | Predictability source |
|---|---|---|
| OLMo-3 7B | Dolma v1.7 (`olmo_corpus_stats.pkl`) | Dolma v1.7 (`ftp_lookup.csv`) |
| BabyLM OPT-* | BabyLM corpus (`babylm_corpus_stats.pkl`) | BabyLM corpus |
| Whisper-small | Dolma v1.7 (`olmo_corpus_stats.pkl`) | Dolma v1.7 (`ftp_lookup.csv`) |

**Frequency** is log(count(V+up)).

**Predictability** is P(up | V) = count(V+up) / count(V), entered as log-odds:

```
log-odds(predictability) = log( count(V+up) / (count(V) − count(V+up)) )
```

Only V+up types with a valid predictability value are included in the analysis, so the same item set is used for both predictors.

---

## Classifier design

**Layer-by-layer**: A separate logistic regression classifier is trained at each layer independently. The classifier at layer *L* is applied only to embeddings from that same layer.

**Negative examples**: Restricted to tokens whose decoded string is entirely alphabetic (`token.isalpha()`). For Whisper, the equivalent criterion is applied to WhisperX word segments.

**UP-as-subword positives** (`Data_upsubword`): Exactly one occurrence per unique up-within-word type (e.g., *setup*, *update*, *upon*), so the classifier generalises across types rather than memorising frequent forms.

---

## Full pipeline — OLMo-3 7B

### Step 1 — Build the C4 corpus dataset *(one-time)*

```bash
cd Analyses/
python create_dataset.py
# → Data/corpus_results.pkl
# → Data/corpus_results_upwords.pkl
```

### Step 2 — Compute corpus stats + predictability *(one-time)*

```bash
cd Analyses/
python get_olmo_corpus_stats.py   # → Data/olmo_corpus_stats.pkl
python export_predic_lookup.py    # → Data/ftp_lookup.csv
```

Queries Dolma v1.7 via the infini-gram API. Requires internet access.

### Steps 3–5 — Run classifiers *(GPU required)*

```bash
cd Analyses/olmo-3-7b/
bash run_pipeline.sh
```

Runs in sequence: (1) `create_train_val_test.py` — tokenize and write train/val/test CSVs; (2) `up_independently.py` — standalone-up classifier; (3) `subwords_containing_up.py` — up-subword classifier. Outputs go to `Data/olmo-3-7b/{Data_up,Data_upsubword}/`.

---

## Full pipeline — BabyLM OPT models

### Step 1 — Compute corpus stats *(one-time)*

```bash
cd Analyses/
python get_babylm_corpus_stats.py   # → Data/babylm_corpus_stats.pkl
```

Downloads `znhoughton/babylm-150m-v3` from HuggingFace (~hours).

### Steps 2–4 — Run classifiers *(GPU required)*

```bash
cd Analyses/babylm/
bash run_pipeline.sh              # all three models sequentially
bash run_pipeline.sh opt-125m     # or a single model
```

Outputs go to `Data/babylm/{opt-125m,opt-350m,opt-1.3b}/{Data_up,Data_upsubword}/`.

---

## Full pipeline — Whisper-small

### Step 1 — Build the audio dataset *(one-time)*

```bash
cd Analyses/whisper/
python build_audio_dataset.py
# → Data/whisper/dataset.csv
# → Data/whisper/audio/<sid>.wav  (gitignored)
```

Reads `Data/up-audio-metadata.csv`, extracts audio segments with ffmpeg, and runs WhisperX forced alignment.

### Step 2 — Run classifiers *(GPU required)*

```bash
cd Analyses/whisper/
bash run_pipeline.sh
# or: python run_whisper_classifier.py --corpus-stats-pkl ../../Data/olmo_corpus_stats.pkl
```

Extracts all 12 encoder + 12 decoder layers in a single forward pass per segment. Outputs go to `Data/whisper/{encoder,decoder}/`.

---

## Output files (per model, per classifier)

| File | Description |
|---|---|
| `train.csv`, `val.csv` | Training/validation sets |
| `test.csv` | Test set with `verb_up`, `frequency`, `predic` |
| `layer_XX.csv` | Classifier outputs at layer XX |
| `layer_XX_plot.png` | Compositionality scatter plot at layer XX |
| `all_layers_results.csv` | All layers concatenated — input to R |
| `layer_metadata.json` | Model info, layer count, sample sizes |

---

## Dataset statistics

### Train / validation splits (Experiment 1 — UP independently)

| Model | Train pos | Train neg | Val pos | Val neg |
|---|---|---|---|---|
| OLMo-3 7B | 1,000 | 1,000 | 992 | 1,000 |
| BabyLM 125M | 1,000 | 1,000 | 999 | 1,000 |
| BabyLM 350M | 1,000 | 1,000 | 999 | 1,000 |
| BabyLM 1.3B | 1,000 | 1,000 | 916 | 1,000 |

### Train / validation splits (Experiment 2 — UP as subword)

| Model | Train pos (standalone) | Train pos (subword, unique types) | Train neg | Val rows |
|---|---|---|---|---|
| OLMo-3 7B | 1,000 | 1,000 | 2,000 | 3,868 |
| BabyLM 125M | 1,000 | 1,000 | 2,000 | 3,582 |
| BabyLM 350M | 1,000 | 1,000 | 2,000 | 3,582 |
| BabyLM 1.3B | 1,000 | 1,000 | 2,000 | 3,773 |

### Test sets (Experiments 1 & 2)

| Model | Test sentences | Unique V+up types | Types w/ valid predic. |
|---|---|---|---|
| OLMo-3 7B | 81,586 | 4,082 | 3,927 |
| BabyLM 125M | 81,543 | 4,082 | 1,039 |
| BabyLM 350M | 81,543 | 4,082 | 1,039 |
| BabyLM 1.3B | 80,425 | 4,081 | 1,039 |

BabyLM has fewer types with valid predictability because the BabyLM corpus (~100M words) is much smaller than Dolma.

**Item-level statistics (unique V+up types with valid predictability):**

| Model | N items | Median freq | Freq range | Median predic. | Med logit(predic.) |
|---|---|---|---|---|---|
| OLMo-3 7B (Dolma v1.7) | 3,927 | 92 | 20–390,086 | 0.0046 | −5.38 |
| BabyLM (BabyLM corpus) | 1,039 | 920 | 20–390,086 | 0.0325 | −3.38 |

### Whisper audio dataset (Experiment 3)

| | Count |
|---|---|
| Segments processed | 224,118 |
| Classified as *word_up* (train/val) | 58,751 |
| Classified as *V+up* (test pool) | 165,367 |
| Unique V+up types | 4,161 |
| Qualifying test types (≥5 occurrences) | 1,466 |
| Train / val positives | 1,000 each |

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
# ffmpeg must be on PATH
```

R packages: `tidyverse`, `brms`, `mgcv`, `tidybayes`, `patchwork`, `viridis`, `plotly`, `jsonlite`, `future`, `furrr`