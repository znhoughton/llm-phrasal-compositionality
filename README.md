# llm-phrasal-compositionality

Probing the internal representations of language models to measure holistic storage of English V+*up* phrasal verbs (e.g. *pick up*, *set up*).

> **Paper**: *The holistic storage of verb+up phrases in text-based and audio-based language models* — Zachary Houghton, Yu Zhou, Dan Pluth, Jordan Hosier, Vijay Gurbani

A logistic regression classifier is trained layer-by-layer on hidden states to distinguish standalone *up* tokens from random other tokens. The classifier's decision logit serves as a proxy for compositionality: high-frequency, idiomatic types (e.g. *end up*) should look less like standalone *up*, while low-frequency, transparent types should look more like it. We examine how this signal varies as a function of corpus frequency and predictability (P(up|V)) across layers in five models: OLMo-3 7B, three BabyLM OPT variants, and Whisper-small.

---

## Repository layout

```
├── Analyses/
│   ├── create_dataset.py            # scan C4 → candidate up/V+up + up-subword sentences (pickles)
│   ├── create_train_val_test.py     # tokenizer-specific train/val/test CSVs (per model)
│   ├── up_independently.py          # standalone-up classifier (text models)
│   ├── subwords_containing_up.py    # up-as-subword classifier (text models)
│   ├── get_*_corpus_stats.py, export_predic_lookup.py   # Dolma/BabyLM frequency + predictability
│   ├── analysis-script.Rmd          # interactive R analysis (brms + GAM)
│   ├── run_models_parallel.R        # parallel brms/GAM model fitting (cached to model_cache/, gitignored)
│   ├── {olmo-3-7b,babylm}/run_pipeline.sh
│   └── whisper/
│       ├── create_dataset.py                # scan GigaSpeech+Common Voice → candidate metadata CSVs
│       ├── build_audio_dataset.py           # extract audio + WhisperX align (standalone up/V+up)
│       ├── build_subword_audio_dataset.py   # same, for up-containing words (Experiment 2 replication)
│       ├── run_whisper_classifier.py
│       ├── validate_reconstruction.py       # QA: compare create_dataset.py's output to the real file
│       ├── run_pipeline.sh
│       └── unused/                          # superseded prototypes, kept for reference only
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

### Step 1 — Build candidate metadata *(one-time)*

```bash
cd Analyses/whisper/
python create_dataset.py
# → Data/up-audio-metadata.csv           (standalone up / V+up candidates)
# → Data/up-audio-metadata-subword.csv   (up-containing-word candidates, e.g. "update"/"upon" — Experiment 2 replication)
```

Scans GigaSpeech + Common Voice transcripts directly from a local corpus mount (not via the HuggingFace `datasets` library — see the script's docstring for why). `Data/up-audio-metadata.csv` is already committed and authoritative, so this **skips regenerating it** unless the file is missing or `--force-dataset1` is passed; `Data/up-audio-metadata-subword.csv` is new and has no existing reference file, but follows the same skip-if-exists rule (`--force-dataset2` to override). `validate_reconstruction.py` compares this script's Dataset-1 output against the real file if you do need to regenerate it (100% recall, ~98% precision at full corpus scale, in the validation run this was checked against).

### Step 2 — Extract audio + align *(one-time, GPU recommended for WhisperX)*

```bash
cd Analyses/whisper/
python build_audio_dataset.py
# → Data/whisper/dataset.csv
# → Data/whisper/audio/<sid>.wav  (gitignored)

# Experiment 2 replication (up-as-subword) -- only needed if extending to that experiment:
python build_subword_audio_dataset.py --metadata ../../Data/up-audio-metadata-subword.csv
# → Data/whisper/dataset_subword.csv
```

Extracts audio segments with ffmpeg and runs WhisperX forced alignment. `build_subword_audio_dataset.py` additionally uses character-level alignment to isolate just the "up" portion of a larger word's audio (e.g. the "up" in "update"), rather than pooling over the whole word — see its docstring for why word-level alignment alone isn't enough here.

### Step 3 — Run classifiers *(GPU required)*

```bash
cd Analyses/whisper/
bash run_pipeline.sh
# or directly:
python run_whisper_classifier.py --corpus-stats-pkl ../../Data/olmo_corpus_stats.pkl

# Experiment 2 replication -- combines subword positives with standalone-up
# for classifier training/validation only; the V+up test set is unchanged:
python run_whisper_classifier.py --corpus-stats-pkl ../../Data/olmo_corpus_stats.pkl \
    --subword-dataset ../../Data/whisper/dataset_subword.csv
    # → Data/whisper_subword/{encoder,decoder}/ -- NOT Data/whisper/, which holds
    #   the primary (non-subword) results the rest of the analysis depends on.
    #   --out-dir defaults to <data-dir>_subword automatically whenever
    #   --subword-dataset is set, specifically to avoid overwriting those.
```

Extracts all 12 encoder + 12 decoder layers in a single forward pass per segment. Outputs go to `Data/whisper/{encoder,decoder}/` for the primary run, `Data/whisper_subword/{encoder,decoder}/` for the Experiment 2 replication.

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

`run_whisper_classifier.py` additionally drops, from train/val/test alike: rows whose transcript contains "up" more than once (decoder token position can't be reliably resolved), V+up rows where a word intervenes between the verb and "up" (e.g. "picked it up", not an adjacent bigram), and rows whose sampled negative word repeats elsewhere in the transcript (same position-ambiguity risk, just for whichever word was drawn as the negative). Counts (indep condition): 36,419 ambiguous-position + 30,103 non-adjacent V+up + 1,429 ambiguous-negative rows dropped; final V+up test set: 1,511 types (21,860 rows). Subword condition additionally drops 2,049 ambiguous-position + 1,871 ambiguous-negative rows from the up-within-word class. `build_audio_dataset.py`'s `sample_negative()` also now restricts negative-word candidates to words that occur exactly once in the segment, so future re-extractions won't produce ambiguous negatives in the first place. See `drop_ambiguous_up_position_rows()` / `drop_nonadjacent_vup_rows()` / `drop_ambiguous_neg_word_rows()`.

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