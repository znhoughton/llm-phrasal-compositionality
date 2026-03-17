# llm-phrasal-compositionality

Probing the internal representations of language models to measure the
compositionality of English V+up phrasal verbs (e.g. *pick up*, *set up*).

A logistic regression classifier is trained layer-by-layer on hidden states to
distinguish standalone "up" tokens from random other tokens. The classifier's
decision logit is used as a proxy for compositionality: high-frequency, idiomatic
types (e.g. *end up*) should look less like standalone "up", while low-frequency,
transparent types should look more like it. Analyses examine how this signal
varies as a function of corpus frequency and Forward Transitional Probability
(FTP = P(up|V)) across layers. The predictability predictor used in all models
is the **log-odds of FTP**: log(FTP / (1 − FTP)) = log(count(V+up) /
(count(V) − count(V+up))).

Corpus statistics are matched to each model's training distribution: OLMo and
Whisper use C4; BabyLM models use the BabyLM training corpus.

**Classifier design**: A separate logistic regression classifier is trained at
each layer independently. The classifier trained at layer *L* is then applied to
V+up test embeddings extracted from that same layer *L*. No classifier trained on
one layer is ever applied to another layer's embeddings.

**Negative examples**: For all three experiments, negative tokens are restricted
to those whose decoded string consists entirely of alphabetic characters
(`token.isalpha()` is true), excluding punctuation, numbers, and mixed
alphanumeric tokens. This does not exclude subword continuations (e.g., the
`"ing"` fragment of `"housing"` would pass), but removes clearly non-lexical
tokens from the negative set. For the Whisper experiment the equivalent
criterion is applied to WhisperX word segments: only words whose full string
is alphabetic (`word.isalpha()`) are eligible as negatives.

**UP-as-subword positives**: The `Data_upsubword` classifier uses exactly one
occurrence per unique up-within-word type as its positive subword examples
(e.g., "setup" appears at most once in the training set, as does "update",
"upon", etc.). This ensures that the classifier generalises across word types
rather than memorising high-frequency forms.

---

## Repository structure

```
llm-phrasal-compositionality/
├── Analyses/
│   ├── create_dataset.py              # Mine C4 for V+up sentences → .pkl files
│   ├── create_train_val_test.py       # Tokenize + build train/val/test CSVs
│   ├── up_independently.py            # Layer-by-layer classifier (standalone up)
│   ├── subwords_containing_up.py      # Layer-by-layer classifier (up subword)
│   ├── get_olmo_corpus_stats.py       # Compute verb frequencies + FTP from C4
│   ├── get_babylm_corpus_stats.py     # Compute verb frequencies + FTP from BabyLM corpus
│   ├── check_whisper_corpus.py        # Check speech corpus data sufficiency
│   ├── analysis-script.Rmd            # Unified R analysis (params: source = "olmo"|"babylm"|"whisper")
│   ├── olmo-3-7b/
│   │   └── run_pipeline.sh            # Run full OLMo-3 7B Python pipeline
│   ├── babylm/
│   │   └── run_pipeline.sh            # Run full BabyLM Python pipeline
│   └── whisper/
│       ├── build_audio_dataset.py     # Audio metadata CSV → WhisperX alignment → dataset.csv
│       ├── run_whisper_classifier.py  # Layer-by-layer encoder+decoder classifier
│       └── run_pipeline.sh            # Run full Whisper pipeline
├── Data/
│   ├── corpus_results.pkl             # V+up + standalone-up sentences (from C4)
│   ├── corpus_results_upwords.pkl     # up-within-word sentences (from C4)
│   ├── olmo_corpus_stats.pkl          # OLMo verb frequencies + FTP (from C4)
│   ├── babylm_corpus_stats.pkl        # BabyLM verb frequencies + FTP (from BabyLM corpus)
│   ├── ftp_lookup.csv                 # FTP values for all V+up types (from C4; used by OLMo + Whisper R analysis)
│   ├── up-audio-metadata.csv          # Audio segment metadata (mixed sources; word+up occurrences)
│   ├── olmo-3-7b/
│   │   ├── Data_up/                   # OLMo results: standalone-up classifier
│   │   └── Data_upsubword/            # OLMo results: up-subword classifier
│   ├── babylm/
│   │   ├── opt-125m/{Data_up,Data_upsubword}/
│   │   ├── opt-350m/{Data_up,Data_upsubword}/
│   │   └── opt-1.3b/{Data_up,Data_upsubword}/
│   └── whisper_audio/
│       ├── dataset.csv                # segment metadata + WhisperX timestamps
│       ├── audio/<sid>.wav            # 16kHz extracted segment clips
│       ├── encoder/                   # layer_XX.csv, all_layers_results.csv, layer_metadata.json
│       └── decoder/                   # same structure as encoder/
├── model_cache/                       # Cached brms/bam .rds files — gitignored
│   ├── olmo/
│   ├── babylm/
│   └── whisper/
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
| Whisper-small | `openai/whisper-small` | 12 enc + 12 dec |

---

## Corpus statistics and predictor definitions

Each model's frequency and FTP values come from the corpus that best reflects
its training distribution:

| Model family | Frequency source | FTP source |
|---|---|---|
| OLMo-3 7B | C4 (`olmo_corpus_stats.pkl`) | C4 (`ftp_lookup.csv`) |
| BabyLM OPT-* | BabyLM corpus (`babylm_corpus_stats.pkl`) | BabyLM corpus |
| Whisper-small | C4 (`corpus_results.pkl`) | C4 (`ftp_lookup.csv`) |

**Frequency** is the raw count of each V+up type in the relevant corpus,
log-transformed before use: log(count(V+up)).

**Predictability** is the Forward Transitional Probability FTP = P(up | V) =
count(V+up) / count(V), entered as its log-odds:

```
log-odds(FTP) = log( count(V+up) / (count(V) − count(V+up)) )
              = log( FTP / (1 − FTP) )
```

This measures the odds that "up" follows verb V versus all other continuations
of V. Only V+up types with a valid (non-null) FTP value are included in the
analysis, ensuring that the same item set is used for both the frequency and
predictability models.

The `ftp_lookup.csv` file is generated from `olmo_corpus_stats.pkl` and shared
between the OLMo and Whisper R analyses. BabyLM reads FTP directly from its
own result CSVs (populated during pipeline step 1).

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
3. `subwords_containing_up.py` — layer-by-layer up-subword classifier; same
   outputs to `Data_upsubword/`

At each layer, a fresh logistic regression classifier is trained on train-set
hidden states from that layer, validated on val-set hidden states from that
layer, and then evaluated on V+up test-set hidden states from that same layer.

### Step 6 — R analysis

```r
rmarkdown::render("Analyses/analysis-script.Rmd", params = list(source = "olmo"))
```

Or open `Analyses/analysis-script.Rmd` in RStudio, set `params: source: "olmo"`
in the YAML header, and knit. Fitted model objects are cached in `model_cache/olmo/`
(gitignored) so re-runs skip refitting.

The analysis covers:
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

```r
rmarkdown::render("Analyses/analysis-script.Rmd", params = list(source = "babylm"))
```

All three models are loaded and compared in a single document. Layers are
normalized to [0, 1] to allow cross-model comparison (OPT-125m has 12 layers;
350m and 1.3b have 24). Fitted models are cached in `model_cache/babylm/`.

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

## Full pipeline — Whisper-small (speech)

Probes Whisper-small's **encoder** (audio) and **decoder** (text) representations
using a mixed-source audio metadata file (`Data/up-audio-metadata.csv`). Requires
WhisperX for forced word-level alignment. Best run on a GPU machine.

### Embedding extraction

Hidden-state embeddings for "up" are extracted as follows:

- **Encoder**: Whisper's encoder operates on 80-dim log-mel spectrograms with
  a CNN frontend that outputs one frame every 20 ms. For each segment, all
  encoder and decoder hidden states are obtained in a **single forward pass**.
  The "up" embedding at a given encoder layer is the **mean-pool of hidden-state
  vectors over the frames spanning the word's time window**, as determined by
  WhisperX forced word-level alignment (i.e. all 20 ms frames from `up_start`
  to `up_end` are averaged).
- **Decoder**: The gold transcript is fed to the decoder as teacher-forced input.
  The "up" embedding at a given decoder layer is the **hidden state at the exact
  token position** of "up" in the tokenized transcript.

### Classifier design

A separate logistic regression classifier is trained at each layer independently,
mirroring the design of Experiments 1 and 2 (OLMo/BabyLM):

- **Train/val** (1,000 instances each): *word_up* embeddings (positive) — audio
  segments where *up* is preceded by a non-verb word (e.g., *it up*, *them up*),
  classified by spaCy POS-tagging on the full segment transcript. This mirrors
  the standalone-up positive used in the text-based experiments. Negative examples
  are randomly selected non-*up* words from the same segment, restricted to
  purely alphabetic words (`word.isalpha()`), matching the `token.isalpha()`
  criterion used in Experiments 1 and 2, with a minimum duration of 20 ms.
- **Test**: V+up phrasal verb embeddings from qualifying types (≥5 occurrences
  in the audio dataset; a lower threshold than Experiments 1 and 2, reflecting
  sparser audio coverage). The `frequency` column in output CSVs uses C4 corpus
  counts (passed via `--vup-pkl`) consistent with the text-based experiments.

For **efficiency**, embeddings at all layers are extracted in a **single Whisper
forward pass per segment** (not one pass per layer). The per-layer training
loop then slices the pre-computed embeddings and fits a fresh classifier at each
layer.

### Step 1 — Build the audio dataset *(one-time)*

Reads `up-audio-metadata.csv`, extracts timestamped audio segments using ffmpeg,
runs WhisperX forced alignment to get word-level timestamps for "up", and
classifies each "up" occurrence as *word_up* or *vup* using spaCy POS-tagging
on the full transcript.

```bash
cd Analyses/whisper/
python build_audio_dataset.py
# → ../../Data/whisper_audio/dataset.csv
# → ../../Data/whisper_audio/audio/<sid>.wav
```

Prints a data-sufficiency summary at the end (target: ≥2,000 *word_up*
occurrences and ≥20 qualifying V+up types with ≥5 occurrences each).

### Step 2 — Layer-by-layer encoder + decoder classifier

Extracts hidden states from all 12 encoder and 12 decoder layers in a single
Whisper forward pass per segment, trains a logistic regression classifier at
each layer independently, and evaluates on V+up test types.

```bash
cd Analyses/whisper/
bash run_pipeline.sh
```

Or run the classifier directly:

```bash
python run_whisper_classifier.py \
  --vup-pkl ../../Data/corpus_results.pkl   # uses C4 frequencies for the frequency column
# → ../../Data/whisper_audio/encoder/layer_XX.csv, all_layers_results.csv, layer_metadata.json
# → ../../Data/whisper_audio/decoder/layer_XX.csv, all_layers_results.csv, layer_metadata.json
```

> Requires `Data/corpus_results.pkl` (produced by `create_dataset.py` in the OLMo pipeline).

### Step 3 — R analysis

```r
rmarkdown::render("Analyses/analysis-script.Rmd", params = list(source = "whisper"))
```

Compares encoder vs. decoder representations across all 12 layers. Fitted models
are cached in `model_cache/whisper/`. Covers:
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
# ffmpeg must also be available on PATH (for audio segment extraction)
```

R packages: `tidyverse`, `brms`, `mgcv`, `tidybayes`, `patchwork`, `viridis`, `plotly`, `jsonlite`
