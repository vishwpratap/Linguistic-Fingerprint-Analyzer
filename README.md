# Linguistic Fingerprint Analyzer

An academic NLP application for describing English writing patterns, comparing two text samples, and running a controlled authorship-classification experiment. Classification is exploratory and does not establish real-world identity or authorship.

## Capabilities

- Paste text or upload `.txt` files for two documents; load fictional demonstration texts or clear inputs.
- Overview of character, word, sentence, paragraph, average length, vocabulary, punctuation, and frequency features.
- Configurable function-word lists, with case-insensitive counts and rates per 1,000 tokens.
- Word unigram, bigram, and trigram counts and TF-IDF; configurable n-gram range and maximum vocabulary.
- Character n-gram counts and TF-IDF; configurable character range and vocabulary limit.
- Penn Treebank POS-tag distributions from NLTK's English averaged-perceptron tagger.
- Hapax and dislegomena counts/rates; TTR, root TTR (Guiraud), log TTR; word- and sentence-length distributions and sentence-length standard deviation.
- Pairwise TF-IDF similarities for word unigrams, word unigrams plus bigrams, configured word n-grams, and character n-grams.
- An experimental weighted combination of pair-normalized numerical stylometry and unigram TF-IDF, with visible formula and configurable weights.
- Plotly and Matplotlib visualizations, top feature tables, warnings for missing POS resources and short samples.
- Labeled CSV dataset validation, class counts/imbalance, duplicate inspection, and a synthetic demonstration corpus.
- Three-way train/validation/test experiments with deterministic stratification or group-disjoint document/source splits.
- Multinomial Naive Bayes, Logistic Regression, and Linear SVC across stylometric, word, character, and hybrid pipelines.
- Validation macro-F1 candidate selection, held-out test metrics, per-author metrics, confusion matrix, misclassified samples, feature weights, and a downloadable Markdown report.
- Held-out length and optional genre/topic robustness, random-versus-grouped evaluation, feature ablation, and preprocessing sensitivity checks on fixed seeds and split proportions.
- Fitted-model-space comparison for Documents A and B using the selected model's exact vocabulary, scaling, and normalization.
- Optional local model persistence. The app loads only app-generated opaque IDs, never arbitrary serialized uploads.

## Project structure

```text
app.py
requirements.txt
README.md
src/
  preprocessing.py       # shared Unicode tokens, normalization, sentences, paragraphs
  stylometry.py           # descriptive, function-word, n-gram, and POS features
  comparison.py           # feature deltas and pair-fitted TF-IDF representations
  dataset_manager.py      # schema validation, corpus QA, and leakage-aware splits
  feature_pipeline.py     # reusable training-only feature transformers
  models.py               # baseline classifiers
  training.py             # validation selection and held-out test evaluation
  explainability.py       # class-specific feature weights
  persistence.py          # constrained local artifact storage
  reporting.py            # Markdown experiment reports
  robustness.py           # length, split, ablation, sensitivity, and error checks
  visualizations.py       # reusable Plotly and Matplotlib figures
tests/
sample_data/
```

## Setup and run (Windows PowerShell)

Run these commands from the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Python 3.11 or a compatible version is recommended. The core tokenizer uses a Unicode-aware regular expression and works offline; it does not need NLTK corpora. Scikit-learn vectorizers are installed with the listed dependencies.

### Enable POS analysis (one-time local model data setup)

The app does not download language data automatically. Install the English averaged-perceptron tagger once:

```powershell
python -m nltk.downloader averaged_perceptron_tagger_eng
```

Restart or rerun the app after setup. If the model is unavailable, the POS tab explains the missing resource and displays no invented tags. Tests inject a deterministic tagger and do not require the model or network access.

## Run tests

```powershell
python -m pytest
```

## Running an authorship experiment

Open **Authorship Laboratory** and load the synthetic corpus or upload a CSV with at least `text` and `author` columns. Optional `document_id` and `source` columns support group-disjoint splitting. When passages come from the same original document, group by that document so its passages stay in one partition.

Review author sample counts, duplicate warnings, corpus provenance, and licence/permission details. Choose at least two authors, representations, classifiers, split proportions, and a random seed. Vectorizers and numeric scaling are fit within each pipeline on training data only. Candidates are ranked by validation macro-F1; the selected candidate alone is scored on the held-out test partition.

POS features are optional and require the one-time NLTK setup documented above. When the local resource is missing, training stops with instructions rather than inventing tags. Synthetic results exercise the workflow but do not support research conclusions.

Saved models and JSON metadata are written under `.artifacts/authorship_models/`, excluded from Git. Joblib is a Python serialization format: only use app-generated files from this local store. Metadata records configuration, provenance, labels, seed, metrics, and a SHA-256 fingerprint of training text and labels; raw text is not copied into the manifest.

The **Robustness and generalization experiments** panel evaluates the selected pipeline on held-out passages at configurable minimum word lengths and short/medium/long groups, with sample counts and low-count warnings. Optional `genre` and `topic` columns provide per-domain held-out metrics. The panel can compare stratified passage splits with document/source-grouped splits, compare stylometric, word, character, and hybrid ablations, and test case preservation, word n-gram, character n-gram, and stop-word variants using the same seed and split policy. Grouped performance decreases can be consistent with source leakage or distribution shift; the score difference alone does not establish its cause. Ablation and sensitivity test scores are descriptive and should not be used to tune against the test set.

The downloadable report includes dataset provenance, split design, configuration, validation selection, test metrics, per-author scores, common confusions, and completed robustness tables. Raw error excerpts are excluded by default; including them is an explicit UI choice. The local error-analysis panel exposes correctly classified and misclassified held-out samples, and Documents A/B can be inspected with feature deltas in the selected model's fitted feature space.

## Reproducing comparisons

Enter or upload both documents, then set the word/character n-gram ranges and vocabulary caps in their tabs. The Document Comparison tab exposes text and numerical-style weights. The combined experimental score uses:

```text
cosine([text_weight × L2-normalized word-unigram TF-IDF,
        style_weight × L2-normalized numerical style vector])
```

Numerical style dimensions are scaled by the pairwise maximum absolute value. Every vectorizer in a two-document comparison is fitted only on those two documents. Its similarities, top terms, and weights describe that pair; they are not generalizable model outputs. The Authorship Laboratory fits text vocabularies and numeric scaling on training partitions only.

## Definitions and limitations

- Lexical word tokens are NFC-normalized, case-folded Unicode letter/number sequences with internal apostrophes; hyphenated forms split. POS analysis uses a parallel NFC-normalized token stream that retains source capitalization. The source text itself is retained for punctuation and other original-text features.
- Function-word categories contain a small documented, configurable selection of articles, pronouns, prepositions, and conjunctions. They are not a complete grammar inventory or fingerprint. Their rates use all tokens as the denominator and are reported per 1,000 tokens.
- Word n-grams use the same tokenization as word counts. Stop-word removal is off by default. When enabled, removed tokens can cause previously separated words to become adjacent in phrase n-grams.
- Character n-grams are built from NFC text after collapsing whitespace to one space and case-folding. Spaces and punctuation remain, so formatting and spelling affect the output.
- POS tags use NLTK's English averaged-perceptron tagger and Penn Treebank labels; frequencies divide each tag count by total tagged tokens. Short passages and tagger errors make distributions uncertain.
- Hapax legomena occur once and dislegomena twice. Their rates are per 1,000 tokens. Word-length frequencies are also per 1,000 tokens. Sentence lengths are token counts per detected sentence; standard deviation is population standard deviation and is zero for fewer than two sentences.
- TTR is unique types divided by tokens and is strongly sensitive to text length. Root TTR (Guiraud) divides types by the square root of tokens. Log TTR is log(types) divided by log(tokens), defined here as zero below two tokens. None fully removes length effects.
- Sentence segmentation uses terminal punctuation as a lightweight heuristic; abbreviations and decimals may be over-segmented. Empty texts and texts without usable features return empty distributions or explicit warnings rather than fabricated values.
- Local cosine similarities measure overlap in the selected word, phrase, character, or numerical features. The combined weighted score is experimental, depends on the supplied weights, and is not a probability of common authorship. Character patterns do not uniquely identify an author.
- Topic, genre, text length, editing, quoted material, spelling, and formatting can affect the measurements. Short samples are flagged and require cautious interpretation.
- Group-aware splits need at least three distinct groups per author and may fail if every author cannot appear in all partitions without group overlap. Random passage splits can overstate performance when related passages exist.
- Closed-set predictions are forced among selected authors. They have no reject option for unknown authors and are not calibrated identity probabilities.

## Extension path

Reusable processing, training, and reporting functions remain in `src/`, separate from the UI. The Authorship Laboratory is an educational research workflow, not forensic evidence or a claim that an author has been identified.
