"""Streamlit interface for the Linguistic Fingerprint Analyzer."""

from __future__ import annotations

import pandas as pd
import random
import streamlit as st

from src.comparison import FEATURE_LABELS, build_document_representations, compare_features, compare_texts_with_fitted_pipeline, lexical_similarity
from src.dataset_manager import (
    DatasetValidationError, eligible_authors, filter_authors, find_near_duplicates,
    synthetic_demo_dataset, validate_dataset,
)
from src.explainability import top_model_features
from src.feature_pipeline import REPRESENTATIONS
from src.models import MODEL_LABELS
from src.persistence import list_saved_models, load_saved_model, save_model
from src.reporting import experiment_report
from src.robustness import (
    compare_random_and_group_splits,
    evaluate_document_length_robustness,
    evaluate_domain_robustness,
    run_feature_ablation,
    run_preprocessing_sensitivity,
)
from src.stylometry import (
    FUNCTION_WORDS,
    PENN_TAG_DESCRIPTIONS,
    STOP_WORDS,
    character_ngram_counts,
    extract_features,
    function_word_frequencies,
    pos_tag_distribution,
    word_ngram_counts,
)
from src.visualizations import (
    document_style_map,
    feature_comparison_chart,
    normalized_frequency_chart,
    punctuation_comparison_chart,
    style_figure,
    word_frequency_chart,
)
from src.training import run_experiments

st.set_page_config(page_title="Linguistic Fingerprint Analyzer", page_icon="◈", layout="wide")

DEMO_A = """The archive opened before sunrise, when the reading room was still cool and the windows held a thin veil of mist. Mara began with the eastern cabinet, as she had done every Monday for nearly a year. She lifted each folder with both hands, checked its number against the ledger, and placed it beneath the green-shaded lamp. The paper varied considerably: some sheets were thick and cream-colored, while others had grown brittle at the edges. In several bundles, a faint line of dust marked the place where a ribbon had once been tied.

She recorded these details in a narrow notebook. Her entries followed the same order: date, material, condition, handwriting, and any mark that might identify the sender. When a letter contained several corrections, she described each one separately. A small word crossed out in blue pencil was different from a sentence rewritten in brown ink; the distinction could reveal when a page had been reviewed. She did not hurry. Instead, she compared the envelope with the letter, then checked both against the catalogue prepared by the previous archivist.

By midmorning, the first cabinet had yielded three useful observations. Two letters shared an unusual watermark, although their dates were twelve years apart. Another carried a postmark from a town absent from the collection's index. Mara entered a question beside that item and left the answer blank. An empty space, she believed, was more accurate than a confident guess.

At noon, the room filled with the quiet sounds of work: a page turning, a pencil resting on the desk, the distant click of a door. Mara paused to compare her notes with the ledger. A few entries required revision, and one date remained uncertain because the final digit had faded. She marked the uncertainty in the margin rather than smoothing it away. Then she returned the letters to their folders, aligned the labels, and closed the cabinet.

The afternoon brought a different set of papers. These had been kept in a wooden box and were arranged by family rather than by year. Mara changed her method accordingly. She made a temporary table, grouped repeated names together, and noted where a signature changed shape. Some patterns emerged at once; others appeared only after several pages had been placed side by side. At the end of the day, she reviewed every new entry, counted the folders she had handled, and wrote a brief summary for the next researcher. The archive seemed no closer to being complete, yet its records were more carefully described than they had been that morning."""
DEMO_B = """I got to the archive at dawn, and honestly, I thought I'd be done with the first cabinet by lunch. That was optimistic. The folders looked tidy from the doorway, but once I started opening them, every page had a little surprise: a different paper, a missing date, a mark in the corner, or a ribbon that had left a dusty stripe behind. I grabbed my notebook and started writing everything down before I forgot what I'd just seen.

My system was simple. I wrote the date first, then the paper, then whatever looked odd. It wasn't elegant, but it worked. One envelope had a stamp from a town I'd never heard of. Another letter had been corrected in blue pencil, while the next one had a whole sentence rewritten in brown ink. Are those details important? Maybe. I didn't know yet, so I put a question mark beside them instead of pretending I had the answer.

By midmorning, I had found two letters with the same strange watermark. The dates were twelve years apart! I checked them again, then checked the catalogue, because I'd rather be sure than announce a pattern too early. The catalogue didn't mention the watermark at all. That's useful, I thought, though it could also mean somebody simply missed it.

At noon I stopped for a minute. The room was quiet, except for pages turning and a door clicking somewhere down the hall. I compared my notes with the ledger and caught a couple of mistakes. One date was hard to read; the last digit had nearly vanished. I left it uncertain. Why guess when the paper won't tell me?

After lunch, the wooden box changed the plan. Those letters were sorted by family, not by year, so I made a quick table and grouped the names. Then I noticed one signature didn't quite match the others. Was it the same person writing in a hurry, or someone else? I couldn't settle it from one page. I spread several letters across the desk, looked from one to the next, and wrote down what seemed different.

The afternoon went faster than I expected, but I still didn't finish the box. Before leaving, I counted the folders, checked the labels, and wrote a short note for whoever comes in tomorrow. The archive isn't solved, and that's fine. I found a few leads, fixed my notes, and left the unanswered questions where they belong: right there on the page."""
DEMO_DOCUMENTS = (
    ("Measured archive report", DEMO_A),
    ("Conversational archive diary", DEMO_B),
    ("Field notes: coastal survey", """At 06:10 the team left the harbor office and walked north along the breakwater. The tide was falling, and a narrow strip of wet sand had appeared below the sea wall. I recorded the wind as moderate from the west. The handheld meter showed 17.8 degrees Celsius at the first station; its reading settled only after we moved it out of direct sunlight.

Station two was placed beside the old signal post. We measured the water twice, then repeated the sample because the first bottle contained suspended grit. The second reading was clearer. A line of small shells marked the upper edge of the recent tide, though several had been displaced by footprints. I photographed the marker and noted its distance from the concrete stair.

By midmorning, low cloud had covered the headland. Visibility fell, but the landmark remained visible from the southern cove. The team shortened the interval between measurements and kept the equipment in sealed cases. These changes were procedural; they were not evidence of a change in water quality.

At the final station, the current ran faster than it had at dawn. We checked the timing against the harbor clock and repeated the measurement. The two results were close enough to retain, with a note about the passing launch. Before returning, we inspected the labels, matched each bottle to its field sheet, and stored the samples upright. The observations describe this morning's route and conditions only. A longer survey would be needed before drawing conclusions about seasonal patterns."""),
    ("A note on the old cinema", """The cinema on Mercer Street was never especially grand. Its sign had three missing bulbs, the carpet smelled faintly of rain, and the seats made a soft complaint whenever someone leaned back. Still, on Friday evenings the place seemed to gather the whole neighborhood under one roof. You could hear people arguing about the best entrance before the doors had even opened.

I went there last week to see whether the lobby was as small as I remembered. It was. The ticket window had been replaced by a tablet, and the poster cases were empty, but the brass rail was still polished by thousands of hands. An usher pointed toward the balcony and said the projector had been repaired again. “Again” did a lot of work in that sentence.

The film began late. Nobody minded at first. A child dropped a packet of sweets; someone whispered a plot prediction that turned out to be wrong; then the lights dimmed and the room settled. For two hours, the old building felt less like a relic than a meeting place.

When I stepped outside, the street seemed brighter than it had on the way in. Perhaps that was the marquee, though only half its bulbs worked. I walked home thinking that a place does not have to be perfect to remain useful. Sometimes it only has to open its doors, keep the sound working, and leave enough room for people to return."""),
    ("Workshop incident memo", """Incident: pressure loss in the east-wing test loop. Date: 14 May. Reported by: facilities technician. The system alarm activated at 09:42 during a scheduled inspection. No personnel were injured, and no material was released outside the contained service area.

The operator isolated the loop at valve E-4 and notified the duty supervisor. Pressure fell from 2.1 bar to 0.4 bar over approximately six minutes. The local gauge and the control-room display showed similar values. A visual inspection identified moisture near the lower coupling; the source was not confirmed during the initial response.

The area was cordoned off while the line cooled. Staff checked the adjacent fittings and found no visible damage. The maintenance log showed that the coupling had been replaced in February, but the installation record did not include a post-repair torque value. This omission is documented as a record gap, not as a confirmed cause.

At 11:15, the loop remained isolated pending a pressure test and review by engineering. The equipment must not be returned to service until the test is complete and the supervisor has signed the release checklist. Follow-up actions are to verify the coupling specification, record the test method and readings, and update the maintenance form. This memo summarizes the information available at the end of the inspection; the investigation remains open."""),
    ("Garden journal", """The first warm Saturday arrived without much ceremony. I opened the shed, found the gloves beneath a stack of seed trays, and stood for a minute deciding where to begin. The back bed was crowded with last year's stems. The pots near the steps were dry. A robin watched from the fence as if it had already reviewed my plan and found it incomplete.

I started with the pots. The mint had survived the winter, naturally, while the thyme looked less certain. I trimmed the dead tips and loosened the soil with a wooden label. Under the surface it was cooler than I expected. After watering, I left the containers in the shade rather than moving them into the full afternoon sun.

The back bed took longer. I pulled away the brittle stems one handful at a time and found a patch of green beneath them. There were small shoots along the edge, not many, but enough to make me stop and look twice. I nearly removed them with the weeds. Next time I should mark the bulbs before autumn, though I say that every year.

By lunch the path was clear and the tools were lined up beside the tap. The garden was not transformed. A few pots still need attention, and the bed could use another hour. But the light had changed, the robin had moved to the roof, and the air smelled like wet soil. I put the labels back in the tin and decided that was a good place to stop."""),
)
OVERVIEW_KEYS = [
    "character_count", "word_count", "sentence_count", "average_sentence_length",
    "average_word_length", "unique_word_count", "type_token_ratio", "paragraph_count",
]


@st.cache_data(show_spinner=False)
def cached_pos_tags(text: str) -> dict:
    """Cache tagger output by the exact document text."""
    return pos_tag_distribution(text)


@st.cache_data(show_spinner=False)
def cached_representations(
    text_a: str,
    text_b: str,
    ngram_range: tuple[int, int],
    char_ngram_range: tuple[int, int],
    max_vocabulary: int,
    remove_stop_words: bool,
    text_weight: float,
    style_weight: float,
) -> dict:
    """Cache pair-local vectorizations with all configuration in the cache key."""
    return build_document_representations(
        text_a,
        text_b,
        ngram_range=ngram_range,
        char_ngram_range=char_ngram_range,
        max_vocabulary=max_vocabulary,
        remove_stop_words=remove_stop_words,
        text_weight=text_weight,
        style_weight=style_weight,
    )


def comparison_rows(features_a: dict, features_b: dict, keys: list[str]) -> pd.DataFrame:
    """Format requested top-level feature values for a side-by-side table."""
    rows = []
    comparison = compare_features(features_a, features_b)
    for key in keys:
        rows.append({
            "Feature": FEATURE_LABELS[key],
            "Document A": features_a.get(key, 0),
            "Document B": features_b.get(key, 0),
            "Absolute difference": comparison[key]["absolute_difference"],
            "Relative difference vs A (%)": comparison[key]["relative_difference_percent"],
        })
    return pd.DataFrame(rows)


def ngram_table(grams_a: dict, grams_b: dict, *, kind: str) -> pd.DataFrame:
    """Merge two n-gram count dictionaries into a readable comparison table."""
    rows = []
    for n in sorted(set(grams_a) | set(grams_b)):
        counts_a = dict(grams_a.get(n, []))
        counts_b = dict(grams_b.get(n, []))
        for gram in sorted(set(counts_a) | set(counts_b), key=lambda value: (-(counts_a.get(value, 0) + counts_b.get(value, 0)), value)):
            label = " ".join(gram) if kind == "word" else gram
            count_a, count_b = counts_a.get(gram, 0), counts_b.get(gram, 0)
            rows.append({
                "N": n, "N-gram": label,
                "Document A count": count_a, "Document B count": count_b,
                "Absolute count difference": abs(count_b - count_a),
                "Relative difference vs A (%)": (count_b - count_a) / abs(count_a) * 100 if count_a else None,
            })
    return pd.DataFrame(rows)


st.markdown(
    """<style>
    /* One fixed, theme-neutral palette: muted slate surfaces with teal accents. */
    :root { color-scheme: light; }
    .stApp { background: #e9eeec; color: #34464b; }
    .stApp, .stApp p, .stApp label, .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stWidgetLabel"] { color: #34464b; }
    [data-testid="stHeader"] { background: rgba(233, 238, 236, .96); }
    [data-testid="stSidebar"] { background: #e1e8e5; border-right: 1px solid #cbd6d2; }
    .block-container { max-width: 1420px; padding-top: 1.6rem; padding-bottom: 3rem; }
    .hero { padding: 1.35rem 1.65rem; border: 1px solid #cbd6d2; border-radius: 18px;
        background: linear-gradient(120deg, #f1f4f2, #e5ece9); margin-bottom: 1.15rem;
        box-shadow: 0 8px 22px rgba(44, 68, 66, .08); }
    .hero h1 { color: #2f5358; letter-spacing: -.025em; margin-bottom: .25rem; }
    .hero p { color: #52666a; margin-bottom: 0; }
    .stApp h2, .stApp h3 { color: #315c61; letter-spacing: -.012em; }
    .stButton > button, .stDownloadButton > button { border-radius: 10px; border: 1px solid #527e80;
        background: #47777a; color: #fff; font-weight: 600; transition: background .15s ease, transform .15s ease; }
    .stButton > button:hover, .stDownloadButton > button:hover { background: #365f64; color: #fff; border-color: #365f64; transform: translateY(-1px); }
    .stButton > button:focus, .stDownloadButton > button:focus { box-shadow: 0 0 0 .2rem rgba(71, 119, 122, .24); }
    [data-baseweb="tab-list"] { gap: .35rem; border-bottom: 1px solid #cad5d1; }
    [data-baseweb="tab"] { color: #52666a; background: #e1e8e5; border-radius: 10px 10px 0 0; padding: .55rem .9rem; }
    [data-baseweb="tab"][aria-selected="true"] { color: #285b60; background: #f5f7f5; border-bottom: 3px solid #47777a; }
    [data-testid="stExpander"] { background: #f1f4f2; border: 1px solid #d0d9d5; border-radius: 12px; }
    [data-testid="stMetric"] { background: #f2f5f3; border: 1px solid #d0d9d5; border-radius: 12px; padding: .8rem 1rem; }
    input, textarea, [data-baseweb="select"] > div { background: #f8faf9 !important; color: #34464b !important; border-color: #becbc6 !important; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid #d0d9d5; border-radius: 10px; overflow: hidden; }
    [data-testid="stAlert"] { border-radius: 10px; }
    @media (max-width: 800px) { .block-container { padding: 1rem .8rem 2rem; } .hero { padding: 1rem 1.1rem; } }
    </style>""",
    unsafe_allow_html=True,
)
st.markdown('<div class="hero"><h1>Linguistic Fingerprint Analyzer</h1><p>An NLP-based writing style analysis and document comparison laboratory.</p></div>', unsafe_allow_html=True)
st.caption("Phase 4 · Robustness, explainability, and controlled authorship experiments · Results are exploratory")

st.subheader("Input workspace")
left_upload, right_upload = st.columns(2)
with left_upload:
    upload_a = st.file_uploader("Upload Document A (.txt)", type=["txt"], key="upload_a")
with right_upload:
    upload_b = st.file_uploader("Upload Document B (.txt)", type=["txt"], key="upload_b")

button_a, button_b, button_demo, button_clear = st.columns([1, 1, 1.2, 1])
with button_a:
    if st.button("Use uploaded file for A", disabled=upload_a is None, width="stretch"):
        st.session_state["document_a"] = upload_a.getvalue().decode("utf-8", errors="replace")
        st.session_state.pop("demo_pair_names", None)
with button_b:
    if st.button("Use uploaded file for B", disabled=upload_b is None, width="stretch"):
        st.session_state["document_b"] = upload_b.getvalue().decode("utf-8", errors="replace")
        st.session_state.pop("demo_pair_names", None)
with button_demo:
    if st.button("Load demonstration example", width="stretch"):
        demo_pairs = [
            (first, second)
            for index, first in enumerate(DEMO_DOCUMENTS)
            for second in DEMO_DOCUMENTS[index + 1:]
        ]
        previous_pair = frozenset(st.session_state.get("demo_pair_names", ()))
        available_pairs = [
            pair for pair in demo_pairs
            if frozenset((pair[0][0], pair[1][0])) != previous_pair
        ]
        selected_pair = random.choice(available_pairs or demo_pairs)
        if random.choice((False, True)):
            selected_pair = selected_pair[::-1]
        (name_a, demo_a), (name_b, demo_b) = selected_pair
        st.session_state["document_a"] = demo_a
        st.session_state["document_b"] = demo_b
        st.session_state["demo_pair_names"] = (name_a, name_b)
with button_clear:
    if st.button("Clear inputs", width="stretch"):
        st.session_state["document_a"] = ""
        st.session_state["document_b"] = ""
        st.session_state.pop("demo_pair_names", None)

if st.session_state.get("demo_pair_names"):
    demo_name_a, demo_name_b = st.session_state["demo_pair_names"]
    st.caption(f"Last demo pair: {demo_name_a} vs {demo_name_b} · Randomly selected from {len(DEMO_DOCUMENTS)} synthetic documents; topic and passage length can affect the comparison.")

st.session_state.setdefault("document_a", "")
st.session_state.setdefault("document_b", "")
doc_col_a, doc_col_b = st.columns(2)
with doc_col_a:
    text_a = st.text_area("Document A", key="document_a", height=220, placeholder="Paste or type the first English text here…")
with doc_col_b:
    text_b = st.text_area("Document B", key="document_b", height=220, placeholder="Paste or type the second English text here…")

features_a, features_b = extract_features(text_a), extract_features(text_b)
counts_a, counts_b = st.columns(2)
with counts_a:
    st.caption(f"Document A · {features_a['character_count']:,} characters · {features_a['word_count']:,} words · {features_a['sentence_count']:,} sentences")
with counts_b:
    st.caption(f"Document B · {features_b['character_count']:,} characters · {features_b['word_count']:,} words · {features_b['sentence_count']:,} sentences")

tabs = st.tabs([
    "Overview", "Function Words", "POS Analysis", "Word N-grams",
    "Character N-grams", "Vocabulary & Structure", "Document Comparison", "Authorship Laboratory",
])

with tabs[0]:
    st.subheader("Basic linguistic statistics")
    if not text_a.strip() or not text_b.strip():
        st.info("Enter text in both documents to compare them. Empty or whitespace-only documents have zero words, sentences, averages, and vocabulary rates.")
    st.dataframe(comparison_rows(features_a, features_b, OVERVIEW_KEYS), hide_index=True, width="stretch")
    st.caption("Character count includes spaces and line breaks. Words are Unicode letter or number tokens with internal apostrophes; contractions stay together. Sentences use terminal punctuation. Paragraphs are non-empty blocks separated by blank lines. TTR is sensitive to document length.")

    punctuation_rows = []
    for mark in features_a["punctuation"]:
        pa, pb = features_a["punctuation"][mark], features_b["punctuation"][mark]
        delta = pb["per_100_words"] - pa["per_100_words"]
        punctuation_rows.append({
            "Mark": mark,
            "A count": pa["count"], "A per 100 words": pa["per_100_words"],
            "A per 1,000 characters": pa["per_1000_characters"], "A per sentence": pa["per_sentence"],
            "B count": pb["count"], "B per 100 words": pb["per_100_words"],
            "B per 1,000 characters": pb["per_1000_characters"], "B per sentence": pb["per_sentence"],
            "Absolute rate difference": abs(delta),
            "Relative rate difference vs A (%)": delta / abs(pa["per_100_words"]) * 100 if pa["per_100_words"] else None,
        })
    with st.expander("Punctuation counts and normalized rates"):
        st.dataframe(pd.DataFrame(punctuation_rows), hide_index=True, width="stretch")
        st.caption("Rates are shown per 100 words, per sentence, and per 1,000 characters in the feature engine. The overview table uses per-100-word rates; empty denominators produce zero rates.")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("**Sentence, word, and vocabulary measures**")
        st.pyplot(feature_comparison_chart(features_a, features_b), width="stretch")
    with chart_right:
        st.markdown("**Punctuation frequency**")
        st.plotly_chart(punctuation_comparison_chart(features_a, features_b), width="stretch")
    st.markdown("**Frequent content words**")
    st.plotly_chart(word_frequency_chart(text_a, text_b), width="stretch")
    st.caption("The word-frequency plot excludes the small English stop-word set in src/stylometry.py. Topic words can dominate lexical comparison.")

    with st.expander("Phase 1 interpretation and limits"):
        st.markdown("Average sentence length summarizes pacing; average word length summarizes token form. TTR estimates vocabulary variety but tends to fall as text grows. All descriptive measures are affected by topic, genre, length, editing, and quotations. Sentence splitting is heuristic; no feature here establishes authorship.")

with tabs[1]:
    st.subheader("Function-word frequencies")
    st.markdown("Counts and rates are case-insensitive. This documented, configurable small list covers selected articles, pronouns, prepositions, and conjunctions; it is not a complete inventory or an author fingerprint.")
    fw_a, fw_b = function_word_frequencies(text_a), function_word_frequencies(text_b)
    rows, rates_a, rates_b = [], {}, {}
    for category, entries in FUNCTION_WORDS.items():
        for word in entries:
            a, b = fw_a[category][word], fw_b[category][word]
            key = f"{word} · {category}"
            rates_a[key], rates_b[key] = float(a["per_1000_tokens"]), float(b["per_1000_tokens"])
            rate_delta = b["per_1000_tokens"] - a["per_1000_tokens"]
            rows.append({
                "Category": category.title(), "Word": word,
                "A count": a["count"], "A per 1,000 tokens": a["per_1000_tokens"],
                "B count": b["count"], "B per 1,000 tokens": b["per_1000_tokens"],
                "Absolute rate difference": abs(rate_delta),
                "Relative rate difference vs A (%)": rate_delta / abs(a["per_1000_tokens"]) * 100 if a["per_1000_tokens"] else None,
            })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.plotly_chart(normalized_frequency_chart(rates_a, rates_b, title="Function-word rates", y_axis_title="Occurrences per 1,000 word tokens"), width="stretch")
    top_words = sorted(rows, key=lambda row: -(row["A per 1,000 tokens"] + row["B per 1,000 tokens"]))[:8]
    st.caption("Most frequent listed items: " + (", ".join(f"{row['Word']} ({row['A count'] + row['B count']})" for row in top_words) if top_words else "none"))

with tabs[2]:
    st.subheader("Part-of-speech tag distribution")
    st.markdown("NLTK's English averaged-perceptron tagger reports Penn Treebank tags. Tags are normalized by the number of tagged word tokens. The app does not download language resources automatically.")
    with st.spinner("Checking the local English POS tagger resource…"):
        pos_a, pos_b = cached_pos_tags(text_a), cached_pos_tags(text_b)
    if not pos_a["available"] or not pos_b["available"]:
        st.warning(pos_a["warning"] or pos_b["warning"])
        st.code("python -m nltk.downloader averaged_perceptron_tagger_eng", language="powershell")
    else:
        tags_a, tags_b = pos_a["distribution"], pos_b["distribution"]
        pos_rows, pos_rates_a, pos_rates_b, descriptions = [], {}, {}, {}
        for tag in sorted(set(tags_a) | set(tags_b)):
            a, b = tags_a.get(tag, {}), tags_b.get(tag, {})
            frequency_delta = b.get("frequency", 0) - a.get("frequency", 0)
            pos_rows.append({
                "Tag": tag, "Meaning": PENN_TAG_DESCRIPTIONS.get(tag, "Penn Treebank tag"),
                "A count": a.get("count", 0), "A frequency (%)": a.get("frequency", 0) * 100,
                "B count": b.get("count", 0), "B frequency (%)": b.get("frequency", 0) * 100,
                "Absolute frequency difference": abs(frequency_delta),
                "Relative frequency difference vs A (%)": frequency_delta / abs(a.get("frequency", 0)) * 100 if a.get("frequency", 0) else None,
            })
            pos_rates_a[tag], pos_rates_b[tag] = float(a.get("frequency", 0)), float(b.get("frequency", 0))
            descriptions[tag] = PENN_TAG_DESCRIPTIONS.get(tag, "Penn Treebank tag")
        pos_rows.sort(key=lambda row: -(row["A frequency (%)"] + row["B frequency (%)"]))
        st.dataframe(pd.DataFrame(pos_rows), hide_index=True, width="stretch")
        st.plotly_chart(normalized_frequency_chart(pos_rates_a, pos_rates_b, title="POS-tag distribution", y_axis_title="Share of tagged tokens (0–1)", descriptions=descriptions), width="stretch")
        st.caption(f"Tagged tokens: A {pos_a['tagged_token_count']:,} · B {pos_b['tagged_token_count']:,}. A short passage may not support stable tag-distribution estimates.")
    with st.expander("POS methodology"):
        st.markdown("The tagger assigns context-dependent Penn Treebank tags to the offline Unicode word tokens. A missing NLTK package or missing English tagger model leaves this section disabled with setup instructions; the system never invents fallback tags.")

with tabs[3]:
    st.subheader("Word n-grams")
    col_range, col_limit, col_stop = st.columns([1, 1, 1.4])
    with col_range:
        word_range = st.slider("Word n-gram range", min_value=1, max_value=3, value=(1, 3), key="word_ngram_range")
    with col_limit:
        vocabulary_limit = st.number_input("Maximum vocabulary", min_value=10, max_value=5000, value=300, step=10, key="word_ngram_vocab")
    with col_stop:
        remove_stops = st.checkbox("Remove listed English stop words", value=False, key="word_ngram_stop")
    st.caption("Unigrams, bigrams, and trigrams are formed from the same case-folded Unicode word tokens. Stop-word filtering is off by default; when enabled, removing tokens can join words that were not adjacent in the source.")
    words_a = word_ngram_counts(text_a, tuple(word_range), int(vocabulary_limit), remove_stops)
    words_b = word_ngram_counts(text_b, tuple(word_range), int(vocabulary_limit), remove_stops)
    st.dataframe(ngram_table(words_a, words_b, kind="word"), hide_index=True, width="stretch")
    ngram_reps = cached_representations(text_a, text_b, tuple(word_range), (3, 5), int(vocabulary_limit), remove_stops, 1.0, 1.0)
    selected = ngram_reps["word_ngram_tfidf"]
    st.markdown("**Most influential pair-local word n-grams (TF-IDF)**")
    if selected["available"]:
        tfidf_rows = []
        for key, label in (("top_a", "A"), ("top_b", "B"), ("distinctive_a", "Distinctive to A"), ("distinctive_b", "Distinctive to B")):
            tfidf_rows.extend({"Document": label, "N-gram": item["feature"], "TF-IDF weight/difference": item.get("weight", item.get("difference"))} for item in selected[key][:10])
        st.dataframe(pd.DataFrame(tfidf_rows), hide_index=True, width="stretch")
        st.caption(f"Local pair word n-gram cosine: {selected['similarity']:.3f}. {selected['warning'] or 'Vectorizer fit on these two documents only; this is not a trained model.'}")
    else:
        st.info(selected["warning"])

with tabs[4]:
    st.subheader("Character n-grams")
    char_min, char_max, char_limit = st.columns(3)
    with char_min:
        char_start = st.number_input("Minimum characters", min_value=2, max_value=8, value=3, key="char_min")
    with char_max:
        char_end = st.number_input("Maximum characters", min_value=2, max_value=10, value=5, key="char_max")
    with char_limit:
        char_vocab = st.number_input("Maximum character vocabulary", min_value=10, max_value=5000, value=300, step=10, key="char_vocab")
    if char_end < char_start:
        st.error("Maximum character n-gram size must be greater than or equal to its minimum.")
    else:
        char_range = (int(char_start), int(char_end))
        chars_a = character_ngram_counts(text_a, char_range, int(char_vocab))
        chars_b = character_ngram_counts(text_b, char_range, int(char_vocab))
        st.caption("NFC-normalized text is case-folded and whitespace is collapsed to single spaces. Spaces and punctuation remain in the character stream. Character patterns can reflect spelling and formatting but do not uniquely identify an author.")
        st.dataframe(ngram_table(chars_a, chars_b, kind="char"), hide_index=True, width="stretch")
        char_reps = cached_representations(text_a, text_b, (1, 3), char_range, int(char_vocab), False, 1.0, 1.0)
        char_result = char_reps["character_ngram_tfidf"]
        st.markdown("**Most influential character patterns (pair-local TF-IDF)**")
        if char_result["available"]:
            pattern_rows = []
            for key, label in (("top_a", "A"), ("top_b", "B"), ("distinctive_a", "Distinctive to A"), ("distinctive_b", "Distinctive to B")):
                pattern_rows.extend({"Document": label, "Pattern": item["feature"], "TF-IDF weight/difference": item.get("weight", item.get("difference"))} for item in char_result[key][:10])
            st.dataframe(pd.DataFrame(pattern_rows), hide_index=True, width="stretch")
            st.caption(f"Local character n-gram cosine: {char_result['similarity']:.3f}. Pair-fitted and sensitive to vocabulary, topic, spelling, and formatting.")
        else:
            st.info(char_result["warning"])

with tabs[5]:
    st.subheader("Vocabulary and sentence structure")
    detail_keys = ["type_token_ratio", "root_ttr", "log_ttr", "hapax_count", "hapax_rate_per_1000", "dislegomena_count", "dislegomena_rate_per_1000", "sentence_length_stddev"]
    st.dataframe(comparison_rows(features_a, features_b, detail_keys), hide_index=True, width="stretch")
    if features_a["word_count"] < 50 or features_b["word_count"] < 50:
        st.warning("At least one sample has fewer than 50 word tokens. Hapax rates, lexical diversity, and sentence-structure summaries may be unstable.")
    st.markdown("**Word-length distributions**")
    lengths_a = {str(length): float(data["per_1000_tokens"]) for length, data in features_a["word_length_distribution"].items()}
    lengths_b = {str(length): float(data["per_1000_tokens"]) for length, data in features_b["word_length_distribution"].items()}
    st.plotly_chart(normalized_frequency_chart(lengths_a, lengths_b, title="Word length", y_axis_title="Words per 1,000 tokens; x-axis is character length"), width="stretch")
    max_sentence_len = max(features_a["sentence_length_distribution"] + features_b["sentence_length_distribution"], default=0)
    sentence_rates_a = {str(length): features_a["sentence_length_distribution"].count(length) for length in range(1, max_sentence_len + 1)}
    sentence_rates_b = {str(length): features_b["sentence_length_distribution"].count(length) for length in range(1, max_sentence_len + 1)}
    st.plotly_chart(normalized_frequency_chart(sentence_rates_a, sentence_rates_b, title="Sentence-length distributions", y_axis_title="Sentence count; x-axis is words per sentence"), width="stretch")
    st.caption("TTR is unique types ÷ tokens and is strongly length-sensitive. Root TTR (Guiraud) is types ÷ √tokens; log TTR is log(types) ÷ log(tokens), defined as zero below two tokens. Hapax legomena occur once; dislegomena occur twice. Their rates use all word tokens as denominator. Sentence-length standard deviation is population standard deviation and is zero for fewer than two detected sentences.")

with tabs[6]:
    st.subheader("Document comparison: lexical versus stylistic")
    st.markdown("Each text vectorizer below is fitted on this pair only. Pairwise similarity is a descriptive local comparison and should not be used as an authorship probability.")
    style_map, style_map_warning = document_style_map(text_a, text_b)
    if style_map is not None:
        st.plotly_chart(style_map, width="stretch", config={"displaylogo": False})
        st.caption("Each marker is a contiguous 100-word passage; teal circles are Document A and terracotta triangles are Document B. Shaded outlines show each document’s passage range. Axes are PCA summaries of standardized function-word rates, sentence and word-length statistics, vocabulary richness, hapax rates, and punctuation rates. Nearby passages have more similar measured profiles in this projection; topic and sample size can still affect the pattern.")
    else:
        st.info(style_map_warning)
    weight_col, style_col, vocab_col = st.columns(3)
    with weight_col:
        text_weight = st.slider("Text-feature weight", 0.0, 2.0, 1.0, 0.1, key="comparison_text_weight")
    with style_col:
        style_weight = st.slider("Numerical-style weight", 0.0, 2.0, 1.0, 0.1, key="comparison_style_weight")
    with vocab_col:
        comparison_vocab = st.number_input("Maximum comparison vocabulary", min_value=10, max_value=5000, value=500, step=10, key="comparison_vocab")
    if text_weight == 0 and style_weight == 0:
        st.error("At least one comparison weight must be greater than zero.")
    else:
        representations = cached_representations(text_a, text_b, (1, 3), (3, 5), int(comparison_vocab), False, text_weight, style_weight)
        score_labels = {
            "word_unigram_tfidf": "Word unigram TF-IDF · lexical overlap",
            "word_unigram_bigram_tfidf": "Word unigram + bigram TF-IDF · lexical phrase overlap",
            "character_ngram_tfidf": "Character 3–5 gram TF-IDF · orthographic overlap",
            "stylometric_numeric": "Normalized numerical features · stylistic profile",
            "combined_experimental": "Weighted text + numerical features · experimental",
        }
        score_rows = []
        for key, label in score_labels.items():
            result = representations[key]
            score_rows.append({"Representation": label, "Cosine similarity": result["similarity"], "Features": result["feature_count"], "Status": result["warning"] or "Available"})
        st.dataframe(pd.DataFrame(score_rows), hide_index=True, width="stretch")
        st.caption("The combined score uses cosine([text_weight × L2-normalized unigram TF-IDF, style_weight × L2-normalized numerical style vector]). Per-feature style inputs are scaled by the pairwise maximum absolute value. It is an experimental comparison score, not a probability or universal author-similarity formula.")
        st.markdown("**Distinctive text-derived features**")
        distinctive_rows = []
        for key, label in (("word_unigram_bigram_tfidf", "Word n-gram"), ("character_ngram_tfidf", "Character n-gram")):
            result = representations[key]
            for side, field in (("A", "distinctive_a"), ("B", "distinctive_b")):
                distinctive_rows.extend({"Representation": label, "Document": side, "Feature": item["feature"], "Pairwise TF-IDF difference": item["difference"]} for item in result[field][:10])
        if distinctive_rows:
            st.dataframe(pd.DataFrame(distinctive_rows), hide_index=True, width="stretch")
        else:
            st.info("No distinct text-derived features are available yet; add text to both documents.")

    with st.expander("Comparison methodology and limits"):
        st.markdown("Word unigram and unigram–bigram representations use scikit-learn TF-IDF with the project's Unicode tokenizer. Character TF-IDF uses normalized, case-folded characters with spaces and punctuation retained. Pair-local vectorizers describe only the supplied documents; future training workflows must fit vectorizers on their training collection. Topic, genre, text length, editing, quotations, and short samples affect every comparison. No universal author score is calculated.")

with tabs[7]:
    st.subheader("Authorship attribution research lab")
    st.markdown("Train and compare interpretable text-classification baselines on a labeled corpus. Results describe a closed-set experiment and do not establish real-world identity or authorship.")
    st.info("Use only text you have permission to analyze. Record corpus provenance and licence below. Synthetic demo results are for checking the workflow and have no research meaning.")

    source_upload = st.file_uploader("Upload a labeled CSV (required columns: text, author; optional: document_id, source, genre)", type=["csv"], key="authorship_csv")
    load_demo_col, load_uploaded_col, _ = st.columns([1, 1, 2])
    with load_demo_col:
        if st.button("Load synthetic demonstration corpus", key="load_synthetic"):
            st.session_state["authorship_frame"] = synthetic_demo_dataset()
            st.session_state["authorship_result"] = None
            st.session_state["authorship_dataset_kind"] = "Synthetic demonstration"
    with load_uploaded_col:
        if st.button("Use uploaded CSV", disabled=source_upload is None, key="load_csv"):
            try:
                uploaded_frame = pd.read_csv(source_upload)
                cleaned_frame, validation = validate_dataset(uploaded_frame)
                st.session_state["authorship_frame"] = cleaned_frame
                st.session_state["authorship_quality"] = validation
                st.session_state["authorship_result"] = None
                st.session_state["authorship_dataset_kind"] = "User-provided CSV"
            except Exception as error:
                st.error(f"Could not load dataset: {error}")

    frame = st.session_state.get("authorship_frame")
    result = st.session_state.get("authorship_result")
    if frame is not None:
        try:
            frame, quality = validate_dataset(frame)
            st.session_state["authorship_frame"] = frame
            st.session_state["authorship_quality"] = quality
            st.caption(f"{st.session_state.get('authorship_dataset_kind', 'Dataset')} · {quality['usable_rows']:,} usable rows · {quality['authors']} authors · imbalance ratio {quality['imbalance_ratio']:.2f}" if quality["imbalance_ratio"] is not None else "Dataset loaded.")
            for warning in quality["warnings"]:
                st.warning(warning)
            st.dataframe(pd.DataFrame([{"author": label, "samples": count} for label, count in quality["class_counts"].items()]).sort_values("samples"), hide_index=True, width="stretch")
            duplicate_check = find_near_duplicates(frame)
            if duplicate_check["pairs"]:
                st.warning(f"Near-duplicate scan found {len(duplicate_check['pairs'])} candidate text pairs at similarity ≥ 0.92. Review source leakage before modeling.")
                st.dataframe(pd.DataFrame(duplicate_check["pairs"]), hide_index=True, width="stretch")
            elif duplicate_check["warning"]:
                st.caption(duplicate_check["warning"])
        except DatasetValidationError as error:
            st.error(str(error))
            frame = None

    if frame is not None:
        with st.expander("Dataset provenance and permission", expanded=True):
            provenance = st.text_input("Where did the corpus come from?", key="authorship_provenance", placeholder="Archive, collection, or project")
            license_note = st.text_input("Licence / permission", key="authorship_license", placeholder="Licence name or permission basis")
        authors_available = eligible_authors(frame, 3)
        if len(authors_available) < 2:
            st.error("At least two authors with three or more usable passages each are needed to run a classification experiment.")
        else:
            control_a, control_b, control_c = st.columns(3)
            with control_a:
                minimum_samples = st.number_input("Minimum passages per author", min_value=3, max_value=100, value=5, step=1, key="lab_min_samples")
                eligible = eligible_authors(frame, int(minimum_samples))
                selected_authors = st.multiselect("Authors to include", eligible, default=eligible, key="lab_authors")
                test_fraction = st.slider("Test fraction", min_value=0.1, max_value=0.35, value=0.2, step=0.05, key="lab_test_fraction")
            with control_b:
                validation_fraction = st.slider("Validation fraction", min_value=0.1, max_value=0.35, value=0.2, step=0.05, key="lab_val_fraction")
                group_options = ["Passage-level random split"] + [column for column in ("document_id", "source") if column in frame.columns]
                default_group_index = group_options.index("document_id") if "document_id" in group_options else 0
                selected_group = st.selectbox("Split strategy", group_options, index=default_group_index, help="When selected, all passages from one document/source stay in a single partition.", key="lab_group")
                group_column = None if selected_group == "Passage-level random split" else selected_group
            with control_c:
                random_seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, key="lab_seed")
                max_vocabulary = st.number_input("Maximum text vocabulary", min_value=100, max_value=20000, value=3000, step=100, key="lab_max_vocab")
                balanced = st.checkbox("Balance class weights for LR / SVC", value=False, key="lab_balance")

            st.markdown("**Representations**")
            selected_representations = st.multiselect("Feature representations", options=list(REPRESENTATIONS), default=["stylometric", "word_unigram_bigram", "character", "hybrid_word"], format_func=lambda key: REPRESENTATIONS[key], key="lab_representations")
            st.markdown("**Classifiers**")
            selected_models = st.multiselect("Models", options=list(MODEL_LABELS), default=list(MODEL_LABELS), format_func=lambda key: MODEL_LABELS[key], key="lab_models")
            ngram_a, ngram_b, c_a, c_b, weight_a, weight_b = st.columns(6)
            with ngram_a:
                word_min = st.number_input("Word n-gram min", 1, 3, 1, key="lab_word_min")
            with ngram_b:
                word_max = st.number_input("Word n-gram max", 1, 3, 2, key="lab_word_max")
            with c_a:
                char_minimum = st.number_input("Character n-gram min", 2, 8, 3, key="lab_char_min")
            with c_b:
                char_maximum = st.number_input("Character n-gram max", 2, 10, 5, key="lab_char_max")
            with weight_a:
                style_weight = st.number_input("Hybrid style weight", 0.0, 3.0, 1.0, 0.1, key="lab_style_weight")
            with weight_b:
                text_weight = st.number_input("Hybrid text weight", 0.0, 3.0, 1.0, 0.1, key="lab_text_weight")
            include_pos = st.checkbox("Include POS tag features (requires the local NLTK English tagger data)", value=False, key="lab_include_pos")
            if include_pos:
                st.caption("Missing tagger data will stop the experiment with setup instructions; the pipeline never invents POS features.")
            c_candidates = st.multiselect("C values for Logistic Regression / Linear SVC", [0.01, 0.1, 1.0, 10.0, 100.0], default=[0.1, 1.0, 10.0], key="lab_c_values")
            alpha_candidates = st.multiselect("Alpha values for MultinomialNB", [0.01, 0.1, 0.5, 1.0, 5.0], default=[0.1, 1.0], key="lab_nb_alphas")
            can_run = bool(selected_authors and len(selected_authors) >= 2 and selected_representations and selected_models and word_min <= word_max and char_minimum <= char_maximum and test_fraction + validation_fraction < 0.8 and style_weight + text_weight > 0 and c_candidates and alpha_candidates)
            if st.button("Run validation and test experiment", type="primary", disabled=not can_run, key="run_authorship"):
                try:
                    selected_frame = filter_authors(frame, selected_authors, int(minimum_samples))
                    with st.spinner("Fitting candidate pipelines on training data and selecting by validation macro-F1…"):
                        st.session_state["authorship_result"] = run_experiments(
                            selected_frame, representations=selected_representations, models=selected_models,
                            test_size=float(test_fraction), validation_size=float(validation_fraction),
                            random_seed=int(random_seed), group_column=group_column,
                            minimum_samples_per_author=int(minimum_samples), max_vocabulary=int(max_vocabulary),
                            word_ngram_range=(int(word_min), int(word_max)), char_ngram_range=(int(char_minimum), int(char_maximum)),
                            include_pos=include_pos, class_weight="balanced" if balanced else None,
                            c_values=tuple(float(value) for value in c_candidates), nb_alphas=tuple(float(value) for value in alpha_candidates),
                            style_weight=float(style_weight), text_weight=float(text_weight),
                        )
                        st.session_state["authorship_frame_selected"] = selected_frame
                        for key in ("robustness_length", "robustness_domain", "robustness_splits", "robustness_ablation", "robustness_preprocessing"):
                            st.session_state.pop(key, None)
                except Exception as error:
                    st.error(f"Experiment could not run: {error}")
            result = st.session_state.get("authorship_result")

    if result is not None:
        winner = result["winner"]
        st.success(f"Validation-selected model: {winner['model_label']} · {winner['representation_label']} · {winner['parameter']} · validation macro-F1 {winner['validation_macro_f1']:.3f}")
        st.markdown("Candidate models are ranked on validation macro-F1. The held-out test set was evaluated once for the selected candidate.")
        st.dataframe(result["candidates"].drop(columns=["candidate_id"]), hide_index=True, width="stretch")
        metric = result["test_metrics"]
        metric_cols = st.columns(5)
        for column, label, value in zip(metric_cols, ("Test accuracy", "Test macro precision", "Test macro recall", "Test macro-F1", "Test weighted F1"), (metric["accuracy"], metric["macro_precision"], metric["macro_recall"], metric["macro_f1"], metric["weighted_f1"])):
            column.metric(label, f"{value:.3f}")
        st.caption(f"Fit {result['test_fit_seconds']:.2f}s · prediction latency {result['test_prediction_ms_per_sample']:.3f} ms/sample · held-out rows {result['split']['sizes']['test']}")
        st.markdown("**Per-author test metrics**")
        st.dataframe(metric["per_author"], hide_index=True, width="stretch")
        import plotly.graph_objects as go
        matrix = metric["confusion_matrix"]
        fig = go.Figure(data=go.Heatmap(z=matrix, x=[str(label) for label in metric["labels"]], y=[str(label) for label in metric["labels"]], colorscale=[[0, "#F4F6F4"], [0.5, "#9ABABD"], [1, "#3E7782"]], text=matrix, texttemplate="%{text}", hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Samples: %{z}<extra></extra>"))
        fig.update_layout(title="Held-out test confusion matrix", xaxis_title="Predicted author", yaxis_title="Actual author")
        st.plotly_chart(style_figure(fig), width="stretch", config={"displaylogo": False})
        st.markdown("**Features most associated with each class in the selected model**")
        explanation = top_model_features(result["winner_pipeline"], limit=10)
        if explanation:
            st.dataframe(pd.DataFrame(explanation), hide_index=True, width="stretch")
            st.caption("Linear model weights or Naive Bayes log-likelihood differences are model associations, not causal explanations or proof of authorship.")
        else:
            st.info("This pipeline does not expose class-specific feature weights.")
        if result["test_misclassifications"]:
            with st.expander("Misclassified held-out passages"):
                st.dataframe(pd.DataFrame(result["test_misclassifications"]), hide_index=True, width="stretch")
        if result["test_correct_classifications"]:
            with st.expander("Correctly classified held-out passages"):
                st.dataframe(pd.DataFrame(result["test_correct_classifications"]), hide_index=True, width="stretch")
        if result["common_confusions"]:
            st.markdown("**Most common author confusions**")
            st.dataframe(pd.DataFrame(result["common_confusions"]), hide_index=True, width="stretch")

        st.markdown("**Try the selected model on Documents A and B**")
        prediction_docs = [text for text in (text_a, text_b) if text.strip()]
        if prediction_docs:
            predicted_authors = result["winner_pipeline"].predict(prediction_docs)
            st.dataframe(pd.DataFrame({"Input": ["Document A", "Document B"][:len(prediction_docs)], "Forced known-class prediction": predicted_authors}), hide_index=True, width="stretch")
            st.caption("These are closed-set predictions among the selected labels. The classifier has no reject option and may be confidently wrong on an unknown author, genre, or source.")
            if text_a.strip() and text_b.strip():
                fitted_comparison = compare_texts_with_fitted_pipeline(result["winner_pipeline"], text_a, text_b)
                st.write(f"Cosine in the selected model's fitted feature space: {fitted_comparison['cosine_similarity']:.3f}" if fitted_comparison["cosine_similarity"] is not None else "Fitted feature-space cosine unavailable for empty inputs.")
                st.caption(f"{fitted_comparison['warning']} This transform reuses the model's exact fitted vocabulary and normalization.")
                if fitted_comparison["feature_differences"]:
                    with st.expander("Inspect strongest fitted-feature differences between A and B"):
                        st.dataframe(pd.DataFrame(fitted_comparison["feature_differences"]), hide_index=True, width="stretch")
        else:
            st.caption("Add text to Document A or Document B above to see closed-set model predictions here.")

        with st.expander("Robustness and generalization experiments", expanded=False):
            selected_frame = st.session_state.get("authorship_frame_selected", result["train_rows"])
            base_config = result["config"]
            run_kwargs = {
                "test_size": result["split"]["test_size"], "validation_size": result["split"]["validation_size"],
                "random_seed": result["split"]["random_seed"], "group_column": result["split"]["group_column"],
                "minimum_samples_per_author": base_config["minimum_samples_per_author"], "max_vocabulary": base_config["max_vocabulary"],
                "word_ngram_range": tuple(base_config["word_ngram_range"]), "char_ngram_range": tuple(base_config["char_ngram_range"]),
                "include_pos": base_config["include_pos"], "class_weight": base_config["class_weight"],
                "c_values": tuple(base_config["c_values"]), "nb_alphas": tuple(base_config["nb_alphas"]),
                "style_weight": base_config["style_weight"], "text_weight": base_config["text_weight"],
                "word_case_sensitive": base_config["word_case_sensitive"], "char_case_sensitive": base_config["char_case_sensitive"],
                "remove_stop_words": base_config["remove_stop_words"],
            }
            thresholds = st.multiselect("Minimum word lengths to evaluate on held-out passages", [0, 20, 50, 100, 200], default=[0, 50, 100], key="robust_length_thresholds")
            if st.button("Evaluate test performance by length", key="run_length_robustness"):
                st.session_state["robustness_length"] = evaluate_document_length_robustness(result, thresholds=tuple(thresholds))
            length_result = st.session_state.get("robustness_length")
            if length_result:
                st.markdown("**Minimum length subsets**")
                st.dataframe(length_result["thresholds"], hide_index=True, width="stretch")
                st.markdown("**Short, medium, and long test groups**")
                st.dataframe(length_result["length_groups"], hide_index=True, width="stretch")
                with st.expander("Length-group per-author results"):
                    for detail in length_result["length_group_details"]:
                        st.markdown(f"**{detail['length_group']} · {detail['samples']} samples**")
                        st.dataframe(detail["per_author"], hide_index=True, width="stretch")

            domain_columns = [column for column in ("genre", "topic") if column in selected_frame.columns]
            if domain_columns:
                domain_column = st.selectbox("Held-out performance by genre/topic", domain_columns, key="robust_domain_column")
                if st.button("Evaluate genre/topic robustness", key="run_domain_robustness"):
                    st.session_state["robustness_domain"] = evaluate_domain_robustness(result, column=domain_column)
                domain_result = st.session_state.get("robustness_domain")
                if domain_result:
                    st.dataframe(domain_result["summary"], hide_index=True, width="stretch")
                    st.caption(domain_result["warning"])
                    with st.expander("Domain per-author metrics"):
                        for detail in domain_result["details"]:
                            st.markdown(f"**{detail['domain']} · {detail['samples']} samples**")
                            st.dataframe(detail["per_author"], hide_index=True, width="stretch")
            else:
                st.caption("Add an optional genre or topic column to the CSV to evaluate held-out performance across those domains.")

            available_group_columns = [column for column in ("document_id", "source") if column in selected_frame.columns]
            if available_group_columns:
                robustness_group = st.selectbox("Group column for source-leakage comparison", available_group_columns, key="robust_group_column")
                if st.button("Compare passage split with group-aware split", key="run_split_robustness"):
                    with st.spinner("Repeating the same validation-selection protocol for both split strategies…"):
                        st.session_state["robustness_splits"] = compare_random_and_group_splits(
                            selected_frame, group_column=robustness_group,
                            run_kwargs={**run_kwargs, "group_column": None, "representations": [winner["representation"]], "models": [winner["model"]]},
                        )
                split_result = st.session_state.get("robustness_splits")
                if split_result:
                    st.dataframe(split_result["summary"], hide_index=True, width="stretch")
                    if split_result["grouped_minus_random_macro_f1"] is not None:
                        st.caption(f"Grouped minus random test macro-F1: {split_result['grouped_minus_random_macro_f1']:+.3f}. {split_result['interpretation']}")
                    else:
                        st.caption(f"{split_result['interpretation']} At least one requested protocol was unavailable.")
            else:
                st.caption("No document_id/source column is available for a grouped leakage comparison.")

            ablation_col, sensitivity_col = st.columns(2)
            with ablation_col:
                if st.button("Run feature ablation", key="run_feature_ablation"):
                    with st.spinner("Comparing stylometric, word, character, and hybrid representations…"):
                        st.session_state["robustness_ablation"] = run_feature_ablation(
                            selected_frame,
                            run_kwargs={**run_kwargs, "group_column": result["split"]["group_column"]},
                            model=winner["model"],
                        )
                ablation_result = st.session_state.get("robustness_ablation")
                if ablation_result:
                    st.dataframe(ablation_result["summary"], hide_index=True, width="stretch")
                    st.caption(ablation_result["warning"] + " Do not choose a representation based on test-set results alone.")
                    with st.expander("Per-author ablation metrics"):
                        st.dataframe(ablation_result["per_author"], hide_index=True, width="stretch")
            with sensitivity_col:
                if st.button("Run preprocessing sensitivity", key="run_preprocessing_sensitivity"):
                    with st.spinner("Comparing case, n-gram, character-range, and stop-word settings on the same protocol…"):
                        st.session_state["robustness_preprocessing"] = run_preprocessing_sensitivity(
                            selected_frame,
                            run_kwargs={**run_kwargs, "group_column": result["split"]["group_column"]},
                            model=winner["model"],
                        )
                sensitivity_result = st.session_state.get("robustness_preprocessing")
                if sensitivity_result:
                    st.dataframe(sensitivity_result["summary"], hide_index=True, width="stretch")
                    st.caption(sensitivity_result["warning"])

        include_error_text = st.checkbox("Include held-out misclassified text excerpts in the report", value=False, key="report_include_error_text", help="Off by default to reduce accidental disclosure of private or copyrighted text.")
        report_robustness = {}
        if st.session_state.get("robustness_length"):
            report_robustness["Document length robustness"] = st.session_state["robustness_length"]["thresholds"]
            report_robustness["Short, medium, long test groups"] = st.session_state["robustness_length"]["length_groups"]
        if st.session_state.get("robustness_splits"):
            report_robustness["Random versus grouped splitting"] = st.session_state["robustness_splits"]["summary"]
        if st.session_state.get("robustness_domain"):
            report_robustness[f"Held-out {st.session_state['robustness_domain']['column']} robustness"] = st.session_state["robustness_domain"]["summary"]
        if st.session_state.get("robustness_ablation"):
            report_robustness["Feature ablation"] = st.session_state["robustness_ablation"]["summary"]
        if st.session_state.get("robustness_preprocessing"):
            report_robustness["Preprocessing sensitivity"] = st.session_state["robustness_preprocessing"]["summary"]
        report_text = experiment_report(result, provenance=st.session_state.get("authorship_provenance", "Not provided"), license_note=st.session_state.get("authorship_license", "Not provided"), include_error_text=include_error_text, robustness=report_robustness)
        st.download_button("Download experiment report (Markdown)", data=report_text, file_name="authorship_experiment_report.md", mime="text/markdown", key="download_authorship_report")
        if st.button("Save selected model to local app storage", key="save_authorship_model"):
            try:
                selected_training = st.session_state.get("authorship_frame_selected", result["train_rows"])
                manifest = save_model(result["winner_pipeline"], metadata={
                    "winner": winner, "configuration": result["config"],
                    "validation_metrics": {key: winner[key] for key in winner if key.startswith("validation_")},
                    "test_metrics": {key: metric[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")},
                    "provenance": st.session_state.get("authorship_provenance", "Not provided"),
                    "licence_note": st.session_state.get("authorship_license", "Not provided"),
                    "authors": result["authors"], "random_seed": result["split"]["random_seed"],
                }, training_rows=selected_training.iloc[result["split"]["train_indices"]] if len(selected_training) == len(result["train_rows"]) + len(result["validation_rows"]) + len(result["test_rows"]) else result["train_rows"])
                st.success(f"Saved local model id: {manifest['model_id']}")
            except Exception as error:
                st.error(f"Could not save model: {error}")

    saved_models = list_saved_models()
    if saved_models:
        with st.expander("Previously saved local models"):
            for manifest in saved_models:
                st.write(f"{manifest.get('created_utc', '')} · {manifest.get('model_id')} · {manifest.get('winner', {}).get('model_label', '')} · {manifest.get('provenance', 'provenance not provided')}")
            selected_id = st.selectbox("Load a saved model", [entry["model_id"] for entry in saved_models], key="load_saved_model_id")
            if st.button("Load selected saved model", key="load_saved_model"):
                try:
                    loaded_pipeline, loaded_manifest = load_saved_model(selected_id)
                    st.session_state["loaded_authorship_model"] = loaded_pipeline
                    st.session_state["loaded_authorship_manifest"] = loaded_manifest
                    st.success("Saved model loaded from the app's local model store.")
                except Exception as error:
                    st.error(f"Could not load saved model: {error}")
            if st.session_state.get("loaded_authorship_model") is not None:
                prediction_text = text_a or text_b
                if prediction_text:
                    loaded_prediction = st.session_state["loaded_authorship_model"].predict([prediction_text])[0]
                    st.write(f"Loaded model closed-set prediction: **{loaded_prediction}**")
                st.caption("Only app-generated model IDs in the local artifact directory are loadable. Joblib files are executable serialized objects; keep the local artifact folder private and use files created by this app.")
