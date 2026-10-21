"""Plotly and Matplotlib figures for the Streamlit comparison workspace."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from matplotlib.figure import Figure

DOC_A_COLOR = "#3E7782"  # muted teal
DOC_B_COLOR = "#C17C5D"  # muted terracotta
TEXT_COLOR = "#34464B"
GRID_COLOR = "#DCE4E1"
PLOT_COLOR = "#F4F6F4"


def style_figure(fig: go.Figure) -> go.Figure:
    """Apply the same restrained chart palette in every app appearance."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=PLOT_COLOR,
        plot_bgcolor=PLOT_COLOR,
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=TEXT_COLOR),
        title_font=dict(size=17, color=TEXT_COLOR),
        legend=dict(bgcolor="rgba(244,246,244,0.88)", bordercolor=GRID_COLOR, borderwidth=1),
        hoverlabel=dict(bgcolor="#F8FAF9", bordercolor="#BCCBC6", font=dict(color=TEXT_COLOR, size=12)),
    )
    fig.update_xaxes(showline=True, linecolor="#BFCBC7", gridcolor=GRID_COLOR, zerolinecolor="#C7D2CE", title_font=dict(size=12))
    fig.update_yaxes(showline=True, linecolor="#BFCBC7", gridcolor=GRID_COLOR, zerolinecolor="#C7D2CE", title_font=dict(size=12))
    return fig


def document_style_map(text_a: str, text_b: str, *, chunk_words: int = 100) -> tuple[go.Figure | None, str | None]:
    """Map comparable, contiguous passages from two documents into 2D style space.

    Each point represents a non-overlapping passage. The PCA axes are exploratory
    summaries of standardized stylometric markers; they are not author scores.
    """
    from scipy.spatial import ConvexHull, QhullError
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from .preprocessing import WORD_PATTERN
    from .stylometry import FUNCTION_WORDS, extract_features, function_word_frequencies

    def chunks(text: str) -> list[str]:
        source = text or ""
        matches = list(WORD_PATTERN.finditer(source))
        spans: list[tuple[int, int, int]] = []
        for start in range(0, len(matches), chunk_words):
            end = min(start + chunk_words, len(matches))
            spans.append((start, end, end - start))
        # Fold a short final remainder into the preceding passage so its rates
        # are not dominated by a much smaller sample.
        if len(spans) > 1 and spans[-1][2] < chunk_words // 2:
            previous_start = spans[-2][0]
            spans[-2:] = [(previous_start, len(matches), len(matches) - previous_start)]
        return [source[matches[start].start():matches[end - 1].end()] for start, end, _ in spans]

    documents = [chunks(text_a), chunks(text_b)]
    if min(map(len, documents)) < 3:
        return None, f"The style map needs at least three {chunk_words}-word passages in each document. Add more text for a stable passage-level view."

    feature_names = [f"function:{category}:{word}" for category, words in FUNCTION_WORDS.items() for word in words]
    feature_names += [
        "average_sentence_length", "average_word_length", "sentence_length_stddev",
        "type_token_ratio", "hapax_rate_per_1000", "dislegomena_rate_per_1000",
    ]
    feature_names += [f"punctuation:{mark}" for mark in (".", ",", ";", ":", "!", "?", "'", '"', "(", ")", "-", "—", "…")]

    rows: list[list[float]] = []
    labels: list[str] = []
    point_numbers: list[int] = []
    word_counts: list[int] = []
    for label, passages in zip(("Document A", "Document B"), documents):
        for index, passage in enumerate(passages, start=1):
            stats = extract_features(passage)
            function_rates = function_word_frequencies(passage)
            row: list[float] = []
            for category, words in FUNCTION_WORDS.items():
                row.extend(float(function_rates[category][word]["per_1000_tokens"]) for word in words)
            row.extend(float(stats[name]) for name in feature_names[len(row):len(row) + 6])
            row.extend(float(stats["punctuation"][mark]["per_100_words"]) for mark in (".", ",", ";", ":", "!", "?", "'", '"', "(", ")", "-", "—", "…"))
            rows.append(row)
            labels.append(label)
            point_numbers.append(index)
            word_counts.append(int(stats["word_count"]))

    import numpy as np

    standardized = StandardScaler().fit_transform(np.asarray(rows, dtype=float))
    pca = PCA(n_components=2, random_state=42).fit(standardized)
    projected = pca.transform(standardized)
    explained = pca.explained_variance_ratio_ * 100
    colors = {"Document A": DOC_A_COLOR, "Document B": DOC_B_COLOR}
    symbols = {"Document A": "circle", "Document B": "triangle-up"}
    fig = go.Figure()

    # Draw the shaded group envelopes behind the passage markers.
    for label in ("Document A", "Document B"):
        indices = [index for index, value in enumerate(labels) if value == label]
        points = projected[indices]
        try:
            hull = ConvexHull(points)
            boundary = points[hull.vertices]
            boundary = np.vstack([boundary, boundary[0]])
            fig.add_trace(go.Scatter(
                x=boundary[:, 0], y=boundary[:, 1], mode="lines",
                line=dict(color=colors[label], width=1.5),
                fill="toself", fillcolor=colors[label], opacity=0.12,
                name=f"{label} range", hoverinfo="skip", showlegend=False,
            ))
        except QhullError:
            pass

    for label in ("Document A", "Document B"):
        indices = [index for index, value in enumerate(labels) if value == label]
        fig.add_trace(go.Scatter(
            x=projected[indices, 0], y=projected[indices, 1], mode="markers",
            name=label, marker=dict(color=colors[label], symbol=symbols[label], size=9, opacity=0.7, line=dict(width=0.7, color=colors[label])),
            customdata=[[point_numbers[index], word_counts[index]] for index in indices],
            hovertemplate="%{fullData.name}<br>Passage %{customdata[0]}<br>%{customdata[1]} words<extra></extra>",
        ))
    fig.update_layout(
        title="Passage-level linguistic marker map",
        xaxis_title=f"Style component 1 ({explained[0]:.1f}% variance)",
        yaxis_title=f"Style component 2 ({explained[1]:.1f}% variance)",
        height=520,
        margin=dict(t=60, r=25, b=55, l=55),
        legend=dict(title="Document"),
    )
    fig.update_xaxes(zeroline=True)
    fig.update_yaxes(zeroline=True)
    return style_figure(fig), None


def feature_comparison_chart(features_a: dict[str, Any], features_b: dict[str, Any]) -> Figure:
    """Compare sentence length, word length, and vocabulary diversity."""
    names = ["Average sentence length (words)", "Average word length (characters)", "Vocabulary diversity (TTR)"]
    keys = ["average_sentence_length", "average_word_length", "vocabulary_diversity"]
    fig = Figure(figsize=(10, 4.4), layout="constrained", facecolor=PLOT_COLOR)
    axis = fig.subplots()
    axis.set_facecolor(PLOT_COLOR)
    positions = list(range(len(names)))
    width = 0.36
    values_a = [features_a[key] for key in keys]
    values_b = [features_b[key] for key in keys]
    bars_a = axis.bar([position - width / 2 for position in positions], values_a, width, label="Document A", color=DOC_A_COLOR, edgecolor=PLOT_COLOR, linewidth=0.8)
    bars_b = axis.bar([position + width / 2 for position in positions], values_b, width, label="Document B", color=DOC_B_COLOR, edgecolor=PLOT_COLOR, linewidth=0.8)
    axis.set_xticks(positions, names, rotation=8, ha="right")
    axis.set_ylabel("Measured value")
    axis.legend(title="Document", frameon=False)
    axis.bar_label(bars_a, fmt="%.2f", padding=2, fontsize=8, color=TEXT_COLOR)
    axis.bar_label(bars_b, fmt="%.2f", padding=2, fontsize=8, color=TEXT_COLOR)
    axis.tick_params(colors=TEXT_COLOR, labelsize=9)
    axis.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["left"].set_color("#BFCBC7")
    axis.spines["bottom"].set_color("#BFCBC7")
    axis.spines[["top", "right"]].set_visible(False)
    return fig


def punctuation_comparison_chart(features_a: dict[str, Any], features_b: dict[str, Any]) -> go.Figure:
    """Show punctuation counts normalized per 100 words."""
    punctuation_a, punctuation_b = features_a["punctuation"], features_b["punctuation"]
    marks = list(punctuation_a)
    fig = go.Figure()
    for label, punctuation, color in (("Document A", punctuation_a, DOC_A_COLOR), ("Document B", punctuation_b, DOC_B_COLOR)):
        rates = [punctuation[mark]["per_100_words"] for mark in marks]
        counts = [punctuation[mark]["count"] for mark in marks]
        fig.add_trace(go.Bar(name=label, x=marks, y=rates, marker_color=color, marker_line_color=PLOT_COLOR, marker_line_width=1, customdata=counts,
                             hovertemplate="Mark: %{x}<br>Per 100 words: %{y:.2f}<br>Count: %{customdata}<extra>%{fullData.name}</extra>"))
    fig.update_layout(barmode="group", xaxis_title="Punctuation mark", yaxis_title="Occurrences per 100 words", legend_title="Document", margin=dict(t=20))
    return style_figure(fig)


def word_frequency_chart(text_a: str, text_b: str, limit: int = 10) -> go.Figure:
    """Display the union of the most common non-stop words in both documents."""
    from .stylometry import word_frequencies

    counts_a = dict(word_frequencies(text_a, limit=limit))
    counts_b = dict(word_frequencies(text_b, limit=limit))
    words = sorted(set(counts_a) | set(counts_b), key=lambda word: (-(counts_a.get(word, 0) + counts_b.get(word, 0)), word))
    fig = go.Figure()
    for label, counts, color in (("Document A", counts_a, DOC_A_COLOR), ("Document B", counts_b, DOC_B_COLOR)):
        fig.add_trace(go.Bar(name=label, x=words, y=[counts.get(word, 0) for word in words], marker_color=color, marker_line_color=PLOT_COLOR, marker_line_width=1,
                             hovertemplate="%{x}<br>Count: %{y}<extra>%{fullData.name}</extra>"))
    fig.update_layout(barmode="group", xaxis_title="Word (English stop words excluded)", yaxis_title="Token count", legend_title="Document", margin=dict(t=20, b=80))
    return style_figure(fig)


def normalized_frequency_chart(
    rates_a: dict[str, float],
    rates_b: dict[str, float],
    *,
    title: str,
    y_axis_title: str,
    descriptions: dict[str, str] | None = None,
) -> go.Figure:
    """Compare normalized feature rates with labels and optional explanations."""
    categories = sorted(set(rates_a) | set(rates_b))
    fig = go.Figure()
    for label, rates, color in (("Document A", rates_a, DOC_A_COLOR), ("Document B", rates_b, DOC_B_COLOR)):
        values = [rates.get(category, 0.0) for category in categories]
        info = [descriptions.get(category, "") if descriptions else "" for category in categories]
        fig.add_trace(go.Bar(name=label, x=categories, y=values, marker_color=color, marker_line_color=PLOT_COLOR, marker_line_width=1, customdata=info,
                             hovertemplate="%{x}<br>Rate: %{y:.3f}<br>%{customdata}<extra>%{fullData.name}</extra>"))
    fig.update_layout(title=title, barmode="group", yaxis_title=y_axis_title, xaxis_title="Feature", legend_title="Document", margin=dict(t=55, b=75))
    return style_figure(fig)
