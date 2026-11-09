import sys
from pathlib import Path
import ast
import gc

import pandas as pd
import streamlit as st


# ============================================================
# SignalLens — Customer Feedback Intelligence
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

st.set_page_config(
    page_title="SignalLens | Customer Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --sl-bg: #080e1b;
        --sl-panel: #111b2b;
        --sl-panel-soft: #0d1726;
        --sl-border: rgba(148, 163, 184, 0.16);
        --sl-text: #edf3fb;
        --sl-muted: #91a0b6;
        --sl-teal: #13cdb5;
        --sl-red: #f05269;
        --sl-amber: #e9ad43;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 0%, rgba(19, 205, 181, 0.07), transparent 28%),
            var(--sl-bg);
    }

    [data-testid="stHeader"] {
        background: rgba(8, 14, 27, 0.82);
    }

    [data-testid="stSidebar"] {
        background: #0a1220;
        border-right: 1px solid var(--sl-border);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--sl-muted);
    }

    .sl-brand {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 8px 0 18px 0;
    }

    .sl-brand-icon {
        width: 42px;
        height: 42px;
        border-radius: 13px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #06131a;
        font-size: 23px;
        font-weight: 800;
        background: linear-gradient(135deg, #24e0c8, #0ba995);
        box-shadow: 0 5px 22px rgba(19, 205, 181, 0.18);
    }

    .sl-brand-name {
        color: var(--sl-text);
        font-size: 22px;
        font-weight: 750;
        letter-spacing: -0.7px;
        line-height: 1.1;
    }

    .sl-brand-caption {
        color: var(--sl-muted);
        font-size: 10px;
        letter-spacing: 1.15px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    .sl-eyebrow {
        color: var(--sl-teal);
        text-transform: uppercase;
        letter-spacing: 1.7px;
        font-size: 10px;
        font-weight: 750;
        margin-bottom: 7px;
    }

    .sl-page-title {
        color: var(--sl-text);
        font-size: clamp(27px, 3vw, 37px);
        font-weight: 760;
        letter-spacing: -1.2px;
        line-height: 1.12;
        margin: 0 0 7px 0;
    }

    .sl-page-subtitle {
        color: var(--sl-muted);
        font-size: 14px;
        margin-bottom: 23px;
    }

    .sl-kpi {
        background: linear-gradient(145deg, #121e30, #0e1827);
        border: 1px solid var(--sl-border);
        border-radius: 15px;
        padding: 17px 18px 15px 18px;
        min-height: 118px;
        box-shadow: 0 8px 26px rgba(0, 0, 0, 0.10);
    }

    .sl-kpi-label {
        color: var(--sl-muted);
        font-size: 10px;
        font-weight: 750;
        letter-spacing: 1.0px;
        text-transform: uppercase;
    }

    .sl-kpi-value {
        color: var(--sl-text);
        font-size: 30px;
        font-weight: 760;
        letter-spacing: -0.8px;
        margin-top: 8px;
        line-height: 1.1;
    }

    .sl-kpi-help {
        color: var(--sl-muted);
        font-size: 11px;
        margin-top: 8px;
    }

    .sl-panel {
        background: linear-gradient(145deg, rgba(17, 27, 43, 0.96), rgba(13, 23, 38, 0.96));
        border: 1px solid var(--sl-border);
        border-radius: 15px;
        padding: 17px 19px;
        margin-bottom: 12px;
    }

    .sl-panel-title {
        color: var(--sl-text);
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .sl-panel-caption {
        color: var(--sl-muted);
        font-size: 11px;
        margin-bottom: 12px;
    }

    .sl-tag {
        display: inline-block;
        background: rgba(19, 205, 181, 0.10);
        color: #54e5d2;
        border: 1px solid rgba(19, 205, 181, 0.20);
        border-radius: 999px;
        padding: 5px 10px;
        margin: 3px 4px 3px 0;
        font-size: 11px;
    }

    .sl-muted {
        color: var(--sl-muted);
        font-size: 12px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #121e30, #0e1827);
        border: 1px solid var(--sl-border);
        border-radius: 13px;
        padding: 13px 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--sl-muted);
    }

    div[data-testid="stMetricValue"] {
        color: var(--sl-text);
    }

    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {
        background: linear-gradient(135deg, #18cdb6, #0ca895);
        color: #041511;
        border: 0;
        border-radius: 10px;
        font-weight: 750;
    }

    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button[kind="primary"]:hover {
        background: #36dfc9;
        color: #041511;
        border: 0;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--sl-border);
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: var(--sl-border);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Data loading
# ============================================================

@st.cache_data
def load_data():
    files = {
        "reviews": "review_level_predictions.csv",
        "aspects": "aspect_level_predictions.csv",
        "aspect_summary": "aspect_sentiment_summary.csv",
        "problematic_aspects": "problematic_aspects.csv",
        "topics": "final_topic_report.csv",
    }

    loaded = {}
    for key, filename in files.items():
        path = OUTPUT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Required output file not found: {path}")
        loaded[key] = pd.read_csv(path)

    return loaded


try:
    data = load_data()
except Exception as exc:
    st.error(f"SignalLens could not load its analytical output files: {exc}")
    st.info(
        "Check that the project's outputs folder contains the CSV files "
        "generated by the SignalLens pipeline."
    )
    st.stop()


# ============================================================
# Helpers
# ============================================================

def filter_domain(frame, selected_domain):
    if selected_domain == "All domains":
        return frame.copy()
    return frame[frame["domain"] == selected_domain].copy()


def parse_list(value):
    """Convert list-like strings stored in CSV cells into Python lists."""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError, TypeError):
        return []


def page_header(eyebrow, title, subtitle):
    st.markdown(
        f"""
        <div class="sl-eyebrow">{eyebrow}</div>
        <div class="sl-page-title">{title}</div>
        <div class="sl-page-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label, value, help_text):
    st.markdown(
        f"""
        <div class="sl-kpi">
            <div class="sl-kpi-label">{label}</div>
            <div class="sl-kpi-value">{value}</div>
            <div class="sl-kpi-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_tags(items):
    if not items:
        st.caption("No associated terms available.")
        return
    html = "".join(
        f'<span class="sl-tag">{str(item)}</span>' for item in items
    )
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# Sidebar and navigation
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sl-brand">
            <div class="sl-brand-icon">∿</div>
            <div>
                <div class="sl-brand-name">SignalLens</div>
                <div class="sl-brand-caption">Customer intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("WORKSPACE")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Aspect Intelligence",
            "Complaint Discovery",
            "Review Explorer",
        ],
        label_visibility="collapsed",
        format_func=lambda item: {
            "Overview": "◫  Overview",
            "Aspect Intelligence": "▥  Aspect Intelligence",
            "Complaint Discovery": "◈  Complaint Discovery",
            "Review Explorer": "⌕  Review Explorer",
        }[item],
    )

    st.markdown("---")
    st.caption("ANALYSIS DOMAIN")
    domain = st.selectbox(
        "Select domain",
        ["All domains", "laptop", "restaurant"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("ABOUT")
    st.markdown(
        '<div class="sl-muted">SignalLens turns unstructured customer '
        'reviews into aspect-level sentiment and recurring complaint '
        'insights.</div>',
        unsafe_allow_html=True,
    )


reviews = filter_domain(data["reviews"], domain)
aspects = filter_domain(data["aspects"], domain)
aspect_summary = filter_domain(data["aspect_summary"], domain)
problematic = filter_domain(data["problematic_aspects"], domain)
topics = filter_domain(data["topics"], domain)


# ============================================================
# PAGE 1 — Overview
# ============================================================

if page == "Overview":
    page_header(
        "CUSTOMER INTELLIGENCE PLATFORM",
        "Understand what your customers are saying.",
        "Turn customer feedback into aspect-level insights and recurring complaint patterns.",
    )

    total_reviews = len(reviews)
    total_aspects = len(aspects)
    negative_mentions = int((aspects["sentiment"] == "negative").sum())
    negative_rate = (
        negative_mentions / total_aspects * 100 if total_aspects else 0
    )
    topic_count = (
        topics[["domain", "topic_id"]].drop_duplicates().shape[0]
        if not topics.empty
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("REVIEWS ANALYSED", f"{total_reviews:,}", "Processed customer reviews")
    with c2:
        kpi_card("ASPECT MENTIONS", f"{total_aspects:,}", "Extracted aspect mentions")
    with c3:
        kpi_card("NEGATIVE SENTIMENT", f"{negative_rate:.1f}%", "Share of aspect mentions classified as negative")
    with c4:
        kpi_card("DISCOVERED TOPICS", f"{topic_count:,}", "Complaint clusters across selected domain")

    st.write("")
    left, right = st.columns([1.55, 1])

    with left:
        st.markdown(
            '<div class="sl-panel"><div class="sl-panel-title">'
            'Aspect-level sentiment distribution</div>'
            '<div class="sl-panel-caption">Predicted sentiment across extracted aspect mentions</div>',
            unsafe_allow_html=True,
        )

        sentiment_order = ["positive", "negative", "neutral", "conflict"]
        sentiment_counts = (
            aspects["sentiment"]
            .value_counts()
            .reindex(sentiment_order, fill_value=0)
        )
        if sentiment_counts.sum():
            st.bar_chart(sentiment_counts, height=270)
        else:
            st.info("No aspect-level sentiment records are available.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="sl-panel"><div class="sl-panel-title">'
            'SignalLens workflow</div>'
            '<div class="sl-panel-caption">From raw feedback to actionable patterns</div>'
            '<div class="sl-muted">01 · Aspect Term Extraction</div>'
            '<div class="sl-muted">Identify product and service attributes in reviews.</div><br>'
            '<div class="sl-muted">02 · Aspect-Based Sentiment</div>'
            '<div class="sl-muted">Classify sentiment toward each extracted aspect.</div><br>'
            '<div class="sl-muted">03 · Complaint Discovery</div>'
            '<div class="sl-muted">Group negative feedback into recurring themes.</div></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Top complaint aspects")
    if problematic.empty:
        st.info("No problematic-aspect records are available for this selection.")
    else:
        top_problematic = problematic.sort_values(
            ["negative_rate", "mentions"], ascending=[False, False]
        ).head(8)

        preview = top_problematic[
            ["aspect_normalized", "mentions", "negative", "negative_rate"]
        ].copy()
        preview["negative_rate"] = (preview["negative_rate"] * 100).round(1)
        preview = preview.rename(
            columns={
                "aspect_normalized": "Aspect",
                "mentions": "Mentions",
                "negative": "Negative mentions",
                "negative_rate": "Negative rate (%)",
            }
        )
        st.dataframe(preview, use_container_width=True, hide_index=True)

    st.subheader("Recently surfaced complaint themes")
    if topics.empty:
        st.info("No complaint topics are available for this selection.")
    else:
        topic_preview = topics.sort_values(
            "record_count", ascending=False
        ).head(5)
        for _, row in topic_preview.iterrows():
            with st.container(border=True):
                c1, c2, c3 = st.columns([2.2, 1, 1])
                c1.markdown(f"**{str(row['final_descriptor']).title()}**")
                c1.caption(f"{str(row['domain']).title()} · Topic {row['topic_id']}")
                c2.metric("Records", int(row["record_count"]))
                c3.caption(f"Quality: {row['quality_flag'].replace('_', ' ')}")

    st.caption(
        "All overview figures are calculated from the saved batch prediction "
        "outputs. They represent model predictions, not manually verified labels."
    )


# ============================================================
# PAGE 2 — Aspect Intelligence
# ============================================================

elif page == "Aspect Intelligence":
    page_header(
        "ASPECT-LEVEL ANALYTICS",
        "What are customers talking about?",
        "Compare sentiment across the specific product and service attributes mentioned in reviews.",
    )

    if aspect_summary.empty:
        st.info("No aspect summary records are available for this domain.")
    else:
        min_mentions = st.select_slider(
            "Minimum number of aspect mentions",
            options=[1, 3, 5, 10, 15],
            value=5,
        )

        filtered_aspects = aspect_summary[
            aspect_summary["mentions"] >= min_mentions
        ].copy()

        if filtered_aspects.empty:
            st.info("No aspects meet the selected mention threshold.")
        else:
            top_negative = filtered_aspects.sort_values(
                ["negative_rate", "mentions"], ascending=[False, False]
            ).head(15)

            total_mentions = int(filtered_aspects["mentions"].sum())
            negative_count = int(filtered_aspects["negative"].sum())
            positive_count = int(filtered_aspects["positive"].sum())
            negative_share = (
                negative_count / total_mentions * 100 if total_mentions else 0
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Aspects in view", f"{len(filtered_aspects):,}")
            c2.metric("Total mentions", f"{total_mentions:,}")
            c3.metric("Negative share", f"{negative_share:.1f}%")

            st.subheader("Aspects with the highest negative sentiment")
            chart_data = top_negative.set_index("aspect_normalized")[
                ["negative_rate"]
            ].sort_values("negative_rate")
            st.bar_chart(chart_data, height=430)
            st.caption(
                "Negative rate is the proportion of an aspect's mentions "
                "classified as negative. It should be interpreted alongside "
                "the number of mentions."
            )

            st.subheader("Sentiment breakdown by aspect")
            display_columns = [
                "domain",
                "aspect_normalized",
                "mentions",
                "positive",
                "negative",
                "neutral",
                "conflict",
                "negative_rate",
                "avg_confidence",
            ]
            display = filtered_aspects[display_columns].copy()
            display["negative_rate"] = (display["negative_rate"] * 100).round(1)
            display["avg_confidence"] = display["avg_confidence"].round(3)
            display = display.sort_values(
                ["negative_rate", "mentions"], ascending=[False, False]
            ).rename(
                columns={
                    "domain": "Domain",
                    "aspect_normalized": "Aspect",
                    "mentions": "Mentions",
                    "positive": "Positive",
                    "negative": "Negative",
                    "neutral": "Neutral",
                    "conflict": "Conflict",
                    "negative_rate": "Negative rate (%)",
                    "avg_confidence": "Average confidence",
                }
            )
            st.dataframe(display, use_container_width=True, hide_index=True)


# ============================================================
# PAGE 3 — Complaint Discovery
# ============================================================

elif page == "Complaint Discovery":
    page_header(
        "UNSUPERVISED COMPLAINT MINING",
        "Discover recurring customer problems.",
        "Explore complaint themes identified by semantic embeddings and unsupervised clustering.",
    )

    if topics.empty:
        st.info("No complaint topics are available for this domain.")
    else:
        topic_options = {}
        for index, row in topics.iterrows():
            label = (
                f"{str(row['domain']).title()} · "
                f"Topic {row['topic_id']} · "
                f"{str(row['final_descriptor']).title()}"
            )
            topic_options[label] = index

        selected_label = st.selectbox(
            "Choose a discovered topic",
            list(topic_options.keys()),
        )
        selected_row = topics.loc[topic_options[selected_label]]

        representative_reviews = parse_list(
            selected_row.get("representative_reviews")
        )
        top_aspects = parse_list(selected_row.get("top_aspects"))
        yake_keywords = parse_list(selected_row.get("yake_keywords"))

        st.markdown(
            f'<div class="sl-eyebrow">{str(selected_row["domain"]).upper()} · '
            f'TOPIC {selected_row["topic_id"]}</div>'
            f'<div class="sl-page-title">{str(selected_row["final_descriptor"]).title()}</div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Complaint records", int(selected_row["record_count"]))
        m2.metric("Unique reviews", int(selected_row["unique_reviews"]))
        m3.metric(
            "Unique review ratio",
            f"{float(selected_row['unique_review_ratio']):.1%}",
        )

        quality = str(selected_row["quality_flag"])
        quality_labels = {
            "strong_candidate": ("Strong candidate", "success"),
            "needs_review": ("Needs review", "warning"),
            "low_evidence": ("Low evidence", "error"),
        }
        quality_label, quality_type = quality_labels.get(
            quality, (quality.replace("_", " ").title(), "info")
        )

        q1, q2 = st.columns([1, 2])
        with q1:
            if quality_type == "success":
                st.success(f"Topic quality: {quality_label}")
            elif quality_type == "warning":
                st.warning(f"Topic quality: {quality_label}")
            elif quality_type == "error":
                st.error(f"Topic quality: {quality_label}")
            else:
                st.info(f"Topic quality: {quality_label}")

        with q2:
            st.metric(
                "Average pairwise similarity",
                f"{float(selected_row['avg_pairwise_similarity']):.3f}",
            )

        st.caption(
            "Quality flags are heuristic indicators. They are not proof "
            "of semantic coherence or independent validation."
        )

        st.divider()
        a_col, k_col = st.columns(2)

        with a_col:
            st.subheader("Associated aspects")
            display_tags(top_aspects)

        with k_col:
            st.subheader("Discovered keywords")
            display_tags(yake_keywords)

        st.subheader("Representative customer reviews")
        if representative_reviews:
            for i, review in enumerate(representative_reviews, start=1):
                with st.expander(f"Representative review {i}"):
                    st.write(review)
        else:
            st.info("No representative reviews are available for this topic.")


# ============================================================
# PAGE 4 — Review Explorer (lazy model loading)
# ============================================================

elif page == "Review Explorer":
    page_header(
        "LIVE MODEL INFERENCE",
        "Explore an individual customer review.",
        "Extract aspects and classify the sentiment expressed toward each one.",
    )

    st.markdown(
        '<div class="sl-panel"><div class="sl-panel-title">'
        'How Review Explorer works</div>'
        '<div class="sl-panel-caption">The selected domain-specific ABSA model is loaded only '
        'when you submit a review. Models are not loaded on the other dashboard pages.</div></div>',
        unsafe_allow_html=True,
    )

    inference_domain = st.selectbox(
        "Review domain",
        ["laptop", "restaurant"],
        key="inference_domain",
        format_func=lambda value: value.title(),
    )

    with st.form("review_analysis_form"):
        review_text = st.text_area(
            "Customer review",
            placeholder=(
                "Example: The battery life is excellent, but the keyboard "
                "is uncomfortable and the screen quality is disappointing."
            ),
            height=145,
        )
        analyze_button = st.form_submit_button(
            "Analyse review",
            type="primary",
            use_container_width=True,
        )

    if analyze_button:
        if not review_text.strip():
            st.warning("Please enter a review before analysing it.")
        else:
            try:
                from src.signallens.pipeline import SignalLensPipeline

                if (
                    "signal_pipeline" not in st.session_state
                    or st.session_state.get("loaded_domain") != inference_domain
                ):
                    if "signal_pipeline" in st.session_state:
                        del st.session_state["signal_pipeline"]
                        st.session_state.pop("loaded_domain", None)
                        gc.collect()

                    with st.spinner(
                        f"Loading {inference_domain} models for the first time..."
                    ):
                        st.session_state["signal_pipeline"] = SignalLensPipeline(
                            model_dir=MODEL_DIR,
                            domain=inference_domain,
                        )
                        st.session_state["loaded_domain"] = inference_domain

                pipeline = st.session_state["signal_pipeline"]

                with st.spinner("Extracting aspects and analysing sentiment..."):
                    result = pipeline.analyze(review_text)

                st.success("Review analysis completed.")
                st.markdown("#### Submitted review")
                st.markdown(
                    f'<div class="sl-panel">{review_text}</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("#### Analysis results")
                if not result["aspects"]:
                    st.info("No aspects were identified in this review.")
                else:
                    st.caption(
                        f"{len(result['aspects'])} aspect(s) identified · "
                        f"{inference_domain.title()} domain"
                    )

                    for item in result["aspects"]:
                        sentiment = item["sentiment"].lower()
                        sentiment_color = {
                            "positive": "#13cdb5",
                            "negative": "#f05269",
                            "neutral": "#91a0b6",
                            "conflict": "#e9ad43",
                        }.get(sentiment, "#91a0b6")

                        confidence_pct = float(item["confidence"]) * 100
                        with st.container(border=True):
                            left, right = st.columns([2, 1])
                            with left:
                                st.markdown(f"**{str(item['aspect']).title()}**")
                                st.markdown(
                                    f'<span style="color:{sentiment_color};'
                                    f'font-weight:750;text-transform:uppercase;'
                                    f'font-size:11px;">{sentiment}</span>',
                                    unsafe_allow_html=True,
                                )
                            with right:
                                st.metric("Confidence", f"{confidence_pct:.1f}%")

                            with st.expander("View sentiment probabilities"):
                                probability_df = pd.DataFrame(
                                    {
                                        "Sentiment": list(item["probabilities"].keys()),
                                        "Probability": [
                                            float(value) * 100
                                            for value in item["probabilities"].values()
                                        ],
                                    }
                                ).set_index("Sentiment")
                                st.bar_chart(probability_df, height=170)

            except Exception as exc:
                st.error(f"Review analysis failed: {exc}")
                st.caption(
                    "Check that the trained model folders are present inside "
                    "the project's models directory and that the local "
                    "Python environment has the required packages installed."
                )


# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#718198;font-size:10px;'
    'letter-spacing:1px;padding:5px 0 12px 0;">'
    'SIGNALLENS · CUSTOMER FEEDBACK INTELLIGENCE</div>',
    unsafe_allow_html=True,
)
