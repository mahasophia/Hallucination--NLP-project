"""
Web UI for the hallucination detector.

Run:  streamlit run app.py
Needs: pip install streamlit   (optional: transformers torch for the NLI scorer)
"""

import html

import streamlit as st

from hallucination_detector import NLIScorer, OverlapScorer, SentenceResult, split_sentences

st.set_page_config(page_title="Hallucination Detector", page_icon="🔍", layout="wide")
st.title("🔍 LLM Hallucination Detector")
st.caption("Sentences that disagree with independently sampled answers are more likely to be hallucinated.")


@st.cache_resource(show_spinner="Loading NLI model (first run downloads it)...")
def load_nli():
    return NLIScorer()


def get_scorer(name):
    if name == "NLI (more accurate)":
        try:
            return load_nli()
        except Exception as e:
            st.warning(f"NLI unavailable ({e}). Using overlap scorer instead.")
    return OverlapScorer()


def render(sentences, threshold):
    """Colour each sentence green -> red by score."""
    parts = []
    for s in sentences:
        hue = int(120 * (1 - min(s.score, 1.0)))  # 120 = green, 0 = red
        parts.append(
            f'<span title="score {s.score:.2f}" style="background:hsla({hue},70%,50%,0.25);'
            f'padding:2px 4px;border-radius:4px;margin-right:2px;'
            f'{"border-bottom:2px solid crimson;" if s.score >= threshold else ""}">'
            f"{html.escape(s.text)}</span>"
        )
    st.markdown(" ".join(parts), unsafe_allow_html=True)


# ------------------------------- sidebar ---------------------------------- #
with st.sidebar:
    st.header("Settings")
    scorer_name = st.radio("Scorer", ["Overlap (fast)", "NLI (more accurate)"])
    threshold = st.slider("Flag threshold", 0.0, 1.0, 0.5, 0.05)

# ------------------------------- main area -------------------------------- #
results = None

col1, col2 = st.columns(2)
answer = col1.text_area("Answer to check", height=260, placeholder="Paste the model's answer here...")
samples_raw = col2.text_area(
    "Reference samples (separate each with a line containing ---)",
    height=260,
    placeholder="First alternative answer\n---\nSecond alternative answer\n---\nThird...",
)
if st.button("Check for hallucinations", type="primary"):
    samples = [s.strip() for s in samples_raw.split("\n---\n") if s.strip()]
    if not answer.strip() or not samples:
        st.error("Provide an answer and at least one sample.")
    else:
        scorer = get_scorer(scorer_name)
        sents = []
        for t in split_sentences(answer):
            sc = sum(scorer.score(t, s) for s in samples) / len(samples)
            sents.append(SentenceResult(t, sc, sc >= threshold))
        results = (answer, samples, sents)

# ------------------------------- results ---------------------------------- #
if results:
    answer, samples, sents = results
    overall = sum(s.score for s in sents) / len(sents) if sents else 0.0
    flagged = [s for s in sents if s.score >= threshold]

    m1, m2, m3 = st.columns(3)
    m1.metric("Overall score", f"{overall:.2f}", help="0 = consistent, 1 = likely hallucinated")
    m2.metric("Flagged sentences", f"{len(flagged)} / {len(sents)}")
    m3.metric("Samples used", len(samples))

    st.subheader("Result")
    render(sents, threshold)

    st.subheader("Sentence scores")
    st.dataframe(
        [{"Sentence": s.text, "Score": round(s.score, 2), "Status": "⚠️ Flagged" if s.score >= threshold else "✅ OK"} for s in sents],
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("View samples"):
        for i, s in enumerate(samples, 1):
            st.markdown(f"**Sample {i}**\n\n{s}")