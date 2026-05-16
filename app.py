# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit frontend for Fake News Detector.
#
# Usage:
#   streamlit run app.py
#
# Requirements:
#   model.pkl and vectorizer.pkl must exist (run train_model.py first)
# ─────────────────────────────────────────────────────────────────────────────

import pickle
import streamlit as st
from utils.preprocess import preprocess_text

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS Styling
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        color: #67bce0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-fake {
        background-color: #b33030;
        border-left: 5px solid #e74c3c;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .result-real {
        background-color: #59a865;
        border-left: 5px solid #27ae60;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .confidence-label {
        font-size: 0.9rem;
        color: #555;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load Model and Vectorizer (cached — loads only once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load trained model and vectorizer from disk. Cached after first load."""
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


try:
    model, vectorizer = load_model()
except FileNotFoundError:
    st.error(
        "⚠️ Model files not found! Please run `python train_model.py` first "
        "to generate `model.pkl` and `vectorizer.pkl`."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Header Section
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">🔍 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Powered by TF-IDF + Logistic Regression · '
    'Trained on 44,000+ news articles</div>',
    unsafe_allow_html=True
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# How It Works Expander (educational — great for resume demos)
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("ℹ️ How does this work?"):
    st.markdown("""
    1. **Input** — You paste a news article or headline.
    2. **Preprocessing** — The text is cleaned: lowercased, stopwords removed, stemmed.
    3. **TF-IDF** — Text is converted to a numerical vector (50,000 features).
    4. **Logistic Regression** — The trained model predicts REAL or FAKE.
    5. **Confidence** — The model outputs a probability (0–100%) for its prediction.
    
    **Dataset:** Kaggle Fake and Real News Dataset (~44,000 articles)  
    **Model Accuracy:** ~98–99% on the test set
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Main Input Section
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("📰 Enter News Article")

user_input = st.text_area(
    label="Paste your news article or headline below:",
    placeholder="e.g., NASA confirms discovery of water on Mars surface...",
    height=200,
    help="Paste a full news article or just the headline for analysis."
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    detect_button = st.button("🔍 Detect News", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# Prediction Logic
# ─────────────────────────────────────────────────────────────────────────────

if detect_button:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text before clicking Detect.")
    elif len(user_input.strip().split()) < 5:
        st.warning("⚠️ Please enter at least a few sentences for accurate results.")
    else:
        with st.spinner("Analyzing article..."):

            # Step 1: Preprocess the input text
            cleaned = preprocess_text(user_input)

            # Step 2: Transform using the fitted TF-IDF vectorizer
            vectorized = vectorizer.transform([cleaned])

            # Step 3: Predict label and probability
            prediction = model.predict(vectorized)[0]          # 0 or 1
            probabilities = model.predict_proba(vectorized)[0] # [p_fake, p_real]

            fake_confidence = probabilities[0] * 100
            real_confidence = probabilities[1] * 100

        st.divider()
        st.subheader("📊 Analysis Result")

        # ── Display result based on prediction ──
        if prediction == 0:
            # FAKE NEWS
            st.markdown(f"""
                <div class="result-fake">
                    <h2>⚠️ FAKE NEWS DETECTED</h2>
                    <p>This article shows strong indicators of being <b>fabricated or misleading</b>.</p>
                    <p>Always verify with trusted sources before sharing.</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Confidence Breakdown")
            st.markdown(f"🔴 **Fake News Probability: {fake_confidence:.1f}%**")
            st.progress(int(fake_confidence))
            st.markdown(f"🟢 Real News Probability: {real_confidence:.1f}%")

        else:
            # REAL NEWS
            st.markdown(f"""
                <div class="result-real">
                    <h2>✅ LIKELY REAL NEWS</h2>
                    <p>This article appears to be <b>credible and factual</b>.</p>
                    <p>However, always cross-check with multiple sources.</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Confidence Breakdown")
            st.markdown(f"🟢 **Real News Probability: {real_confidence:.1f}%**")
            st.progress(int(real_confidence))
            st.markdown(f"🔴 Fake News Probability: {fake_confidence:.1f}%")

        # ── Additional context ──
        st.divider()
        with st.expander("🔬 See preprocessing details"):
            st.markdown("**Original text (first 300 chars):**")
            st.code(user_input[:300])
            st.markdown("**After NLP preprocessing (first 300 chars):**")
            st.code(cleaned[:300])
            st.caption(
                f"Original length: {len(user_input.split())} words → "
                f"Cleaned length: {len(cleaned.split())} tokens"
            )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.8rem'>"
    "Built with Streamlit · Scikit-learn · NLTK · "
    "Model trained on Kaggle Fake & Real News Dataset"
    "</div>",
    unsafe_allow_html=True
)