import streamlit as st
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="NewsGuard AI",
    page_icon="📰",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1023 0%, #111827 18%, #1d4ed8 42%, #7c3aed 72%, #ec4899 100%);
        color: #f8fafc;
    }

    .main .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .title {
        text-align: center;
        font-size: 48px;
        font-weight: 900;
        margin-bottom: 8px;
        color: #ffffff;
        letter-spacing: 1px;
        text-shadow: 0 6px 22px rgba(255, 255, 255, 0.25);
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #e2e8f0;
        margin-bottom: 28px;
        padding: 12px 20px;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.38);
        border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.18);
    }

    .hero-panel {
        padding: 24px 24px 18px 24px;
        border-radius: 22px;
        background: rgba(15, 23, 42, 0.38);
        border: 1px solid rgba(255, 255, 255, 0.14);
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.24);
        margin-bottom: 22px;
        backdrop-filter: blur(10px);
    }

    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.9), rgba(168, 85, 247, 0.9));
        color: white;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        padding: 8px 12px;
        border-radius: 999px;
        margin-bottom: 12px;
        box-shadow: 0 8px 20px rgba(96, 165, 250, 0.38);
    }

    .hero-panel h3 {
        margin: 0 0 10px 0;
        font-size: 1.8rem;
        color: #fff !important;
    }

    .hero-panel p {
        margin: 0;
        color: #dbeafe;
        line-height: 1.6;
    }

    .card {
        padding: 25px;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        margin-bottom: 25px;
        background: rgba(15, 23, 42, 0.42);
        box-shadow: 0 15px 35px rgba(15, 23, 42, 0.18);
    }

    .result {
        padding: 22px 18px;
        border-radius: 18px;
        text-align: center;
        font-size: 30px;
        font-weight: 800;
        margin-top: 25px;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(244, 63, 94, 0.25));
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: #fff;
        box-shadow: 0 10px 20px rgba(15, 23, 42, 0.2);
    }

    .info-box {
        padding: 20px 22px;
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.12);
        margin-top: 20px;
        color: #e2e8f0;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.2);
    }

    .stButton > button {
        background: linear-gradient(135deg, #22c55e 0%, #10b981 35%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 14px;
        font-weight: 800;
        font-size: 1rem;
        padding: 0.8rem 1.5rem;
        box-shadow: 0 12px 28px rgba(59, 130, 246, 0.38);
        transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
        animation: pulseGlow 2.4s infinite ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 16px 32px rgba(59, 130, 246, 0.48);
        filter: brightness(1.05);
    }

    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 10px 26px rgba(59, 130, 246, 0.35); }
        50% { box-shadow: 0 16px 32px rgba(168, 85, 247, 0.42); }
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(15, 23, 42, 0.52);
        color: #f8fafc;
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 14px;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
    }

    .stTextInput label,
    .stTextArea label,
    h3,
    .stMarkdown {
        color: #f8fafc !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    [data-testid="stMetricDelta"] {
        color: #bfdbfe !important;
    }

    .stMetric {
        background: rgba(15, 23, 42, 0.42);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
    }

    .stAlert {
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.12);
    }

    .stCaption {
        color: #dbeafe !important;
    }

    hr {
        border: 1px solid rgba(255, 255, 255, 0.16);
    }

    .bottom-section {
        margin-top: 28px;
        padding: 24px 22px;
        border-radius: 20px;
        background: rgba(15, 23, 42, 0.42);
        border: 1px solid rgba(255, 255, 255, 0.14);
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.22);
    }

    .bottom-section h4 {
        margin: 0 0 14px 0;
        font-size: 1.5rem;
        color: #ffffff !important;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(200px, 1fr));
        gap: 16px;
        margin-top: 16px;
    }

    .feature-box {
        padding: 18px 16px;
        border-radius: 16px;
        background: rgba(30, 41, 59, 0.52);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #dbeafe;
        min-height: 110px;
    }

    .feature-box strong {
        display: block;
        color: #ffffff;
        margin-bottom: 8px;
        font-size: 1.05rem;
    }

    .footer-note {
        margin-top: 20px;
        text-align: center;
        color: #dbeafe;
        font-size: 0.9rem;
        letter-spacing: 0.08em;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="title">�️ NewsGuard </div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Fake News Detection using NLP & Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-panel">
        <div class="feature-badge">Live AI Analysis</div>
        <h3>Check whether a headline or article is likely fake or real</h3>
        <p>
            This intelligent detector uses TF-IDF text analysis and a trained machine learning model
            to flag suspicious content with a confidence score.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# News input
# -----------------------------
st.markdown("### 📝 Enter News Information")

title = st.text_input(
    "News Headline",
    placeholder="Enter the news headline..."
)

text = st.text_area(
    "News Article",
    placeholder="Paste the complete news article here...",
    height=250
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 CHECK NEWS", use_container_width=True):

    if title.strip() == "" and text.strip() == "":
        st.warning("Please enter a news headline or article.")

    else:

        news = title + " " + text

        # Convert text using the trained TF-IDF vectorizer
        news_tfidf = tfidf.transform([news])

        # Prediction
        prediction = model.predict(news_tfidf)[0]

        # Model probability
        probability = model.predict_proba(news_tfidf)[0]

        confidence = max(probability) * 100

        # -----------------------------
        # Display result
        # -----------------------------
        if prediction == 0:

            st.error(
                "⚠️ FAKE NEWS",
                icon="🚨"
            )

            st.markdown(
                f'<div class="result">⚠️ FAKE NEWS</div>',
                unsafe_allow_html=True
            )

        else:

            st.success(
                "✅ REAL NEWS",
                icon="✅"
            )

            st.markdown(
                f'<div class="result">✅ REAL NEWS</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            f"""
            <div class="info-box">
            <b>Model confidence:</b> {confidence:.2f}%<br><br>
            <small>
            This prediction is generated by a machine-learning model
            and should not be treated as definitive proof of whether
            a news article is factually true.
            </small>
            </div>
            """,
            unsafe_allow_html=True
        )


st.caption(
    "TruthCheck • Fake News Detection Project • NLP + Machine Learning"
)

st.markdown(
    """
    <div class="bottom-section">
        <h4>Why this tool stands out</h4>
        <div class="feature-grid">
            <div class="feature-box">
                <strong>Fast Detection</strong>
                Instantly analyzes text and gives a prediction with confidence for quick decision-making.
            </div>
            <div class="feature-box">
                <strong>Smart AI Model</strong>
                Uses TF-IDF and trained machine learning to identify misleading patterns in news content.
            </div>
            <div class="feature-box">
                <strong>Reliable Insights</strong>
                Designed to help readers assess suspicious stories with a modern, user-friendly interface.
            </div>
        </div>
    </div>
    <div class="footer-note">NewsGuard AI • Built for smarter, safer information checking</div>
    """,
    unsafe_allow_html=True,
)
