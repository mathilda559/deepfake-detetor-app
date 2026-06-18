import streamlit as st
import tempfile
import pandas as pd
import os

from video_utils import extract_frames
from model import predict_video

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Video Detector",
    page_icon="🎥",
    layout="wide"
)

# --------------------------------------------------
# DARK BLUE DESIGN
# --------------------------------------------------

st.markdown("""
<style>

.stApp{
    background-color:#081426;
}

h1,h2,h3,h4,p,label{
    color:white !important;
}

[data-testid="stMetric"]{
    background:#10243f;
    padding:15px;
    border-radius:12px;
}

.stButton button{
    background:#1d4ed8;
    color:white;
    border-radius:10px;
    border:none;
    font-weight:bold;
}

.result-box{
    text-align:center;
    padding:30px;
    border-radius:20px;
    margin-top:20px;
    margin-bottom:20px;
}

.ai-box{
    background:#7f1d1d;
}

.real-box{
    background:#14532d;
}

.big-result{
    font-size:40px;
    font-weight:bold;
}

.big-score{
    font-size:26px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🎥 AI Video Detector")

st.write(
    "Lade ein Video hoch und analysiere, ob es "
    "wahrscheinlich KI-generiert oder echt ist."
)

# --------------------------------------------------
# VIDEO UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Video hochladen",
    type=["mp4", "mov", "avi", "mkv"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button("🔍 Video analysieren"):

        with st.spinner("Video wird analysiert..."):

            frames = extract_frames(video_path)

            result = predict_video(frames)

            ai_score = result["ai_score"]
            real_score = result["real_score"]
            prediction = result["prediction"]

        # ------------------------------------------
        # GROSSE ERGEBNISANZEIGE
        # ------------------------------------------

        if prediction == "AI GENERATED":

            st.markdown(f"""
            <div class="result-box ai-box">
                <div class="big-result">
                🚨 KI-GENERIERT
                </div>

                <div class="big-score">
                Wahrscheinlichkeit: {ai_score:.2f} %
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-box real-box">
                <div class="big-result">
                ✅ ECHTES VIDEO
                </div>

                <div class="big-score">
                Wahrscheinlichkeit: {real_score:.2f} %
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ------------------------------------------
        # DETAILWERTE
        # ------------------------------------------

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "KI Wahrscheinlichkeit",
                f"{ai_score:.2f}%"
            )

        with col2:
            st.metric(
                "Echt Wahrscheinlichkeit",
                f"{real_score:.2f}%"
            )

        st.divider()

        # ------------------------------------------
        # FEEDBACK SYSTEM
        # ------------------------------------------

        # ------------------------------------------
# FEEDBACK / KORREKTUR
# ------------------------------------------

st.divider()

st.subheader("📝 War die Analyse korrekt?")

actual_result = st.radio(
    "Tatsächliches Ergebnis",
    [
        "KI-generiert",
        "Echt"
    ]
)

if st.button("Analyse korrigieren"):

    st.session_state["corrected_result"] = actual_result

    st.success(
        f"Ergebnis wurde korrigiert zu: {actual_result}"
    )

# ------------------------------------------
# KORRIGIERTES ERGEBNIS ANZEIGEN
# ------------------------------------------

if "corrected_result" in st.session_state:

    corrected = st.session_state["corrected_result"]

    st.divider()

    st.markdown("## 🔄 Korrigiertes Ergebnis")

    if corrected == "KI-generiert":

        st.markdown(
            """
            <div style="
                background:#7f1d1d;
                padding:30px;
                border-radius:20px;
                text-align:center;
            ">
                <h1 style="color:white;">
                    🚨 KI-GENERIERT
                </h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div style="
                background:#14532d;
                padding:30px;
                border-radius:20px;
                text-align:center;
            ">
                <h1 style="color:white;">
                    ✅ ECHTES VIDEO
                </h1>
            </div>
            """,
            unsafe_allow_html=True
        )
