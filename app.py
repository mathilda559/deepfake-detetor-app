import streamlit as st
import tempfile
import os

from video_utils import extract_frames
from model import predict_video

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Video Detector PRO",
    page_icon="🎥",
    layout="wide"
)

# Session States initialisieren
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
if "result_data" not in st.session_state:
    st.session_state["result_data"] = None

# --------------------------------------------------
# STYLING (Dark Blue Premium Design)
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #081426;
}
h1, h2, h3, h4, p, label {
    color: white !important;
}
[data-testid="stMetric"] {
    background: #10243f;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #1d4ed8;
}
.stButton button {
    background: #1d4ed8;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
    padding: 10px 24px;
    transition: all 0.3s;
}
.stButton button:hover {
    background: #2563eb;
    transform: scale(1.02);
}
.result-box {
    text-align: center;
    padding: 30px;
    border-radius: 20px;
    margin-top: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.ai-box {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 2px solid #ef4444;
}
.real-box {
    background: linear-gradient(135deg, #14532d, #166534);
    border: 2px solid #22c55e;
}
.big-result {
    font-size: 38px;
    font-weight: bold;
    margin-bottom: 10px;
}
.big-score {
    font-size: 24px;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🎥 AI Video Detector (Deepfake Focus)")
st.write(
    "Dieses Tool analysiert Videos auf mikroskopische Bildartefakte, Fourier-Frequenzmuster "
    "und unnatürliche Gesichts-Inkonsistenzen, um KI-Generierung zu entlarven."
)

# --------------------------------------------------
# VIDEO UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Video hochladen",
    type=["mp4", "mov", "avi", "mkv"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    # Layout: Video links, Analyse rechts
    col_vid, col_actions = st.columns([1, 1])
    
    with col_vid:
        st.subheader("🎥 Video-Vorschau")
        st.video(video_path)
        
    with col_actions:
        st.subheader("🔍 Analyse-Steuerung")
        st.write("Klicke auf den Button, um die mathematische Bildforensik zu starten.")
        
        if st.button("🚀 Video forensisch analysieren"):
            with st.spinner("Extrahiere Frames & isoliere Gesichtsebenen..."):
                frames = extract_frames(video_path)
                
            if not frames:
                st.error("Das Video konnte nicht gelesen werden.")
            else:
                with st.spinner("Berechne mathematische Frequenzspektren..."):
                    result = predict_video(frames)
                    st.session_state["result_data"] = result
                    st.session_state["analysis_done"] = True
                    
                    if "corrected_result" in st.session_state:
                        del st.session_state["corrected_result"]

    # Anzeige der Ergebnisse nach Abschluss
    if st.session_state["analysis_done"] and st.session_state["result_data"]:
        res = st.session_state["result_data"]
        ai_score = res["ai_score"]
        real_score = res["real_score"]
        prediction = res["prediction"]
        
        st.divider()
        st.subheader("📊 Analyseergebnis")
        
        if prediction == "AI GENERATED":
            st.markdown(f"""
            <div class="result-box ai-box">
                <div class="big-result">🚨 HÖCHSTWAHRSCHEINLICH KI-GENERIERT</div>
                <div class="big-score">Anomalie-Score: {ai_score:.2f} %</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box real-box">
                <div class="big-result">✅ WAHRSCHEINLICH ECHTES VIDEO</div>
                <div class="big-score">Authentizitäts-Score: {real_score:.2f} %</div>
            </div>
            """, unsafe_allow_html=True)
            
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("KI-Wahrscheinlichkeit", f"{ai_score:.2f}%")
        with col_m2:
            st.metric("Authentizitäts-Wahrscheinlichkeit", f"{real_score:.2f}%")
            
        # ------------------------------------------
        # FEEDBACK / KORREKTUR
        # ------------------------------------------
        st.divider()
        st.subheader("📝 War die Analyse korrekt?")
        
        actual_result = st.radio(
            "Wähle das tatsächliche Ergebnis:",
            ["KI-generiert", "Echt"],
            key="feedback_radio"
        )
        
        if st.button("Ergebnis manuell korrigieren"):
            st.session_state["corrected_result"] = actual_result
            st.success(f"Ergebnis wurde korrigiert zu: {actual_result}")
            
        if "corrected_result" in st.session_state:
            corrected = st.session_state["corrected_result"]
            st.markdown("### 🔄 Überschriebener System-Status")
            if corrected == "KI-generiert":
                st.markdown('<div class="result-box ai-box"><div class="big-result">🚨 MANUELL ALS KI GEKENNZEICHNET</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box real-box"><div class="big-result">✅ MANUELL ALS ECHT GEKENNZEICHNET</div></div>', unsafe_allow_html=True)

    try:
        if os.path.exists(video_path):
            os.unlink(video_path)
    except:
        pass
