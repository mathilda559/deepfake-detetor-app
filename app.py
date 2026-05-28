import streamlit as st
import tempfile
from video_utils import extract_frames
from model import predict_video

# UI CONFIG
st.set_page_config(page_title="AI Video Detector", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #0B1F3A;
    }
    .main {
        background-color: #0B1F3A;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎥 AI Video Detection System")
st.subheader("Detect AI-generated vs Real Videos using Ensemble Deep Learning")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.video(uploaded_file)

    with st.spinner("Analyzing video with AI models..."):
        frames = extract_frames(tfile.name)
        result = predict_video(frames)

    st.success("Analysis Complete!")

    col1, col2, col3 = st.columns(3)

    col1.metric("AI Probability", f"{result['ai_score']}%")
    col2.metric("Real Probability", f"{result['real_score']}%")
    col3.metric("Result", result["prediction"])

    st.progress(result["ai_score"] / 100)

    st.info("Model uses: Face Embeddings + Frequency Analysis + Temporal Consistency Ensemble")
