import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from src.processor import BicepCurlProcessor

# 1. Page Configuration (Must be the first command)
st.set_page_config(
    page_title="AI Fitness Pro",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for "App-like" Feel
st.markdown("""
    <style>
    /* Remove top padding to make it look like an app */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    /* Center the WebRTC Video */
    div[data-testid="stVerticalBlock"] > div:has(div[class*="stWebrtc"]) {
        display: flex;
        justify_content: center;
    }
    /* Hide Streamlit Footer */
    footer {visibility: hidden;}
    /* Make the start button look professional */
    button {
        height: auto;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar Controls
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Select Exercise Mode", ["Normal (Single Arm)", "Combine (Double Arm)"])

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Ensure your entire upper body is visible. Stand about 2-3 meters back.")

# 4. Main Layout
st.title("💪 AI Fitness Tracker")

# We use columns to center the content nicely on Desktop
col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    st.markdown("### Live Camera Feed")
    # 5. WebRTC Streamer
    webrtc_streamer(
        key="bicep-curl",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        video_processor_factory=lambda: BicepCurlProcessor(mode),
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, 
            "audio": False
        },
        async_processing=True,
    )

# 6. Instructions at the bottom
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <b>Instructions:</b> Select a mode from the sidebar • Click 'Start' • Perform bicep curls
</div>
""", unsafe_allow_html=True)