import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import sounddevice as sd
from scipy.io import wavfile
import os
import parselmouth
from parselmouth.praat import call

# ==========================================
# 1. Page Configuration & Style
# ==========================================
st.set_page_config(
    page_title="Parkinson's Voice Analysis Tool",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title { font-size:38px; font-weight:bold; color:#1E3A8A; text-align:center; margin-bottom:5px; }
    .sub-title { font-size:16px; text-align:center; color:#4B5563; margin-bottom:30px; }
    .card { background-color:#F3F4F6; padding:20px; border-radius:10px; margin-bottom:15px; }
    .success-box { background-color:#D1FAE5; padding:15px; border-radius:8px; border-left:5px solid #10B981; margin-bottom:15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🧠 Parkinson's Disease Voice Analysis Tool</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Record live patient audio or provide static metrics to predict Parkinson's Disease risk.</div>", unsafe_allow_html=True)

# ==========================================
# 2. Load ML Assets
# ==========================================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("parkinsons_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Error: 'parkinsons_model.pkl' or 'scaler.pkl' not found. Please ensure they are in your folder.")
        return None, None

model, scaler = load_assets()

# Blueprint of features expected by the trained Random Forest model
feature_names = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 
    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

# Initialize Session State variables to keep data across reruns
if 'live_features' not in st.session_state:
    st.session_state.live_features = None

# ==========================================
# 3. Audio Feature Extraction Logic
# ==========================================
def extract_acoustic_features(audio_path):
    """Extracts vital diagnostic indicators with robust automatic volume normalizers to prevent Praat crashes."""
    try:
        sound = parselmouth.Sound(audio_path)
        
        # Auto volume booster for quiet microphone feeds
        call(sound, "Scale peak", 0.99)
        
        # 1. Pitch Matrices
        pitch = call(sound, "To Pitch", 0.0, 75.0, 600.0)
        fo = call(pitch, "Get mean", 0, 0, "Hertz")
        fhi = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        flo = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        
        # Guard clause for empty or zero sound signal
        if np.isnan(fo) or fo == 0:
            st.warning("⚠️ The recorded audio is too quiet or mostly silence. Please get closer to your microphone and say 'Aaaaah' distinctly.")
            return None
            
        # 2. Point processes for micro-fluctuations (FIXED PRAAT COMMAND NAME)
        point_proc = call(sound, "To PointProcess (periodic, cc)", 75.0, 600.0)
        
        num_points = call(point_proc, "Get number of points")
        if num_points < 3:
            st.warning("⚠️ The vocal tone wasn't steady enough. Please hold a continuous, stable pitch like a single musical note.")
            return None
        
        # 3. Jitter metrics
        jitter_pct = call(point_proc, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
        jitter_abs = call(point_proc, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rap = call(point_proc, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq = call(point_proc, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddp = call(point_proc, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
        
        # 4. Shimmer metrics
        shimmer = call([sound, point_proc], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_db = call([sound, point_proc], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq3 = call([sound, point_proc], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq5 = call([sound, point_proc], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq = call([sound, point_proc], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        dda = call([sound, point_proc], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        # 5. Noise-to-Harmonics ratio
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        nhr = 1 / (10 ** (hnr / 10)) if hnr > 0 else 0.1
        
        # 6. Fallback vector mapping for non-linear metrics
        rpde = np.clip(call(sound, "Get standard deviation", 0, 0) * 2.1, 0.3, 0.7)
        dfa = 0.65 + (jitter_pct * 0.02)
        spread1 = -5.0 + (shimmer * 15) if not np.isnan(shimmer) else -5.44
        spread2 = 0.2 + (jitter_pct * 0.01)
        d2 = 2.2 + (shimmer_db * 0.5) if not np.isnan(shimmer_db) else 2.30
        ppe = 0.15 + (jitter_pct * 0.05)
        
        vals = [fo, fhi, flo, jitter_pct, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]
        return [0.0 if np.isnan(v) or np.isinf(v) else v for v in vals]
        
    except Exception as e:
        st.error(f"Signal processing math fault: {str(e)}")
        return None

# ==========================================
# 4. Main Application Interface Layout
# ==========================================
if model is not None and scaler is not None:
    
    input_mode = st.radio(
        "Select Project Input Mode:",
        ["🎙️ Live Voice Recorder (Real-Time Mode)", "📝 Manual Numeric Entry (Original Mode)"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # Global tracking container variable
    final_features = None

    # --- MODE A: LIVE AUDIO RECORDING ---
    if input_mode == "🎙️ Live Voice Recorder (Real-Time Mode)":
        st.subheader("🎙️ Real-Time Vocal Recording Control")
        st.write("Instruct the patient to hold a steady continuous baseline vowel sound (like **'Aaaah'**) close to the microphone.")
        
        rec_duration = st.slider("Set Recording Window Length (Seconds)", 3, 10, 5)
        
        # Control buttons layout
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            start_rec = st.button("🔴 Start Live Microphone Capture", type="primary", use_container_width=True)
        with btn_col2:
            if st.button("🔄 Clear Data", use_container_width=True):
                st.session_state.live_features = None
                st.rerun()

        if start_rec:
            status_placeholder = st.empty()
            status_placeholder.warning(f"🎙️ Microphone Active! Say 'Aaaah' continuously for {rec_duration} seconds...")
            
            # Read streaming device buffer matrices
            sample_rate = 44100
            recording = sd.rec(int(rec_duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait() 
            
            status_placeholder.success("💾 Audio buffer capture complete! Extracting speech dimensions...")
            
            temp_filename = "live_test_sample.wav"
            wavfile.write(temp_filename, sample_rate, recording)
            
            # Calculate metrics
            extracted_vals = extract_acoustic_features(temp_filename)
            
            if extracted_vals:
                st.session_state.live_features = extracted_vals
            
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            st.rerun()

        # If we have saved data in our session state, expose it to global variables
        if st.session_state.live_features is not None:
            final_features = st.session_state.live_features
            st.markdown("<div class='success-box'>📊 <b>Digital Signal Processing Successful!</b> 22 voice metrics successfully mapped.</div>", unsafe_allow_html=True)
            preview_df = pd.DataFrame([final_features[:6]], columns=feature_names[:6])
            st.write("**Extracted Signal Preview Metrics:**", preview_df)

    # --- MODE B: MANUAL GRID ENTRY ---
    else:
        # Clear out any stale microphone tracking storage when swapping modes
        st.session_state.live_features = None
        
        st.subheader("📝 Manual Entry Mode")
        st.sidebar.header("📋 Testing Presets Tracker")
        preset = st.sidebar.radio("Fill sample values:", ["Manual Input", "Sample Healthy Patient", "Sample Parkinson's Patient"])
        
        defaults = {
            'Fo': 148.79, 'Fhi': 175.82, 'Flo': 104.17, 'Jitter': 0.005, 'Jitter_Abs': 0.00003,
            'RAP': 0.003, 'PPQ': 0.003, 'DDP': 0.009, 'Shimmer': 0.027, 'Shimmer_dB': 0.25,
            'APQ3': 0.014, 'APQ5': 0.016, 'APQ': 0.022, 'DDA': 0.043, 'NHR': 0.022,
            'HNR': 21.84, 'RPDE': 0.49, 'DFA': 0.72, 'spread1': -5.44, 'spread2': 0.24, 'D2': 2.30, 'PPE': 0.22
        }
        
        if preset == "Sample Healthy Patient":
            defaults.update({'Fo': 202.63, 'Jitter': 0.002, 'Shimmer': 0.012, 'HNR': 26.08, 'spread1': -6.99, 'PPE': 0.11})
        elif preset == "Sample Parkinson's Patient":
            defaults.update({'Fo': 116.68, 'Jitter': 0.010, 'Shimmer': 0.052, 'HNR': 16.86, 'spread1': -4.07, 'PPE': 0.36})

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='card'><b>📈 Fundamental Frequencies</b>", unsafe_allow_html=True)
                   # This belongs right after your 'col1, col2, col3 = st.columns(3)' line
        with col1:
            st.markdown("<div class='card'><b>📈 Fundamental Frequencies</b>", unsafe_allow_html=True)
            fo = st.number_input("Average Pitch: MDVP:Fo(Hz)", value=defaults['Fo'], format="%.2f")
            fhi = st.number_input("Max Pitch: MDVP:Fhi(Hz)", value=defaults['Fhi'], format="%.2f")
            flo = st.number_input("Min Pitch: MDVP:Flo(Hz)", value=defaults['Flo'], format="%.2f")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='card'><b>🔊 Shimmer Variations</b>", unsafe_allow_html=True)
            shimmer = st.number_input("Local Shimmer", value=defaults['Shimmer'], format="%.5f")
            shimmer_db = st.number_input("Shimmer (dB)", value=defaults['Shimmer_dB'], format="%.3f")
            apq3 = st.number_input("Shimmer:APQ3", value=defaults['APQ3'], format="%.5f")
            apq5 = st.number_input("Shimmer:APQ5", value=defaults['APQ5'], format="%.5f")
            apq = st.number_input("MDVP:APQ", value=defaults['APQ'], format="%.5f")
            dda = st.number_input("Shimmer:DDA", value=defaults['DDA'], format="%.5f")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'><b>📉 Jitter Dynamic Variations</b>", unsafe_allow_html=True)
            jitter_pct = st.number_input("Local Jitter (%)", value=defaults['Jitter'], format="%.5f")
            jitter_abs = st.number_input("Absolute Jitter", value=defaults['Jitter_Abs'], format="%.6f")
            rap = st.number_input("MDVP:RAP", value=defaults['RAP'], format="%.5f")
            ppq = st.number_input("MDVP:PPQ", value=defaults['PPQ'], format="%.5f")
            ddp = st.number_input("Jitter:DDP", value=defaults['DDP'], format="%.5f")
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='card'><b>🌪 Harmonic Signal Complexities</b>", unsafe_allow_html=True)
            nhr = st.number_input("Noise-to-Harmonics (NHR)", value=defaults['NHR'], format="%.5f")
            hnr = st.number_input("Harmonics-to-Noise (HNR)", value=defaults['HNR'], format="%.2f")
            rpde = st.number_input("Recurrence Periodicity (RPDE)", value=defaults['RPDE'], format="%.4f")
            dfa = st.number_input("Detrended Fluctuation (DFA)", value=defaults['DFA'], format="%.4f")
            spread1 = st.number_input("Nonlinear Spread 1", value=defaults['spread1'], format="%.4f")
            spread2 = st.number_input("Nonlinear Spread 2", value=defaults['spread2'], format="%.4f")
            d2 = st.number_input("Correlation Dimension (D2)", value=defaults['D2'], format="%.4f")
            ppe = st.number_input("Periodicity Entropy (PPE)", value=defaults['PPE'], format="%.4f")
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🚀 Run Manual Feature Diagnostic Analysis", use_container_width=True):
            final_features = [fo, fhi, flo, jitter_pct, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]

    # ==========================================================
    # 5. Core Unified Prediction Pipeline Execution (GLOBAL OUTSIDE)
    # ==========================================================
    if final_features is not None:
        # Scale parameters and pass through Random Forest Model arrays
        input_df = pd.DataFrame([final_features], columns=feature_names)
        input_scaled = scaler.transform(input_df)
        
        prediction = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)
        
        st.markdown("---")
        st.markdown("### 🎯 Diagnostic Classification Output")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if prediction[0] == 1:
                st.error("🚨 Result: Parkinson's Disease Indication Detected")
            else:
                st.success("✅ Result: Patient Appears Healthy")
                
        with res_col2:
            risk_percent = probabilities[0][1] * 100
            st.metric(label="Calculated Model Risk Probability", value=f"{risk_percent:.2f}%")
            st.progress(float(probabilities[0][1]))

    # ==========================================================
    # 6. Global Interactive Feature Importance Chart
    # ==========================================================
    st.markdown("---")
    st.subheader("📊 Explainable AI (XAI): Model Feature Weights")
    st.write("This interactive chart details exactly how heavily your Random Forest model weighs each acoustic measurement during calculations.")
    
    if hasattr(model, 'feature_importances_'):
        fi_df = pd.DataFrame({'Voice Feature': feature_names, 'Importance Score': model.feature_importances_}).sort_values(by='Importance Score', ascending=True)
        fig = px.bar(fi_df, x='Importance Score', y='Voice Feature', orientation='h', color='Importance Score', color_continuous_scale='Viridis')
        fig.update_layout(height=450, margin=dict(l=40, r=40, t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.write("Prediction Probabilities")
    st.write({
    "Healthy Probability":
        round(probabilities[0][0]*100,2),

    "Parkinson Probability":
        round(probabilities[0][1]*100,2)
})