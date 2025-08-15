import streamlit as st
import torch
import numpy as np
import sounddevice as sd
import queue
import torchaudio.transforms as T
import time
import threading
from collections import deque

from utils.SpectrogramExtractor import SpectrogramExtractor
from model import ConvVAD, config_VAD



# ===== Load Model =====
MODEL_PATH = r"src\model\ConvVAD.pt" 
model = ConvVAD(**config_VAD)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()



# ===== Preprocessing =====
extractor = SpectrogramExtractor()

def preprocess_audio(audio, sample_rate):
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != extractor.sample_rate:
        waveform = T.Resample(sample_rate, extractor.sample_rate)(waveform)
    
    # Force exactly 1 second (16000 samples at 16kHz)
    target_length = 16000
    if waveform.shape[1] < target_length:
        # Pad with zeros if too short
        padding = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif waveform.shape[1] > target_length:
        # Crop to exactly 1 second if too long
        waveform = waveform[:, :target_length]
    
    mel_spec = extractor.mel_spectrogram(waveform)
    log_mel_spec = extractor.amplitude_to_db(mel_spec)
    
    return log_mel_spec.unsqueeze(0)

def predict_voice_activity(audio_chunk, sample_rate):
    x = preprocess_audio(audio_chunk, sample_rate)
    with torch.no_grad():
        output = model(x)
        # Convert to binary VAD: 0 = no voice, 1 = voice detected
        prediction = torch.argmax(output, dim=1).item()
        return 1 if prediction > 0 else 0

# Real-time audio processor
class RealTimeAudioProcessor:
    def __init__(self):
        self.is_recording = False
        self.audio_queue = queue.Queue(maxsize=10)  # Limit queue size
        self.current_prediction = 0
        self.predictions_history = deque(maxlen=100)  # Keep last 100 predictions
        self.processing_thread = None
        self.stream = None
    
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        if self.is_recording:
            # Non-blocking put - if queue is full, drop oldest
            try:
                self.audio_queue.put_nowait(indata.copy())
            except queue.Full:
                # Drop oldest item and add new one
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(indata.copy())
                except queue.Empty:
                    pass
    
    def process_audio_continuously(self):
        """Background thread that continuously processes audio"""
        while self.is_recording:
            try:
                # Wait for audio with timeout
                audio_chunk = self.audio_queue.get(timeout=0.5)
                
                # Make VAD prediction
                pred = predict_voice_activity(audio_chunk.flatten(), 16000)
                self.current_prediction = pred
                self.predictions_history.append(pred)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue
    
    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            try:
                # Start audio stream - exactly 1 second chunks
                self.stream = sd.InputStream(
                    samplerate=16000,
                    channels=1,
                    callback=self.audio_callback,
                    blocksize=16000,  # Exactly 1 second at 16kHz
                    latency='low'
                )
                self.stream.start()
                
                # Start processing thread
                self.processing_thread = threading.Thread(
                    target=self.process_audio_continuously,
                    daemon=True
                )
                self.processing_thread.start()
                
                return True
            except Exception as e:
                self.is_recording = False
                print(f"Error starting recording: {e}")
                return False
        return True
    
    def stop_recording(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)

# Initialize processor
if "processor" not in st.session_state:
    st.session_state.processor = RealTimeAudioProcessor()

# UI Setup
st.set_page_config(page_title="Real-Time Voice Activity Detection", layout="wide")
st.title("🎙️ Real-Time Voice Activity Detection")

# Control buttons
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("▶️ Start", type="primary", use_container_width=True):
        if st.session_state.processor.start_recording():
            st.session_state.is_recording = True
            st.success("Recording started!")
        else:
            st.error("Failed to start recording!")

with col2:
    if st.button("⏹️ Stop", use_container_width=True):
        st.session_state.processor.stop_recording()
        st.session_state.is_recording = False
        st.info("Recording stopped!")

with col3:
    # Status indicator
    if hasattr(st.session_state, 'is_recording') and st.session_state.is_recording:
        st.success("🔴 LIVE - Detecting voice activity...")
    else:
        st.info("⚪ Stopped")

# Main display area
col_main, col_side = st.columns([3, 1])

with col_main:
    # Large VAD status display
    current_vad = st.session_state.processor.current_prediction
    vad_status = "VOICE DETECTED" if current_vad == 1 else "NO VOICE"
    vad_color = "#28a745" if current_vad == 1 else "#6c757d"
    vad_emoji = "🗣️" if current_vad == 1 else "🔇"
    
    st.markdown(f"""
    <div style="
        text-align: center; 
        padding: 2rem; 
        background: linear-gradient(135deg, {vad_color} 0%, {'#20c997' if current_vad == 1 else '#495057'} 100%);
        border-radius: 15px;
        margin: 1rem 0;
    ">
        <h1 style="color: white; font-size: 3rem; margin: 0;">
            {vad_emoji}
        </h1>
        <h2 style="color: white; font-size: 2.5rem; margin: 0.5rem 0;">
            {current_vad}
        </h2>
        <h3 style="color: white; margin: 0; opacity: 0.9;">
            {vad_status}
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Real-time line chart
    if len(st.session_state.processor.predictions_history) > 0:
        st.subheader("📈 Voice Activity Over Time")
        
        # Convert deque to list for plotting
        history_list = list(st.session_state.processor.predictions_history)
        
        # Create chart data
        chart_data = []
        for i, pred in enumerate(history_list[-50:]):  # Show last 50 points
            chart_data.append(pred)
        
        st.line_chart(chart_data, height=300)

with col_side:
    st.subheader("📊 Statistics")
    
    if len(st.session_state.processor.predictions_history) > 0:
        history = list(st.session_state.processor.predictions_history)
        
        # Calculate stats
        total_predictions = len(history)
        recent_predictions = history[-10:] if len(history) >= 10 else history
        
        st.metric("Total Predictions", total_predictions)
        st.metric("Current VAD", "Speaker(s) Detected" if current_vad == 1 else "No Speaker Detected")
        
        if len(recent_predictions) > 0:
            # Voice activity percentage
            voice_count = sum(recent_predictions)
            voice_percentage = (voice_count / len(recent_predictions)) * 100
            st.metric("Voice Activity %", f"{voice_percentage:.1f}%")
            
            # Show recent distribution
            st.write("**Recent Activity:**")
            silence_count = recent_predictions.count(0)
            voice_count = recent_predictions.count(1)
            silence_pct = (silence_count / len(recent_predictions)) * 100
            voice_pct = (voice_count / len(recent_predictions)) * 100
            
            st.write(f"🔇 Silence: {silence_count} ({silence_pct:.0f}%)")
            st.write(f"🗣️ Voice: {voice_count} ({voice_pct:.0f}%)")
    else:
        st.write("No predictions yet...")
        st.write("Click 'Start' to begin!")

# Auto-refresh when recording
if hasattr(st.session_state, 'is_recording') and st.session_state.is_recording:
    time.sleep(0.1)
    st.rerun()

# Cleanup warning
if st.session_state.processor.is_recording:
    st.sidebar.warning("⚠️ Recording active - close browser tab safely or click Stop first")