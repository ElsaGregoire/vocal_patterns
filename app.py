import streamlit as st
import matplotlib as plt

import numpy as np
import pandas as pd
import time

import librosa, librosa.display
from audio_recorder_streamlit import audio_recorder


st.markdown("### Record your audio here:")

audio_bytes = audio_recorder(energy_threshold=(-1.0, 1.0),
  pause_threshold=10.0, text="",
    recording_color="#e8b62c",
    neutral_color="#6aa36f",
    icon_name="microphone",
    icon_size="6x")

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")


st.markdown("### Or upload your audio files 🎶(max duration: 10 seconds):")

upload_files = st.file_uploader('', type = ['wav', 'mp3', 'mp4'], accept_multiple_files = True)

if upload_files:
    with st.spinner('Uploading and processing files...'):
        time.sleep(3)
    st.success('Files uploaded successfully!✅')
    for file in upload_files:
        # Process each file as needed
        st.audio(file, format='audio/wav')
    for class_name, files in organized_files.items():
        st.write(f"Class: {class_name}")

    with st.spinner('Checking the audio...'):
        time.sleep(3)
else:
    st.warning('Please upload an audio file.')
