# Import the necessary libraries
import streamlit as st
import numpy as np
import time
from io import BytesIO
import matplotlib.pyplot as plt

import librosa
import librosa.display

#
st.set_page_config(
    page_title="Voxalyze",
    page_icon="🎙️",
    initial_sidebar_state="auto",

)

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://slack-files.com/T02NE0241-F067KRT4SGN-5673fc5b5b);
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "Voxalize";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
print(add_logo())


# Audio uploaded
uploaded_file1 = st.file_uploader("Pick a wave file!", type='wav', key="sample1")

if uploaded_file1 is None:
    st.info("Please upload a wave file.")
    st.stop()

with st.spinner('Checking the audio...'):
    time.sleep(3)
    st.success('Audio recogniced successfully!✅')
    # Load the uploaded audio file with a specified sampling rate
    audio_data, sr = librosa.load(BytesIO(uploaded_file1.read()), sr=None)

with st.spinner('Generating the spectogram...'):
    time.sleep(2)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    # Add a check for empty audio data
    if audio_data is not None:
        spectrogram = librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max),
                                               y_axis='log', x_axis='time', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        st.pyplot()

        # Basic sound type recognition
        # Replace this with your actual recognition logic
        if np.max(spectrogram) > threshold:
            sound_type = "Arpeggio"
        elif np.mean(spectrogram) > threshold:
            sound_type = "Scale"
        else:
            sound_type = "Other"

        st.markdown(f"## Sound Type Recognition: {sound_type}")

    else:
        st.warning("The audio file is empty or couldn't be loaded.")
