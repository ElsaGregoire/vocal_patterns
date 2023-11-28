import streamlit as st
import numpy as np
import time
from io import BytesIO
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder

import librosa
import librosa.display

st.set_option('deprecation.showPyplotGlobalUse', False)

# First title
st.markdown("### Record your audio here:")

# Audio recorder
audio_bytes = audio_recorder(energy_threshold=(-1.0, 1.0),
                             pause_threshold=10.0,
                             text="",
                             recording_color="#e8b62c",
                             neutral_color="#6aa36f",
                             icon_name="microphone",
                             icon_size="6x")

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    with st.spinner('Uploading and processing files...'):
        time.sleep(3)
    st.success('Audio uploaded successfully!✅')

# Plot spectrogram for recorded audio
if audio_bytes is not None:
    # Convert audio_bytes to a NumPy array
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    float_audio_array = audio_array.astype(float)
    sr = 44100  # You need to define the sampling rate for audio_bytes
    plt.figure(figsize=(10, 4))
    spectrogram = librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(float_audio_array)), ref=np.max),
                                           y_axis='log', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    st.pyplot()

# Second title
st.markdown("### Or upload your audio files 🎶(max duration: 10 seconds):")

# Audio uploaded
uploaded_file1 = st.file_uploader("Pick a wave file!", type='wav', key="sample1")

if uploaded_file1 is None:
    st.info("Please upload a wave file.")
    st.stop()

with st.spinner('Checking the audio...'):
    time.sleep(3)
    # Load the uploaded audio file with a specified sampling rate
    audio_data, sr = librosa.load(BytesIO(uploaded_file1.read()), sr=None)

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

#else:
    #st.warning('Please upload an audio file.')
