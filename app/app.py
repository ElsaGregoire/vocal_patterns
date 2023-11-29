import streamlit as st
import numpy as np
import time
from io import BytesIO
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import requests

import librosa
import librosa.display

st.set_page_config(
    page_title="Vocal Pattern App",
    page_icon="🎙️",
    initial_sidebar_state="auto",

)

voxlyze_base_uri = 'http://localhost:8000/'
response = requests.get(voxlyze_base_uri)

resp = response.json()

print(resp)

st.write(resp)

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: 'https://slack-files.com/T02NE0241-F067KRT4SGN-5673fc5b5b';
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


st.title('Vocal Pattern App')

st.write('''🎈🎈🎈 Welcome to Voxalyze 🎈🎈🎈''')
st.write('''Here you can record a sound 🎙️ or upload a sound file 🎵 of maximum 6 seconds.
         Our app will show you the *spectogram* 📊 of the sound and will classify the sound as an **Arpegio**,
         a **Scale** or **Other type** of sound (as *melodies*, *long notes*, a funk and beautiful *improvisation* 🕺🏾, ...)''')

st.subheader('Please, select one of the options below', divider='red') #Adding a divider


st.set_option('deprecation.showPyplotGlobalUse', False)

if st.checkbox(' 🎙️ **Record your own sound** 🎙️'):

# First title
    st.markdown("### Record your audio here ⬇️")

# Audio recorder
    audio_bytes = audio_recorder(energy_threshold=(-1.0, 1.0),
                             pause_threshold=6.0,
                             text="",
                             recording_color="#6aa36f",
                             neutral_color="565656",
                             icon_name="microphone",
                             icon_size="6x")


    if audio_bytes is None:
        st.info("Please record a sound")
        st.stop()

        with st.spinner('Uploading and processing audio...'):
            st.audio(audio_bytes, format="audio/wav")
            time.sleep(2)
            st.success('Audio uploaded successfully!✅')

# Plot spectrogram for recorded audio
    if audio_bytes is not None:
        with st.spinner('Uploading and processing audio...'):
            st.audio(audio_bytes, format="audio/wav")
            time.sleep(2)
            st.success('Audio recogniced successfully!✅')
        with st.spinner('Generating the spectogram...'):
            time.sleep(4)

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

if st.checkbox(' 🎵 **Upload a sound file** 🎵'):

# Second title
    st.markdown("### Upload your audio file here ⬇️")

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

#else:
    #st.warning('Please upload an audio file.')
