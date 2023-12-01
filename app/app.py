import streamlit as st
import numpy as np
import time
from io import BytesIO
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import requests
import noisereduce as nr
import soundfile as sf

import librosa
import librosa.display

sample_rate = 22050


def reduce_noise(float_audio_array, sample_rate):
    return nr.reduce_noise(
        y=float_audio_array,
        sr=sample_rate,
        n_std_thresh_stationary=1.5,
        stationary=True,
    )


def display_spectrogram(audio):
    sr = 44100  # You need to define the sampling rate for audio_bytes
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max),
        y_axis="log",
        x_axis="time",
        sr=sr,
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    st.pyplot()


def get_prediction(float_audio_array_as_list):
    response = requests.post(
        voxlyze_predict_uri,
        json={"float_audio_array_as_list": float_audio_array_as_list},
    )
    return response


def show_response(resp):
    prediction = resp["response"]["prediction"]
    confidence = round(resp["response"]["confidence"])

    return f"# {prediction}: ({confidence}%)"


def response_display(float_audio_array):
    # float_audio_array = reduce_noise(float_audio_array, sample_rate)
    st.audio(float_audio_array, format="audio/wav", sample_rate=sample_rate)
    st.success("Audio recognized successfully!✅")

    display_spectrogram(float_audio_array)

    st.write("### Your recording result is ⬇️")
    float_audio_array_as_list = float_audio_array.tolist()
    resp = get_prediction(float_audio_array_as_list).json()
    st.write(show_response(resp))


st.set_page_config(
    page_title="Voxalyze",
    page_icon="🎙️",
    initial_sidebar_state="auto",
)


st.sidebar.image("voxalyze.png", use_column_width=True)

# voxlyze_base_uri = "http://localhost:8000/"
voxlyze_base_uri = "https://vocalpatterns-mqofeud75a-ew.a.run.app/"
voxlyze_predict_uri = voxlyze_base_uri + "predict"


st.title("Voxalyze")

st.write("""🎈🎈🎈 Welcome to our Vocal Pattern App 🎈🎈🎈""")

st.write(
    """Here you can record a sound 🎙️ or upload a sound file 🎵 between 4 and 6 seconds.
         Our app will show you the *spectogram* 📊 of the sound and will classify the sound as an **Arpegio**,
         a **Scale** or **Other type** of sound (as *melodies*, *long notes*, a funk and beautiful *improvisation* 🕺🏾, .."""
    ""
)

st.subheader(
    "Please, select one of the options below", divider="red"
)  # Adding a divider


# add_selectbox = st.sidebar.radio(
#     "Where would you like to go?",
#     ("App", "Knowledge"))


st.set_option("deprecation.showPyplotGlobalUse", False)

options = st.radio("Select an option", ("record", "upload"))

if options == "record":
    # First title
    st.markdown("### Record your audio here ⬇️")

    # Audio recorder
    audio_bytes = audio_recorder(
        pause_threshold=6.0,
        text="",
        recording_color="#6aa36f",
        neutral_color="565656",
        icon_name="microphone",
        icon_size="6x",
        sample_rate=sample_rate,
    )

    if audio_bytes is None:
        st.info("Please record a sound")
    else:
        st.spinner("Generating the spectrogram...")
        audio_array = np.frombuffer(audio_bytes, dtype=np.int32)
        float_audio_array = audio_array.astype(float)
        response_display(float_audio_array)
        st.stop()

else:
    st.markdown("### Upload your audio file here ⬇️")
    uploaded_file1 = st.file_uploader("Pick a wave file!", type="wav", key="sample1")

    if uploaded_file1 is None:
        st.info("Please upload a wave file.")
        st.stop()

    if uploaded_file1 is not None:
        st.spinner("Checking the audio...")
        float_audio_array, sr = librosa.load(
            BytesIO(uploaded_file1.read()), sr=sample_rate
        )
        response_display(float_audio_array)
