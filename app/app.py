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
    result = (f"# {prediction} ({confidence}%)")
    if confidence >= 70:
        st.balloons()
    else:
        st.snow()
    return result


def response_display(float_audio_array):
    # float_audio_array = reduce_noise(float_audio_array, sample_rate)
    st.audio(float_audio_array, format="audio/wav", sample_rate=sample_rate)
    progress_text = "Generating Spectrogram. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

    st.success("Audio recognized successfully! ✅")

    display_spectrogram(float_audio_array)


    st.write("### Your recording result is 🥁")
    float_audio_array_as_list = float_audio_array.tolist()
    resp = get_prediction(float_audio_array_as_list).json()
    st.write(show_response(resp))


st.set_page_config(
    page_title="Voxalyze",
    page_icon="🎙️",
    initial_sidebar_state="collapsed"
)

col1, col2, col3 = st.columns([3,5,3])
with col1:
    st.write()
with col2:
    st.image("voxalyze.png", width= 300)
    st.write("[![Stars](https://img.shields.io/github/stars/ElsaGregoire/vocal_patterns.svg?logo=github&style=social)](https://github.com/ElsaGregoire/vocal_patterns)")
with col3:
    st.write("")



st.sidebar.image("voxalyze.png", use_column_width=True)

voxlyze_base_uri = "http://localhost:8000/"
#voxlyze_base_uri = "https://vocalpatterns-mqofeud75a-ew.a.run.app/"
voxlyze_predict_uri = voxlyze_base_uri + "predict"


col1, col2, col3 = st.columns([3,2,3])
with col1:
    st.write()
with col2:
    st.title("Voxalyze")
with col3:
    st.write("")

col1, col2, col3 = st.columns([0.2,0.85,0.2])
with col1:
    st.write()
with col2:
    st.write(" ### 🎈 **Welcome to our Vocal Pattern App** 🎈 ")
with col3:
    st.write("")



st.write(
    """Here you can record a sound 🎙️ or upload a sound file 🎵 between 4 and 6 seconds.
         Our app will show you the *spectrogram* 📊 of the sound and will classify the sound as an **Arpeggio**,
         a **Scale** or **Other type** of sound (as *melodies*, *long notes*, a funk and beautiful *improvisation* 🕺🏾 ...)"""
    ""
)


st.subheader(
    "Please, select one of the options below", divider="red"
)  # Adding a divider


# add_selectbox = st.sidebar.radio(
#     "Where would you like to go?",
#     ("App", "Knowledge"))


st.set_option("deprecation.showPyplotGlobalUse", False)

options = st.radio("What do you want to do? ", ("Record  🎙️", "Upload a file 🎵"),
                   captions= ['Warm up and sing your best arpeggios and scales', 'Share your beautiful recorded voice files'])

if options == "Record  🎙️":
    # First title
    st.markdown("### Record your audio here  ⬇️",
                help="""Press the microphone icon to stat recordig.
                When the icon turns green means it is recording.
                To stop it, press it again.
                It will automatically stop if your record is too long.""")

    col1, col2, col3 = st.columns([5,3,5])
    with col1:
        st.write()
    with col2:
        audio_bytes = audio_recorder(
        pause_threshold=6.0,
        text="",
        recording_color="#6aa36f",
        neutral_color="565656",
        icon_name="microphone",
        icon_size="8x",
        sample_rate=sample_rate)
    with col3:
        button = st.button('Try again', type='primary', use_container_width=True)

    # Audio recorder
    if audio_bytes is None:
        st.info("Please record a sound")
    else:
        if button == False:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int32)
            float_audio_array = audio_array.astype(float)
            st.download_button('Dowload your recording', data=audio_bytes, use_container_width=True)
            response_display(float_audio_array)
            st.stop()
        else:
            if button:
                st.info("Please record a new sound")


else:
    st.markdown("### Upload your audio file here ⬇️")
    uploaded_file1 = st.file_uploader("Pick a wave file!", type="wav", key="sample1", help= 'Upload your wave audio file. It should last between 4 and 6 seconds.')

    if uploaded_file1 is None:
        st.info("Please upload a wave file.")
        st.stop()

    if uploaded_file1 is not None:
        if st.button('Upload a new file', use_container_width=True, type='primary') == False:
            st.spinner("Checking the audio...")
            float_audio_array, sr = librosa.load(
                BytesIO(uploaded_file1.read()), sr=sample_rate
            )
            response_display(float_audio_array)
        else:
            st.info("Please upload a new wave file")
