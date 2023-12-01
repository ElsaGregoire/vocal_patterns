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


st.set_page_config(
    page_title="Voxalyze",
    page_icon="🎙️",
    initial_sidebar_state="auto",
)


st.sidebar.image("voxalyze.png", use_column_width=True)

voxlyze_base_uri = "http://localhost:8000/"
voxlyze_predict_uri = voxlyze_base_uri + "predict"
sample_rate = 22050

prediction_map = {
    0: "Arpeggio",
    1: "Other",
    2: "Scale",
}


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

float_audio_array = None

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
    )

    if audio_bytes is None:
        st.info("Please record a sound")

        # with st.spinner("Uploading and processing audio..."):
        # st.audio(audio_bytes, format="audio/wav")
        # st.success("Audio uploaded successfully!✅")

    # Plot spectrogram for recorded audio
    else:
        st.audio(audio_bytes, format="audio/wav", sample_rate=sample_rate)
        st.success("Audio recognized successfully!✅")

        st.spinner("Generating the spectrogram...")
        # Convert audio_bytes to a NumPy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        # Reducing noise
        # audio_array = nr.reduce_noise(
        #     y=audio_array, sr=sample_rate, n_std_thresh_stationary=1.5, stationary=True
        # )
        float_audio_array = audio_array.astype(float)
        # float_audio_array_r, sr = librosa.load(
        # BytesIO(float_audio_array.read()), sr=sample_rate)
        display_spectrogram(float_audio_array)
        st.write(
            "If your result is a **1**, you are singuing an *Arpegio*. If it is a **2** you are singuing a *Scale*. If your result is **0** you are singuing *something different*."
        )
        st.write("### Your singuing result is ⬇️")
        float_audio_array_as_list = float_audio_array.tolist()

        # Display the spectrogram

        float_audio_array_as_list = float_audio_array.tolist()
        # Send the audio to the API

        response = requests.post(
            voxlyze_predict_uri,
            json={"float_audio_array_as_list": float_audio_array_as_list},
        )
        # Get the response from the API
        resp = response.json()
        prediction = prediction_map[resp["prediction"]]
        st.write(f"# {prediction} #")

        st.stop()


# #if options == 'Upload your file ':
#     # Second title
#     st.markdown("### Upload your audio file here ⬇️")

#     # Audio uploaded
#     uploaded_file1 = st.file_uploader("Pick a wave file!", type="wav", key="sample1")

#     if uploaded_file1 is None:
#         st.info("Please upload a wave file.")
#         st.stop()

#     with st.spinner("Checking the audio..."):
#         time.sleep(3)
#         st.success("Audio recogniced successfully!✅")
#         # Load the uploaded audio file with a specified sampling rate
#         float_audio_array, sr = librosa.load(
#             BytesIO(uploaded_file1.read()), sr=sample_rate
#         )


# with st.spinner("Generating the spectogram..."):
#     time.sleep(2)

else:
    st.markdown("### Upload your audio file here ⬇️")
    uploaded_file1 = st.file_uploader("Pick a wave file!", type="wav", key="sample1")

    if uploaded_file1 is None:
        st.info("Please upload a wave file.")
        st.stop()

    if uploaded_file1 is not None:
        st.spinner("Checking the audio...")
        st.success("Audio recogniced successfully!✅")
        float_audio_array, sr = librosa.load(
            BytesIO(uploaded_file1.read()), sr=sample_rate
        )


if float_audio_array is not None:
    # Reducing noise
    # float_audio_array = nr.reduce_noise(
    #     y=float_audio_array,
    #     sr=sample_rate,
    #     n_std_thresh_stationary=1.5,
    #     stationary=True,
    # )
    st.audio(float_audio_array, format="audio/wav", sample_rate=sample_rate)
    # Display the spectrogram
    display_spectrogram(float_audio_array)
    st.write(
        "If your result is a **1**, you are singing an *Arpeggio*. If it is a **2** you are singuing a *Scale*. If your result is **0** you are singuing *something different*."
    )

    prediction_map = {
        0: "Arpeggio",
        1: "Other",
        2: "Scale",
    }
    st.write("### Your recording result is ⬇️")
    float_audio_array_as_list = float_audio_array.tolist()
    # Send the audio to the API

    response = requests.post(
        voxlyze_predict_uri,
        json={"float_audio_array_as_list": float_audio_array_as_list},
    )
    # Get the response from the API
    resp = response.json()
    prediction = prediction_map[resp["prediction"]]
    st.write(f"# {prediction} #")
