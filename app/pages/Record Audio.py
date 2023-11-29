import streamlit as st
import numpy as np
import time
from io import BytesIO
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder


import librosa
import librosa.display

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
        time.sleep(2)

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
