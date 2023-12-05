# Importing the needed packages
import streamlit as st
import numpy as np
import time
from io import BytesIO
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import os
import base64


import librosa
import librosa.display


# Basic Setup
st.set_page_config(
    page_title="Voxalyze",
    page_icon="🎙️",
    initial_sidebar_state="auto",

)
st.sidebar.image('voxalyze.png', use_column_width=True)

# Introduction
st.title('Voxalyze')

st.write('''🎈🎈🎈 Welcome to our Vocal Pattern App 🎈🎈🎈''')
st.write('''Here you can record a sound 🎙️ or upload a sound file 🎵 between 4 and 6 secods.
         Our app will show you the *spectogram* 📊 of the sound and will classify the sound as an **Arpegio**,
         a **Scale** or **Other type** of sound (as *melodies*, *long notes*, a funk and beautiful *improvisation* 🕺🏾, ...)''')


st.subheader("What's the difference between an *Arpeggio*, a *Scale* or *Other type* of sounds?") #Adding a divider

#Arpegios
st.write('''An arpeggio is a broken chord, or a chord in which individual
notes are struck one by one, rather than all together at once.
The word “arpeggio” comes from the Italian word “arpeggiare,” which means
"to play on a harp." (“Arpa” is the Italian word for “harp.”) *MasterClass, 2021*''')

def process_audio_file(uploaded_file):
    if uploaded_file is not None:
        st.audio(uploaded_file.read(), format="audio/wav")

def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)



# Construct the full file path to the audio file in the "sounds" directory
downloads_path = "sounds"
file_name_arpegio = "Arpegio.wav"
file_path_arpegio = os.path.join(downloads_path, file_name_arpegio)
file_name_scale = "Scale.wav"
file_path_scale = os.path.join(downloads_path, file_name_scale)
file_name_other = "Other.wav"
file_path_other = os.path.join(downloads_path, file_name_other)


# Calling the audio file
st.audio(file_path_arpegio, format="audio/wav")

# Ploting the spectogram for arpegios
audio_array_arpegio, sr_arpegio = librosa.load(file_path_arpegio, sr=None)

plt.figure(figsize=(10, 4))
spectrogram_arpegio = librosa.display.specshow(
    librosa.amplitude_to_db(np.abs(librosa.stft(audio_array_arpegio)), ref=np.max),
    y_axis='log', x_axis='time', sr=sr_arpegio
)
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram - Arpeggio')
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)


#Scales
st.write('''In music theory, a scale is any set of musical notes ordered by
fundamental frequency or pitch. A scale ordered by increasing pitch is an
ascending scale, and a scale ordered by decreasing pitch is a descending scale. *Wikipedia*''')

# Calling the audio file
st.audio(file_path_scale, format="audio/wav")

# Ploting the spectogram for scales
audio_array_scale, sr_scale = librosa.load(file_path_scale, sr=None)

plt.figure(figsize=(10, 4))
spectrogram_scale = librosa.display.specshow(
    librosa.amplitude_to_db(np.abs(librosa.stft(audio_array_scale)), ref=np.max),
    y_axis='log', x_axis='time', sr=sr_scale
)
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram - Scale')
st.pyplot()


# Other sounds
st.write('''We can also find other types of sounds, as *melodies*,
*long notes*, a funk and beautiful *improvisation* ''')

# Calling the audio file
st.audio(file_path_other, format="audio/wav")

# Ploting the spectogram for other sounds
audio_array_other, sr_other = librosa.load(file_path_other, sr=None)

plt.figure(figsize=(10, 4))
spectrogramother = librosa.display.specshow(
    librosa.amplitude_to_db(np.abs(librosa.stft(audio_array_other)), ref=np.max),
    y_axis='log', x_axis='time', sr=sr_other
)
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram - Arpegio')
st.pyplot()
