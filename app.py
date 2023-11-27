import streamlit as st
import matplotlib as plt

import numpy as np
import pandas as pd

import librosa, librosa.display
from audio_recorder_streamlit import audio_recorder

st.markdown("""# This is a header
## This is a sub header
This is text""")


audio_bytes = audio_recorder()
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
