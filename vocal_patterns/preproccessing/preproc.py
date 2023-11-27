import librosa
import librosa.display
import numpy as np

# audio path to my wav file
audio_path = '/Users/elsagregoire/Desktop/Vocal Set Le Wagon/arpeggios/arpeggios f1/slow_forte/f1_arpeggios_c_slow_forte_a.wav'

# loads the audio file and resamples the sr to 22050
y, sr = librosa.load(audio_path, sr=22050)
# y is a numpy array  giving us the amplitude of the wave form at each value
# sr is the sampling rate -> it will automatially resample to 22 050 when the file is loaded
