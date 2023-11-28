import librosa
import librosa.display
import numpy as np

# audio path to my wav file
audio_path = "/Users/elsagregoire/Desktop/Vocal Set Le Wagon/arpeggios/arpeggios f1/slow_forte/f1_arpeggios_c_slow_forte_a.wav"


def load_file(audio_path):
    # Load audio file using librosa
    y, sr = librosa.load(audio_path, sr=22050)
    # y is a numpy array  giving us the amplitude of the wave form at each value
    # sr is the sampling rate -> it will automatially resample to 22 050 when the file is loaded
    return (y, sr)


def process_audio(y, sr):
    # Set the target length to 6 seconds
    start_sample = int(0.0 * sr)
    target_length_sec = 6.0
    target_length_samples = int(target_length_sec * sr)  # ex: 6sec * 22 050 = 132 300

    # Check the current length of the audio
    current_length_samples = len(y)  # 165 853

    if current_length_samples > target_length_samples:
        # If the current length is longer, truncate the audio to 6 seconds
        y_trunc = y[start_sample:target_length_samples]
    else:
        # If the current length is shorter, pad the audio to 6 seconds
        padding_samples = target_length_samples - current_length_samples
        padded_signal = librosa.util.pad_center(
            y, target_length_samples + padding_samples
        )

        # Assign the padded signal to the truncated signal
        y_trunc = padded_signal

    return y_trunc  # this is our new y_truncated


def mel_spectrogram(y_trunc, sr):
    mel_spectrogram = librosa.feature.melspectrogram(y=y_trunc, sr=sr)
    db_spectrogram = librosa.display.specshow(
        librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis="mel", x_axis="time"
    )

    return db_spectrogram
