import librosa
import librosa.display
import numpy as np
import pandas as pd


sample_rate = 22050


def process_audio(waveform, sr):
    # Set the target length to 6 seconds
    start_sample = int(0.0 * sr)
    target_length_sec = 6.0
    target_length_samples = int(target_length_sec * sr)  # ex: 6sec * 22 050 = 132 300
    # Check the current length of the audio
    current_length_sample = len(waveform)  # 165 853

    if current_length_sample > target_length_samples:
        # If the current length is longer, truncate the audio to 6 seconds
        wave_trunc = waveform[start_sample:target_length_samples]
    else:
        # If the current length is shorter, pad the audio to 6 seconds
        padded_signal = librosa.util.pad_center(waveform, size=target_length_samples)
        # Assign the padded signal to the truncated signal
        wave_trunc = padded_signal

    return wave_trunc  # this is our new wave_truncated


def mel_spectrogram(wave_trunc, sr):
    mel_spectrogram = librosa.feature.melspectrogram(y=wave_trunc, sr=sr)
    return librosa.power_to_db(mel_spectrogram, ref=np.max)


def scaling_spectrogram(power_to_db):
    min_value = np.min(power_to_db)
    max_value = np.max(power_to_db)

    # NORMALIZING gray array so that all values lie between [0, 1]
    normalized_spectrogram = (power_to_db - min_value) / (max_value - min_value)

    return normalized_spectrogram


def base_preprocess(
    waveform: np.ndarray, sr: int, augmentations: list | None = None
) -> np.ndarray:
    wave_trunc = process_audio(waveform, sr)
    spectrogram = mel_spectrogram(wave_trunc, sr)
    scaled_spectrogram = scaling_spectrogram(spectrogram)
    return scaled_spectrogram


def preprocess_train(X: pd.DataFrame, augmentations: list | None = None) -> np.ndarray:
    def run_preprocessing(filename):
        waveform, sr = librosa.load(filename, sr=sample_rate)
        return base_preprocess(waveform, sr, augmentations)

    X_processed_list = X["path"].map(run_preprocessing).tolist()
    # Stack the spectrograms along a new axis to create a 3D array
    X_processed_array = np.stack(X_processed_list, axis=0)
    return X_processed_array


def preprocess_predict(waveform: np.ndarray) -> np.ndarray:
    processed_waveform = base_preprocess(waveform, sample_rate)
    processed_waveform_3d = np.expand_dims(processed_waveform, axis=0)
    return processed_waveform_3d
