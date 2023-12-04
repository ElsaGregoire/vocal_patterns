import os
import librosa
import librosa.display
import numpy as np
import pandas as pd


sample_rate = 22050


def scaled_spectrogram(wave_trunc, sr):
    mel_spectrogram = librosa.feature.melspectrogram(y=wave_trunc, sr=sr)
    power_to_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    min_value = np.min(power_to_db)
    max_value = np.max(power_to_db)
    # NORMALIZING gray array so that all values lie between [0, 1]
    normalized_spectrogram = (power_to_db - min_value) / (max_value - min_value)
    return np.expand_dims(normalized_spectrogram, axis=-1)


def slice_waves(waveform, sr, snippet_duration=4, overlap=3):
    # Calculate the frame size and hop length
    frame_size = int(snippet_duration * sr)
    hop_length = int((snippet_duration - overlap) * sr)

    # Get the total number of snippets
    num_snippets = int(np.floor((len(waveform) - frame_size) / hop_length)) + 1

    new_4sec_arrays = []
    # Slice the audio into snippets
    for i in range(num_snippets):
        start_sample = i * hop_length
        end_sample = start_sample + frame_size
        snippet = waveform[start_sample:end_sample]
        # Append the snippet to the list
        new_4sec_arrays.append(snippet)

    return new_4sec_arrays


def stretch_waveforms(waveform, sr, target_duration=4.0):
    current_duration = librosa.get_duration(
        y=waveform, sr=sr
    )  # will be put ouo in float seconds

    if current_duration < target_duration:
        stretch_factor = current_duration / target_duration
        # Stretch the audio
        stretched_audio = librosa.effects.time_stretch(waveform, rate=stretch_factor)
        waveform = stretched_audio
        return waveform

    else:
        # else return the original audio
        return waveform


def noise_up_waveform(waveform, noise_level=0.001):
    np.random.normal(size=len(waveform))
    return waveform + (noise_level * np.random.normal(size=len(waveform)))


def preprocess_df(
    data: pd.DataFrame, augmentations: list | None = None, clearCashed: bool = False
):
    def process_data():
        data_list = []
        for index, row in data.iterrows():
            exercise = row["exercise"]
            technique = row["technique"]
            waveform, sr = librosa.load(row["path"], sr=sample_rate)

            stretch_waveforms(waveform, sr, target_duration=4.0)

            slice_waveforms = slice_waves(waveform, sr)
            for w in slice_waveforms:
                normalized_spectrogram = scaled_spectrogram(w, sr)
                data_list.append(
                    {
                        "spectrogram": normalized_spectrogram,
                        "exercise": exercise,
                        "technique": technique,
                    }
                )
        set_df = pd.DataFrame(data_list)
        return set_df

    if clearCashed == True:
        try:
            os.remove("preproc.pkl")
            print("Removed cached data")
        except FileNotFoundError:
            print("No cached data to remove")
    try:
        set_df = pd.read_pickle("preproc.pkl")
        print("Loaded cached preprocessing data")
    except FileNotFoundError:
        print("No cached preprocessing data found, preprocessing now...")
        set_df = process_data()
        set_df.to_pickle("preproc.pkl")

    return set_df


def preprocess_predict(waveform: np.ndarray):
    waveform = noise_up_waveform(waveform, noise_level=0.001)
    spectrograms = []
    stretched_waveform = stretch_waveforms(
        waveform, sr=sample_rate, target_duration=4.0
    )
    slice_waveforms = slice_waves(stretched_waveform, sr=sample_rate)
    for waveform in slice_waveforms:
        normalized_spectrogram = scaled_spectrogram(waveform, sr=sample_rate)
        spectrograms.append(normalized_spectrogram)

    return spectrograms
