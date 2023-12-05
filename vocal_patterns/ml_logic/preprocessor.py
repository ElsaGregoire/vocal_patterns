import os
import random
import librosa
import librosa.display
import numpy as np
import pandas as pd

from vocal_patterns.params import SAMPLE_RATE


sample_rate = SAMPLE_RATE


def scaled_spectrogram(wave_trunc, sr=sample_rate):
    mel_spectrogram = librosa.feature.melspectrogram(y=wave_trunc, sr=sr)
    power_to_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    min_value = np.min(power_to_db)
    max_value = np.max(power_to_db)
    # NORMALIZING gray array so that all values lie between [0, 1]
    normalized_spectrogram = (power_to_db - min_value) / (max_value - min_value)
    return np.expand_dims(normalized_spectrogram, axis=-1)


def slice_waves(waveform, sr=sample_rate, snippet_duration=4, overlap=3):
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


def stretch_waveforms(waveform, sr=sample_rate, target_duration=4.0):
    current_duration = librosa.get_duration(
        y=waveform, sr=sr
    )  # will be put ouo in float seconds

    if current_duration < target_duration:
        stretch_factor = current_duration / target_duration
        # Stretch the audio
        stretched_audio = librosa.effects.time_stretch(waveform, rate=stretch_factor)
        waveform = stretched_audio
        return waveform, sr

    else:
        # else return the original audio
        return waveform, sr


def noise_up_waveform(waveform, noise_level=0.001):
    np.random.normal(size=len(waveform))
    return waveform + (noise_level * np.random.normal(size=len(waveform)))


def add_background_noise(waveform, sr=sample_rate, noise_level=0.8):
    sample = "/Users/jake/code/jchaselubitz/vocal_patterns/data/raw_data/office-ambience-6322.mp3"
    background_sample, sr = librosa.load(sample, sr=sr, mono=True)
    waveform_length = len(waveform)
    sample_length = len(background_sample)
    length_difference = sample_length - waveform_length
    if sample_length > waveform_length:
        random_sample_with_wf_length = random.randint(20, length_difference)
        background_sample = background_sample[
            random_sample_with_wf_length : waveform_length
            + random_sample_with_wf_length
        ]
        mixed_waveform = waveform + noise_level * background_sample
    else:
        waveform = waveform[sample_length]
        mixed_waveform = waveform + noise_level * background_sample
    return mixed_waveform, sr


def preprocess_df(
    data: pd.DataFrame, augmentations: dict | None = None, clearCached: bool = False
):
    def process_data():
        data_list = []
        for index, row in data.iterrows():
            print(index, "/", len(data), f"({index/len(data)*100:.2f}%)")
            exercise = row["exercise"]
            technique = row["technique"]
            waveform, sr = librosa.load(row["path"], sr=sample_rate)
            waveform, sr = stretch_waveforms(
                waveform,
                sr=sample_rate,
                target_duration=augmentations["stretch_target_duration"]
                if augmentations is not None
                else 4.0,
            )
            if "background_noise" in augmentations:
                waveform, sr = add_background_noise(
                    waveform, sr, noise_level=augmentations["background_noise"]
                )
            if "noise_up" in augmentations:
                waveform = noise_up_waveform(
                    waveform, noise_level=augmentations["noise_up"]
                )
            if "snippets" in augmentations:
                duration = augmentations["snippets"]["duration"]
                overlap = augmentations["snippets"]["overlap"]
                slice_waveforms = slice_waves(
                    waveform,
                    sr=sample_rate,
                    overlap=overlap,
                    snippet_duration=duration,
                )
                for w in slice_waveforms:
                    normalized_spectrogram = scaled_spectrogram(w, sr)
                    data_list.append(
                        {
                            "spectrogram": normalized_spectrogram,
                            "exercise": exercise,
                            "technique": technique,
                        }
                    )
            else:
                normalized_spectrogram = scaled_spectrogram(waveform, sr)
                data_list.append(
                    {
                        "spectrogram": normalized_spectrogram,
                        "exercise": exercise,
                        "technique": technique,
                    }
                )
        set_df = pd.DataFrame(data_list)
        return set_df

    if clearCached == True:
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


def preprocess_predict(waveform: np.ndarray, model=None):
    waveform = noise_up_waveform(waveform, noise_level=0.01)

    augmentations = model.augmentations
    slice_params = augmentations["snippets"]
    spectrograms = []
    stretched_waveform, sr = stretch_waveforms(
        waveform, sr=sample_rate, target_duration=4.0
    )
    assert stretched_waveform.shape[0] >= sample_rate * 4
    slice_waveforms = slice_waves(
        stretched_waveform,
        sr=sample_rate,
        overlap=slice_params["overlap"],
        snippet_duration=slice_params["duration"],
    )
    for waveform in slice_waveforms:
        normalized_spectrogram = scaled_spectrogram(waveform, sr=sample_rate)
        spectrograms.append(normalized_spectrogram)

    return spectrograms
