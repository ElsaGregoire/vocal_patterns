import os
import random
import librosa
import librosa.display
import numpy as np
import pandas as pd
import noisereduce as nr

from vocal_patterns.params import SAMPLE_RATE


sample_rate = SAMPLE_RATE


def scaled_spectrogram(wave_trunc, sr=sample_rate, fmin=400, fmax=1500):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=wave_trunc, sr=sr, fmin=fmin, fmax=fmax
    )
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

    # if current_duration < target_duration:
    stretch_factor = current_duration / target_duration
    # Stretch the audio
    stretched_audio = librosa.effects.time_stretch(waveform, rate=stretch_factor)
    waveform = stretched_audio
    return waveform, sr

    # else:
    #     # else return the original audio
    #     return waveform, sr


def noise_up_waveform(waveform, noise_level=0.001):
    np.random.normal(size=len(waveform))
    return waveform + (noise_level * np.random.normal(size=len(waveform)))


def reduce_noise(waveform):
    waveform = nr.reduce_noise(
        y=waveform,
        sr=sample_rate,
        n_std_thresh_stationary=0.5,
        stationary=True,
    )
    return waveform


def clip_margins(waveform, margin_percent=15):
    margin_decimal = margin_percent / 100
    margin = int(margin_decimal * len(waveform))
    end = -margin
    return waveform[margin:end]


def add_background_noise(waveform, sr=sample_rate, noise_level=0.8):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    up_two_parents = os.path.dirname(os.path.dirname(script_dir))
    data_file_path = os.path.join(up_two_parents, "data")
    sample = os.path.join(data_file_path, "office-ambience-6322.mp3")
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
            fmin = augmentations["fmin"]
            fmax = augmentations["fmax"]
            waveform, sr = librosa.load(row["path"], sr=sample_rate)
            if "clip_margins" in augmentations:
                waveform = clip_margins(
                    waveform, margin_percent=augmentations["margin_percent"]
                )
            waveform, sr = stretch_waveforms(
                waveform,
                sr=sample_rate,
                target_duration=augmentations["stretch_target_duration"],
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
                    normalized_spectrogram = scaled_spectrogram(
                        w, sr, fmin=fmin, fmax=fmax
                    )
                    data_list.append(
                        {
                            "spectrogram": normalized_spectrogram,
                            "exercise": exercise,
                            "technique": technique,
                        }
                    )
            else:
                normalized_spectrogram = scaled_spectrogram(
                    waveform, sr, fmin=fmin, fmax=fmax
                )
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
    # waveform = reduce_noise(waveform)

    augmentations = model.augmentations
    spectrograms = []
    if "clip_margins" in augmentations:
        waveform = clip_margins(
            waveform, margin_percent=augmentations["margin_percent"]
        )
    stretched_waveform, sr = stretch_waveforms(
        waveform,
        sr=sample_rate,
        target_duration=augmentations["stretch_target_duration"],
    )
    assert (
        stretched_waveform.shape[0]
        >= sample_rate * augmentations["stretch_target_duration"]
    )
    try:
        assert augmentations["snippets"] is not None
        slice_params = augmentations["snippets"]
        slice_waveforms = slice_waves(
            stretched_waveform,
            sr=sample_rate,
            overlap=slice_params["overlap"],
            snippet_duration=slice_params["duration"],
        )
    except:  # if no snippets augmentation
        slice_waveforms = [stretched_waveform]
    for waveform in slice_waveforms:
        fmin = augmentations["fmin"]
        fmax = augmentations["fmax"]
        normalized_spectrogram = scaled_spectrogram(
            waveform, sr=sample_rate, fmin=fmin, fmax=fmax
        )
        spectrograms.append(normalized_spectrogram)

    return spectrograms
