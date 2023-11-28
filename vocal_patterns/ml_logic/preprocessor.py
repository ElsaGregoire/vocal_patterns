import librosa
import librosa.display
import numpy as np

# audio path to my wav file
audio_path = "/Users/elsagregoire/Desktop/Vocal Set Le Wagon/arpeggios/arpeggios f1/slow_forte/f1_arpeggios_c_slow_forte_a.wav"


def load_file(audio_path):
    # Load audio file using librosa
    waveform, sr = librosa.load(audio_path, sr=22050)
    # y is a numpy array  giving us the amplitude of the wave form at each value
    # sr is the sampling rate -> it will automatially resample to 22 050 when the file is loaded
    return (waveform, sr)


def process_audio(waveform, sr):
    # Set the target length to 6 seconds
    start_sample = int(0.0 * sr)
    target_length_sec = 6.0
    target_length_samples = int(target_length_sec * sr)  # ex: 6sec * 22 050 = 132 300

    # Check the current length of the audio
    current_length_samples = len(waveform)  # 165 853

    if current_length_samples > target_length_samples:
        # If the current length is longer, truncate the audio to 6 seconds
        wave_trunc = waveform[start_sample:target_length_samples]
    else:
        # If the current length is shorter, pad the audio to 6 seconds
        padding_samples = target_length_samples - current_length_samples
        padded_signal = librosa.util.pad_center(
            waveform, size=target_length_samples + padding_samples
        )

        # Assign the padded signal to the truncated signal
        wave_trunc = padded_signal

    return wave_trunc  # this is our new wave_truncated


def mel_spectrogram(wave_trunc, sr):
    # Generate a spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=wave_trunc, sr=sr)

    # convert to decibels, logscale
    # 2D NumPy array containing the intensity values at different frequencies and time points.
    # power_to_db represents as a grayscale image, not a color image!
    # Each element in the matrix represents the intensity or magnitude of the signal at a specific frequency and time.
    power_to_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # db_spectrogram = librosa.display.specshow(
    #     librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis="mel", x_axis="time"
    # )

    return power_to_db


def scaling_spectrogram(power_to_db):
    min_value = np.min(power_to_db)
    max_value = np.max(power_to_db)

    # NORMALIZING gray array so that all values lie between [0, 1]
    normalized_spectrogram = (power_to_db - min_value) / (max_value - min_value)

    return normalized_spectrogram
