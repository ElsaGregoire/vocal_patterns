import librosa
import argparse
import numpy as np
from vocal_patterns.interface.main import predict
from vocal_patterns.ml_logic.registry import load_model
from vocal_patterns.ml_logic.preprocessor import preprocess_predict


def run_prediction(spectrograms, model):
    raw_predictions = []

    for spectrogram in spectrograms:
        spectrogram_expanded = np.expand_dims(spectrogram, axis=0)

        prediction = predict(spectrogram_expanded, model)  # predict is from main
        raw_predictions.append(prediction)

    # print("raw_predictions_sum", np.mean(raw_predictions, axis=0))
    prediction_map = {
        0: "Arpeggio",
        1: "Other",
        2: "Scale",
    }

    mean_prediction = np.mean(raw_predictions, axis=0)
    prediction = np.argmax(mean_prediction)
    confidence = np.max(mean_prediction) * 100
    prediction_str = prediction_map[prediction]

    return prediction_str, confidence


def recording_test(data, model):
    results = []
    results_key = []
    for index, row in data.iterrows():
        orig_waveform, sr = librosa.load(row["path"], sr=22050)

        # plot_spectrogram_librosa(orig_waveform, sr)
        processed = preprocess_predict(orig_waveform, model)
        prediction_str, confidence = run_prediction(processed, model)
        filename = row["filename"]

        reality = "Scale" if "scale" in filename.lower() else "Arpeggio"
        correct = reality == prediction_str

        results.append(correct)
        results_key.append({reality, correct, confidence})
        # Audio(orig_waveform, rate=sr)

    print("Accuracy:", np.mean(results) * 100, "%")
    return results_key
