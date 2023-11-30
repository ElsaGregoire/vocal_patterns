def slice_4(waveform, sr, snippet_duration=4, overlap=3):
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



def data_augm(input_df):

    def load_and_augment(row):
        audio_path = row["path"]
        exercise = row["exercise"]

        # Load the audio file
        waveform, sr = librosa.load(audio_path, sr=None)

        # Perform data augmentation, e.g., change pitch, speed, add noise, etc.
        # augmented_waveform = your_augmentation_function(waveform)

        # Slice the waveform into 4-second snippets
        snippets = slice_4(waveform, sr)

        # Create a DataFrame with snippets and exercise labels
        df = pd.DataFrame({"waveform": snippets, "exercise": [exercise] * len(snippets)})
        return df

    # Apply the load_and_augment function to the first 5 rows of the input DataFrame
    augmented_data = pd.concat([load_and_augment(row) for _, row in input_df.head(5).iterrows()], ignore_index=True)

    return augmented_data
