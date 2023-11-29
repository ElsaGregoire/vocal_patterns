
def pitch_up_3(waveform, sr):
    '''this pitches up of 3 semi-tones, it's just after
    loading the file beacuse it changes the size of the
    wav file'''
    steps = float(3)
    y_pitched_up=librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    return y_pitched_up

def pitch_down_3(waveform, sr):
    '''this pitches up of 3 semi-tones, it's just after
    loading the file beacuse it changes the size of the
    wav file'''
    steps = float(-3)
    y_pitched_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    return y_pitched_down
