import numpy as np
import soundfile as sf

def load_audio_file(path):
    data, samplerate = sf.read(path)
    return data, samplerate