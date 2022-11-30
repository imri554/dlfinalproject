
from model import Wav2Vec

if __name__ == '__main__':
    # Testing only, going to get the development data

    names, audio, transcripts = get_librispeech_data('../data/LibriSpeech/dev-clean')

    print('Done loading')
    print(names[0], audio[0], transcripts[0])

    # Group and reshape audio
    # 10 seconds of audio
    window_size = 10 * 16000
    padded_audio = [tf.keras.utils.pad_sequences(a, padding='post') for a in audio]
    train_inputs = tf.constant(padded_audio)

    model = Wav2VecPretraining()

    model(inputs)

    # Probably not possible to compile like this?
    # model.compile(optimizer=Adam(1e-5),
    #               loss=None,
    #               metrics=[])
    model.compile(optimizer=Adam(1e-5))
