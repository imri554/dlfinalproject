import tensorflow as tf
from cnn import SpeechCNN
from quantization import ProductQuantization
from transformer import SpeechTransformer

# Have a separate class for unsupervised, supervised
class Wav2VecPretraining(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Wav2Vec, self).__init__(**kwargs)

        self.cnn = SpeechCNN()
        self.mask = Mask(0.065, 10)
        self.transformer = SpeechTransformer(8, 64)
        self.quantization = ProductQuantization(2) # TODO: also add V argument
    
    def train_step(self, data):
        # TODO

        audio_features = self.cnn(audio)

        quantized_features = self.quantization(audio_features)

        # TODO: Mask audio features
        masked_features = audio_features
        
        prediction = self.transformer(captions, masked_features)

        # TODO: Compare prediction with actual quantization (maybe?)
        # self.add_loss
        
    def call(self, audio):
        """Predict text for a waveform of any length.
        
        Inputs should be a batch of a 1D vectors representing audio."""

        ## Split audio into segments (or do that above)
        ## Extract features from audio
        audio_features = self.cnn(audio)

        # TODO: Position

        transcript = self.transformer(captions, audio_features)

        return quantized_features

