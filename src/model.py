import tensorflow as tf
from cnn import SpeechCNN
from quantization import ProductQuantization
from transformer import SpeechTransformer
from mask import Mask
from losses import contrastive_loss, diversity_loss

class QuantizationModule(tf.keras.layers.Layer):
    def __init__(self, num_groups, num_codebook_entries, out_size, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(num_groups * num_codebook_entries)
        self.quantization = ProductQuantization(num_groups)
        self.dense2 = tf.keras.layers.Dense(out_size)
    def call(self, inputs):
        x = inputs
        x = self.dense1(x)
        x = self.quantization(x)
        x = self.dense2(x)
        return x


# Have a separate class for unsupervised, supervised
class Wav2VecPretraining(tf.keras.Model):
    def __init__(self,
            hidden_size,

            num_feature_channels=512,

            num_transformer_embedding_blocks=16,
            transformer_embedding_kernel_size=16, # 128
            
            transformer_ffn_size=3072,
            num_transformer_blocks=12,
            num_attention_heads=8,

            mask_proportion=0.065,
            mask_timesteps=10,

            num_codebook_groups=2, 
            codebook_group_size=320,

            **kwargs):
        super().__init__(**kwargs)

        self.normalization = tf.keras.layers.Normalization()
        self.cnn = SpeechCNN(num_channels=num_feature_channels)
        self.mask = Mask(proportion=mask_proportion, num_timesteps=mask_timesteps)
        self.transformer = SpeechTransformer(
            num_embedding_blocks=num_transformer_embedding_blocks,
            embedding_kernel_size=transformer_embedding_kernel_size,
            hidden_size=hidden_size,
            ffn_size=transformer_ffn_size,
            num_blocks=num_transformer_blocks,
            num_heads=num_attention_heads)

        # Quantization module
        self.quantization = QuantizationModule(num_codebook_groups, codebook_group_size, hidden_size)

        # Used to weight losses
        self.diversity_weight = 1
    
    def batch_step(self, data, training=True):
        # TODO: There's something wrong about the transformer and product
        # quantization. I'm also not sure if the model structure is missing pieces,
        # but hopefully it's mostly correct
        audio = data

        with tf.GradientTape() as tape:
            # Feature encoder module
            norm_audio = self.normalization(audio)
            audio_features = self.cnn(norm_audio)

            quantized_features = self.quantization(audio_features)

            # Mask audio features
            mask, masked_features = self.mask(audio_features)
            # TODO: prediction should be same size as quantization. Not sure
            # what the correct size for either would be.
            prediction = self.transformer(masked_features, masked_features)

            # Compare prediction with actual quantization
            c_loss = contrastive_loss(quantized_features, prediction, mask)
            # Add diversity loss
            d_loss = diversity_loss(prediction)

            loss = c_loss + self.diversity_weight * d_loss

        if training:
            if (tf.math.is_nan(loss)):
                print('nan loss, skipping')
            else:
                gradients = tape.gradient(loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        return {'contrastive_loss': c_loss, 'diversity_loss': d_loss}
        
    def train_step(self, data):
        return self.batch_step(data, training=True)
    
    def test_step(self, data):
        return self.batch_step(data, training=False)

    def call(self, audio):
        """Transform audio.
        
        Inputs should be a batch of a sequence of 1D vectors."""

        ## Split audio into segments (or do that above)
        ## Extract features from audio
        audio_features = self.cnn(audio)
        # Transform masked/masked audio
        transformed_features = self.transformer(audio_features, audio_features)

        return transformed_features


class Wav2VecFineTuning(tf.keras.Model):
    def __init__(self, pretrained_model, num_classes, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.pretrained = pretrained_model
        self.projection = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='leaky_relu'),
            tf.keras.layers.Dense(num_classes),
        ])
    def call(self, audio):
        x = audio
        x = self.pretrained(x)
        x = self.projection(x)
        return x
