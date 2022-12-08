import tensorflow as tf

# Transformers with convolutional context for ASR
# Abdelrahman Mohamed, Dmytro Okhonko, Luke Zettlemoyer
# https://doi.org/10.48550/arXiv.1904.11660
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, out_size, num_heads, attention_size, fc_size, dropout_rate, **kwargs):
        super().__init__(**kwargs)

        self.atten = tf.keras.layers.MultiHeadAttention(num_heads, attention_size)
        self.fc1 = tf.keras.layers.Dense(fc_size, activation='relu')
        self.fc2 = tf.keras.layers.Dense(out_size)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization()
    def call(self, x, context):
        x0 = x
        x = self.atten(x, context)
        x = self.dropout(x)
        x = self.layer_norm(x0 + x)
        x0 = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.layer_norm(x0 + x)
        return x


class EncoderConvolution(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, pool_size=2, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(num_filters, kernel_size)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.activation = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.pool = tf.keras.layers.MaxPool2D(2)
    def call(self, x):
        x = self.conv(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x

class DecoderConvolution(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(num_filters, kernel_size)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.activation = tf.keras.layers.Activation(tf.keras.activations.relu)
    def call(self, x):
        x = self.conv(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        return x
    
class ConvolutionEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(num_filters, kernel_size, padding='SAME')
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.activation = tf.keras.layers.Activation(tf.keras.activations.relu)
    def call(self, x):
        x = self.conv(x)
        # x = self.layer_norm(x)
        x = self.activation(x)
        return x


# The convolutional layer modeling relative positional embeddings
# has kernel size 128 and 16 groups.
# We experiment with two model configurations which use the same encoder architecture but differ in
# the Transformer setup: BASE contains 12 transformer blocks, model dimension 768, inner dimension
# (FFN) 3,072 and 8 attention heads.

# Conv layer : kernel size, num_groups
# 12 blocks (each?), 768 for each attention head, 3072 for first dense layer, 8 attention heads

class DecoderBlock(tf.keras.layers.Layer):
    """Models self-attention, context attention, and a feed-forward network."""
    def __init__(self, out_size, num_heads, attention_size, fc_size, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.self_atten = tf.keras.layers.MultiHeadAttention(num_heads, attention_size)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.context_atten = TransformerBlock(out_size, num_heads, attention_size, fc_size, dropout_rate)
    def call(self, inputs, context):
        x = inputs
        x = self.layer_norm(x + self.self_atten(x, x))
        x = self.context_atten(x, context)
        return x


class SpeechTransformer(tf.keras.Model):
    def __init__(self,
            num_embedding_blocks,
            embedding_kernel_size,
            hidden_size, ffn_size,
            num_blocks=1,
            num_heads=1, **kwargs):
        super().__init__(**kwargs)

        self.num_blocks = num_blocks # TODO
        # BASE contains 12 transformer blocks, model dimension 768, inner dimension
        # (FFN) 3,072 and 8 attention heads.

        # TODO: Check that parameters match up
        encoder_embedding_kernel_size = embedding_kernel_size
        decoder_embedding_kernel_size = embedding_kernel_size
        num_encoder_embedding_blocks = num_embedding_blocks
        num_decoder_embedding_blocks = num_embedding_blocks

        num_encoder_blocks = num_blocks
        num_decoder_blocks = num_blocks

        # TODO: Not sure about these parameters
        encoder_attention_size = hidden_size
        encoder_out_size = hidden_size
        decoder_attention_size = hidden_size
        decoder_out_size = hidden_size

        num_encoder_embedding_filters = encoder_out_size
        num_decoder_embedding_filters = decoder_out_size

        encoder_fc_size = ffn_size
        decoder_fc_size = ffn_size

        encoder_num_heads = num_heads
        decoder_num_heads = num_heads

        dropout_rate = 0.1


        self.encoder_embeddings = [
            ConvolutionEmbedding(
                name='encoder_embedding',
                num_filters=num_encoder_embedding_filters,
                kernel_size=encoder_embedding_kernel_size)
            for _ in range(num_encoder_embedding_blocks)]
        self.encoders = [
            TransformerBlock(
                name='encoder_transformer_block',
                out_size=encoder_out_size,
                num_heads=encoder_num_heads,
                attention_size=encoder_attention_size,
                fc_size=encoder_fc_size,
                dropout_rate=dropout_rate)
            for _ in range(num_encoder_blocks)]
        
        # self.decoder_attention = [
        #     tf.keras.layers.MultiHeadAttention(num_ema_heads, ema_size)
        #     for _ in range (num_decoder_blocks)]
        
        self.decoder_embeddings = [
            ConvolutionEmbedding(
                name='decoder_embedding',
                num_filters=num_decoder_embedding_filters,
                kernel_size=decoder_embedding_kernel_size,)
            for _ in range(num_decoder_embedding_blocks)]
        self.decoders = [
            DecoderBlock(
                name='decoder_transformer_block',
                out_size=decoder_out_size,
                num_heads=decoder_num_heads,
                attention_size=decoder_attention_size,
                fc_size=decoder_fc_size,
                dropout_rate=dropout_rate)
            for _ in range(num_decoder_blocks)]


    def call(self, audio_in, audio_pred):
        # TODO: Need to embed inputs with positions
        
        x = audio_in
        for emb in self.encoder_embeddings:
            x = emb(x)
        for enc in self.encoders:
            x = enc(x, x)
        encoder_output = x
        # TODO: I have forgotten everything about transformers

        # Embed decoder inputs
        x = audio_pred
        for emb in self.decoder_embeddings:
            x = emb(x)
        # Feed input into decoder blocks
        for dec in self.decoders:
            context = encoder_output
            x = dec(x, context)
        
        return x
