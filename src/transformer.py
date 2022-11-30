import tensorflow as tf

class TransformerEncoder(tf.keras.Model):
    def __init__(self, num_heads, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads, hidden_size)
        self.layer_norm = tf.keras.layers.LayerNormalization()
    
    def call(self, inputs):
        x = inputs
        # Call self-attention
        x = self.attention(x, x)
        x = self.layer_norm(inputs + x)
        return x

class TransformerDecoder(tf.keras.Model):
    def __init__(self, num_heads, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads, hidden_size)
        self.context_attention = tf.keras.layers.MultiHeadAttention(num_heads, hidden_size)
        self.layer_norm = tf.keras.layers.LayerNormalization()
    
    def call(self, inputs, context):
        x = inputs
        # Call self-attention
        x2 = self.self_attention(x, x)
        x = self.layer_norm(x + x2)
        x2 = self.context_attention(x, context)
        x = self.layer_norm(x + x2)
        return x

class SpeechTransformer(tf.keras.Model):
    def __init__(self, num_heads, hidden_size, **kwargs):
        super().__init__(**kwargs)

        self.embedding = tf.keras.layers.Dense(hidden_size)
        self.encoder = TransformerEncoder(num_heads, hidden_size)
        self.decoder = TransformerDecoder(num_heads, hidden_size)

    def call(self, audio, transcript):
        # TODO: Need to embed inputs with positions
        inputs = self.embedding(transcript)
        
        context = self.encoder(audio)
        decoded = self.decoder(inputs, context)

        return decoded
