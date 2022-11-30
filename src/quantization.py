import tensorflow as tf

def gumbel_softmax(logits, temperature):
    u = tf.random.uniform(tf.shape(logits), minval=0, maxval=1)
    n = -tf.math.log(-tf.math.log(u))
    logits = logits + n

    # TODO: fill out function
    return tf.nn.softmax((logits + n) / temperature)

class ProductQuantization(tf.keras.layers.Layer):
    def __init__(self, num_groups, **kwargs):
        super().__init__(**kwargs)
        # TODO: consider using num_codebooks or codebook_size
        self.num_groups = num_groups
        self.linear = tf.keras.layers.Dense(num_groups) # TODO: Check argument
    
    def call(self, inputs):
        """
        Input shape: (batch_num,  num_channels)
        """

        # Split inputs into groups of appropriate size
        # TODO: expand inputs so that the input size is compatible with the
        # split size
        grouped_inputs = tf.split(inputs,
                                  self.num_groups,
                                  axis=-1)

        # Choose one entry from each codebook using (TODO: Gumbel) softmax
        # The Gumbel softmax temperature τ is annealed from 2
        # to a minimum of 0.5 for BASE
        probs = gumbel_softmax(grouped_inputs, 2)
        max_indices = tf.math.argmax(probs, axis=-1)

        # Apply linear layer
        out = self.linear(max_indices)
        return out
