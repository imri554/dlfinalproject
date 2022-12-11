import tensorflow as tf

from gumbelsoftmax import gumbel_softmax, hard_gumbel_softmax


class ProductQuantization(tf.keras.layers.Layer):
    def __init__(self, num_groups, **kwargs):
        super().__init__(**kwargs)
        # G = num_groups
        # V is feature_space_depth // G
        # d = feature_space_depth
        # codebook_entry_size = d / G
        # TODO: consider using num_codebooks or codebook_size
        self.num_groups = num_groups
    
    def build(self, input_shape):
        self.codebooks = tf.keras.layers.Dense(input_shape[-1] // self.num_groups)
    

    def call(self, inputs):
        """
        Input shape: (batch_size, num_extracted_features, num_channels)
        """

        # Split inputs into groups of appropriate size
        # TODO: expand inputs so that the input size is compatible with the
        # split size
        original_shape = tf.shape(inputs)
        grouped_shape = [x for x in original_shape]
        grouped_shape[-1] = self.num_groups
        grouped_shape.append(-1)
        grouped_inputs = tf.reshape(inputs, grouped_shape)

        # Choose one entry from each codebook using Gumbel softmax
        # The Gumbel softmax temperature τ is annealed from 2
        # to a minimum of 0.5 for BASE
        onehot_entries = hard_gumbel_softmax(grouped_inputs, 2)
        coded_entries = self.codebooks(onehot_entries)
        # Concatenate entries
        coded_entries = tf.reshape(coded_entries, original_shape)

        return coded_entries
