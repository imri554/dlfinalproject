import tensorflow as tf

def shift_mask(mask, distance):
    """
    Input shape: (batch_size, audio_window_size)
    """
    if (distance == 0):
        return mask
    out = mask[:, :-distance]
    out = tf.pad(out, [[0, 0], [distance, 0]], mode='CONSTANT', constant_values=True)
    return out
 
def generate_start_mask(input_shape, proportion):
    """False values represent masked values"""
    rand_vals = tf.random.uniform(input_shape)
    start_val = rand_vals >= proportion

    return start_val

def create_mask(input_shape, proportion, num_timesteps):
    start_mask = generate_start_mask(input_shape, proportion)
    ## Mask M consecutive time steps
    cumulative_mask = tf.fill(input_shape, True)
    for i in range(num_timesteps):
        cumulative_mask &= shift_mask(start_mask, i)
    return cumulative_mask

def mask_audio(inputs, mask, default_val):
    """
    Mask is (batch_size, audio_window_size).
    Inputs is (batch_size, audio_window_size, n_channels)
    """
    mask = tf.cast(mask, inputs.dtype)
    mask = tf.expand_dims(mask, -1)
    inputs = inputs * (mask) + default_val * (1 - mask)
    return inputs


class Mask(tf.keras.layers.Layer):
    def __init__(self, proportion, num_timesteps, **kwargs):
        super().__init__(**kwargs)
        self.proportion = proportion
        self.num_timesteps = num_timesteps
    def build(self, input_shape):
        """
        Input shape: (batch_size, audio_window_size, num_channels)
        """
        flattened_input_shape = [1 for x in input_shape]
        flattened_input_shape[-1] = input_shape[-1]
        self.mask_default = self.add_weight(
            name='mask_default',
            shape=flattened_input_shape)
    

    def call(self, inputs):
        """
        Input shape: (batch_size, num_audio_features, num_channels)
        """
        # Mask a portion of the audio
        mask = create_mask(tf.shape(inputs)[:-1], self.proportion, self.num_timesteps)

        return mask, mask_audio(
            inputs,
            mask,
            self.mask_default)
