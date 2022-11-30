import tensorflow as tf

def shift_mask(mask, distance):
    """
    Input shape: (batch_size, audio_window_size)
    """
    mask = mask[:, :-distance]
    shifted = tf.pad(mask, [[0, 0], [distance, 0]], mode='CONSTANT', constant_values=True)
    return mask
 
def generate_starting_indices(input_shape, proportion):
    rand_vals = tf.random.uniform(input_shape)
    start_val = rand_vals < self.num_timesteps

    return start_val

def create_mask(input_shape, proportion, num_timesteps):
    start_val = generate_starting_indices
    ## Mask M consecutive time steps
    cumulative_mask = tf.ones(tf.shape(tf.inputs))
    for i in range(M):
        cumulative_mask *= shift_mask(mask, i)
    return cumulative_mask

def mask_audio(self, inputs, proportion, num_timesteps, default_val):
    """To mask the latent speech representations output by the encoder, we
    randomly sample without replacement a certain proportion p of all time
    steps to be starting indices and then mask the subsequent M consecutive
    time steps from every sampled index; spans may overlap."""

    mask = create_mask(tf.shape(inputs, proportion, num_timesteps))
    inputs = inputs * mask + default_val * (1 - mask)

    # Notes: try not to sample a time that comes after the audio clip
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
    

    def call(inputs, training=True):
        if (training):
            # Mask a portion of the audio
            return mask_audio(
                inputs,
                self.proportion,
                self.num_timesteps,
                self.mask_default)
        else:
            # Don't perform any masking
            return inputs
