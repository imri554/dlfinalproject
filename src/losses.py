import tensorflow as tf

# Note: I'm pretty sure these shouldn't subclass tf.keras.losses.Loss because
# The training step is somewhat specialized

class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        """
        True dimensions: (batch_num, quantization_size)
        Predicted dimensions: (batch_num, quantization_size, alphabet_size)
        """
        # Calculate similarities between y_true and y_pred

        # Apply softmax
        # Apply -log


class DiversityLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        # Inputs: 
        # We should actually disregard y_true
        
        # calculate probabilities using a softmax
        y_probs = tf.nn.softmax(y_pred)
        # TODO: Calculate entropy
        information = tf.log(y_probs)
        
        out = tf.math.reduce_sum(y_probs * information)
        # TODO: calculate G and V
        out = out / G / V
        return out
