import numpy as np
import tensorflow as tf

# Note: I'm pretty sure these shouldn't subclass tf.keras.losses.Loss because
# The training step is somewhat specialized

def contrastive_loss(y_true, y_pred, mask):
    """
    y_true: (batch_size, audio_window_size, quantized_size)
    y_pred: (batch_size, audio_window_size, quantized_size)
    mask: (batch_size, audio_window_size)
    """
    tf.ensure_shape(y_pred, tf.shape(y_true))

    num_mask = tf.cast(mask, y_pred.dtype)
    num_masked_steps = tf.reduce_sum(1 - num_mask, axis=-1, keepdims=True)
    num_distractors = num_masked_steps - 1

    # Normalize so that we can compute cosine similarity using the dot product
    normalized_y_true = tf.linalg.l2_normalize(y_pred, axis=-1)
    normalized_y_pred = tf.linalg.l2_normalize(y_true, axis=-1)
    
    y_true_masked = normalized_y_true * tf.expand_dims(1 - num_mask, -1)
    y_pred_masked = normalized_y_pred * tf.expand_dims(1 - num_mask, -1)

    # (batch_size, num_audio_features, num_audio_features)
    similarity = tf.matmul(
        tf.transpose(y_pred_masked, [0, 1, 2]),
        tf.transpose(y_true_masked, [0, 2, 1]))
    # similarity_i,j = similarity(y_pred_i, y_true_j)

    # Mask columns
    similarity += -1e7 * tf.expand_dims(num_mask, -2)

    loss = -tf.nn.log_softmax(similarity / tf.expand_dims(num_distractors, -1))
    # Select the correct element
    loss = tf.linalg.diag_part(loss) # (batch_size, num_audio_features)
    loss *= (1 - num_mask)
    loss = tf.math.reduce_sum(loss, axis=-1)

    return tf.math.reduce_mean(loss)

def diversity_loss(y_pred):
    # calculate probabilities using a softmax
    y_probs = tf.nn.softmax(y_pred)

    out = tf.math.reduce_sum(y_probs * tf.math.log(y_probs), axis=-1)
    out = out / tf.cast(tf.shape(y_pred)[-1], out.dtype)

    out = tf.math.reduce_sum(out, axis=-1)
    return tf.math.reduce_mean(out)


class FlattenedSparseCategoricalCrossentropy(tf.losses.Loss):
    def __init__(self, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.scce = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits)
    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, (-1,))
        y_pred = tf.reshape(y_pred, (-1, tf.shape(y_pred)[-1]))
        return self.scce(y_true, y_pred)

        