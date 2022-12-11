import tensorflow as tf

def gumbel_softmax(logits, temperature):
    rand_shape = [x for x in tf.shape(logits)]
    # rand_shape[-2] = 1
    u = tf.random.uniform(rand_shape, minval=0, maxval=1)
    n = -tf.math.log(-tf.math.log(u))

    # TODO: fill out function
    return tf.nn.softmax((logits + n) / temperature)

# https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html
def hard_probs(soft):
    original_shape = tf.shape(soft)
    num_categories = original_shape[-1]
    flattened_shape = [-1, num_categories]

    # Take samples from soft probabilities
    logits = soft / (1 - soft)
    samples = tf.random.categorical(tf.reshape(logits, flattened_shape), 1)
    # Create hard probabilities
    hard = tf.one_hot(samples, num_categories)
    hard = tf.reshape(hard, original_shape)
    hard = tf.cast(hard, tf.dtypes.float32)
    return hard - tf.stop_gradient(soft) + soft

def hard_gumbel_softmax(logits, temperature):
    soft = gumbel_softmax(logits, temperature)
    return hard_probs(soft)
