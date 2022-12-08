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
    max_prob = tf.math.reduce_max(soft, axis=-1, keepdims=True)
    hard = tf.cast(soft == max_prob, tf.dtypes.float32)
    return hard - tf.stop_gradient(soft) + soft

def hard_gumbel_softmax(logits, temperature):
    soft = gumbel_softmax(logits, temperature)
    return hard_probs(soft)
