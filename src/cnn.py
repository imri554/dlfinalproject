import tensorflow as tf

class SpeechCNN(tf.keras.Model):
    def __init__(self, num_channels):
        super().__init__()

        self.seq = tf.keras.Sequential()

        kernel_widths = (10,3,3,3,3,2,2)
        strides = (5,2,2,2,2,2,2)
        for width, stride in zip(kernel_widths, strides):
            self.seq.add(tf.keras.layers.Conv1D(num_channels,
                                                width,
                                                strides=stride,
                                                input_shape=(None, 1)))
        
        self.seq.add(tf.keras.layers.LayerNormalization())
        self.seq.add(tf.keras.layers.Activation(tf.keras.activations.gelu))
        
    
    def call(self, inputs):
        return self.seq(inputs)
