from keras import backend as K
from keras.engine.topology import Layer, InputSpec

# I tried to do batch normalization, and I think I did something like it
# I followed the equations from this webpage:
# https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c
# And then I did my best to implement a custom layer in keras to do it. I know
# that gamma and beta should ideally be learned, but I just hardcoded them.

class CustomNormalization(Layer):
    def __init__(self, n=2, gamma=0.001, beta=0.70, **kwargs):
        self.n = n
        self.gamma = gamma
        self.beta = beta

        super(CustomNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape

        super(CustomNormalization, self).build(input_shape)

    def call(self, x, mask=None):

        # Get Mean
        avg_pool = K.pool2d(x, (self.n, self.n), strides=(1, 1),
        padding="same", pool_mode="avg") # mini-batch mean

        mid = K.square(x - avg_pool)

        # Get Variance
        var_pool = K.pool2d(mid, (self.n, self.n), strides=(1, 1),
        padding="same", pool_mode="avg") # mini-batch var

        # Normalize
        normalized = (x - avg_pool) / K.sqrt(var_pool)
        normalized = self.gamma * normalized + self.beta

        return normalized

    def get_output_shape_for(self, input_shape):

        return input_shape
