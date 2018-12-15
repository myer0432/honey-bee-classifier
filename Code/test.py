from keras import backend as K

from keras.engine.topology import Layer, InputSpec

class CustomNormalization(Layer):

def __init__(self, n=5, gamma=0.0005, beta=0.75, k=2, **kwargs):
    self.n = n
    self.gamma = gamma
    self.beta = beta
    self.k = k

    super(CustomNormalization, self).__init__(**kwargs)

def build(self, input_shape):
    self.shape = input_shape

    super(CustomNormalization, self).build(input_shape)
def call(self, x, mask=None):
    if K.image_dim_ordering == "tf":
        _, f, r, c = self.shape
    else:
        _, r, c, f = self.shape

    squared = K.square(x)
    pooled = K.pool2d(squared, (n, n), strides=(1, 1),
    padding="same", pool_mode="avg")

    if K.image_dim_ordering == "tf":
        summed = K.sum(pooled, axis=1, keepdims=True)
        averaged = self.alpha * K.repeat_elements(summed, f, axis=1)

    else:
        summed = K.sum(pooled, axis=3, keepdims=True)
        averaged = self.alpha * K.repeat_elements(summed, f, axis=3)

    denom = K.pow(self.k + averaged, self.beta)

    return x / denom

def get_output_shape_for(self, input_shape):

    return input_shape
