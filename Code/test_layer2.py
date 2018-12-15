from keras.layers.core import Dropout, Reshape
from test_layer import test_layer
from norm_custom_layer import CustomNormalization
from keras.layers.convolutional import ZeroPadding2D
import numpy as np


x = np.random.randn(10, 10)
layer = Dropout(0.5)
y = test_layer(layer, x)
assert(x.shape == y.shape)

x = np.random.randn(10, 10, 3)
layer = ZeroPadding2D(padding=(1,1))
y = test_layer(layer, x)
assert(x.shape[0] + 2 == y.shape[0])
assert(x.shape[1] + 2 == y.shape[1])

x = np.random.randn(10, 10)
layer = Reshape((5, 20))
y = test_layer(layer, x)
assert(y.shape == (5, 20))

x = np.random.randn(10, 10, 3)
layer = CustomNormalization()
y = test_layer(layer, x)
assert(y.shape == (10, 10, 3))
