from keras.models import Sequential
from norm_custom_layer import CustomNormalization
from keras.layers.core import Dropout, Reshape
import numpy as np

def test_layer(layer, x):
    layer_config = layer.get_config()

    layer_config["input_shape"] = x.shape

    layer = layer.__class__.from_config(layer_config)

    model = Sequential()
    model.add(layer)
    model.compile("rmsprop", "mse")

    x_ = np.expand_dims(x, axis=0)

    return model.predict(x_)[0]
