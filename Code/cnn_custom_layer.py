from keras import backend as K
from keras.engine.topology import Layer

class cnn_dense(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(cnn_layer, self).__init__(**kwargs)

    # Called when the model containing the layer is built
    # This is where you set up the weights of the layer
    # The input_shape is accepted as an argument to the function
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = K.variable('''an_init_numpy_array?''')
        self.trainable_weights = [self.W]
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.output_dim),
            initializer='uniform', trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    # Defines the computations performed on the input
    # Accepts the input tensor as its argument
    # Returns the output tensor after applying the required operations
    def call(self, x):
        return K.dot(x, self.kernel)

    # Required for Keras to infer the shape of the output
    # This allows Keras to do shape inference without actually executing the computation
    # The input_shape is passed as the argument
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
