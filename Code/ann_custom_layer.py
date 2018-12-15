from keras import backend
from keras.engine.topology import Layer

# Custom fully connected layer
class dense_custom_layer(Layer):
    # Constructor
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ann_dense, self).__init__(**kwargs)

    # Build function
    #
    # Description: Sets the weights for the layer
    # @param input_shape: The shape of input into the layer
    def build(self, input_shape):
        # Add weights
        self.kernel = self.add_weight(name='kernel',
            shape=(input_shape[1], self.output_dim), initializer='uniform', trainable=True)
        # Build
        super(ann_dense, self).build(input_shape)

    # Call function
    #
    # Description: Performs a dot product on the weights
    def call(self, x):
        tensor = backend.dot(x, self.kernel)
        return tensor

    # Compute output shape function
    #
    # Description: Allows Keras to determine the output shape
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
