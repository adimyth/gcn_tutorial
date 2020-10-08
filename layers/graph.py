import tensorflow as tf


def matmul(x, y, sparse=False):
    """Wrapper for sparse matrix multiplication."""
    if sparse:
        return tf.sparse.sparse_dense_matmul(x, y)
    return tf.matmul(x, y)


class GraphConvLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation=None, use_bias=False, name="graph_conv"):
        super(GraphConvLayer, self).__init__()
        """Initialise a Graph Convolution layer.

        Args:
            output_dim (int): The output dimensionality, i.e. the number of
                units.
            activation (callable): The activation function to use. Defaults to
                no activation function.
            use_bias (bool): Whether to use bias or not. Defaults to `False`.
            name (str): The name of the layer. Defaults to `graph_conv`.
        """
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        self.input_dim = input_shape[1][-1]
        self.w = self.add_weight(
            name="w",
            shape=(self.input_dim, self.output_dim),
            initializer=tf.initializers.glorot_uniform(),
        )

        if self.use_bias:
            self.b = tf.add_weight(
                name="b", initializer=tf.constant(0.1, shape=(self.output_dim,))
            )

    def call(self, inputs):
        adj_norm = inputs[0]
        x = inputs[1]
        x = matmul(x=x, y=self.w, sparse=False)  # XW
        x = matmul(x=adj_norm, y=x, sparse=True)  # AXW

        if self.use_bias:
            x = tf.add(x, self.use_bias)  # AXW + B

        if self.activation is not None:
            x = self.activation(x)  # activation(AXW + B)

        return x
