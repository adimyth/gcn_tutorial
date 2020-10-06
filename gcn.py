import tensorflow as tf
from layers.graph import GraphConvLayer


class GCNModel(tf.keras.Model):
    def __init__(self, input_dim, layer_blocks):
        super(GCNModel, self).__init__()
        self.input_dim = input_dim
        self.layer_blocks = layer_blocks
        self.gcn_layers = []
        for idx in range(len(self.layer_blocks)):
            if idx == 0:
                self.gcn_layers.append(GraphConvLayer(input_dim=self.input_dim,
                                                    output_dim=self.layer_blocks[0],
                                                    name=f"fc{idx+1}",
                                                    activation=tf.nn.tanh))
            else:
                self.gcn_layers.append(GraphConvLayer(input_dim=self.layer_blocks[idx-1],
                                                    output_dim=self.layer_blocks[idx],
                                                    name=f"fc{idx+1}",
                                                    activation=tf.nn.tanh))

    def call(self, inputs):
        adj_norm, x = inputs
        for idx in range(len(self.layer_blocks)):
            if idx==0:
                output = self.gcn_layers[idx](adj_norm, x, sparse=True)
            else:
                output = self.gcn_layers[idx](adj_norm, output)
        return output
