import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse
from spektral.layers import GraphConv
import tensorflow as tf
import tensorflow.keras.backend as K

import utils.sparse as us

g = nx.read_graphml("R/karate.graphml")

nx.draw(
    g,
    cmap=plt.get_cmap('jet'),
    node_color=np.log(list(nx.get_node_attributes(g, 'membership').values())))

# Adjacency Matrix
adj = nx.adj_matrix(g)
# Get important parameters of adjacency matrix
n_nodes = adj.shape[0]
# Features are just the identity matrix
feat_x = np.identity(n=adj.shape[0])

# Semi-supervised
memberships = [m - 1 for m in nx.get_node_attributes(g, "membership").values()]

nb_classes = len(set(memberships))
targets = np.array([memberships], dtype=np.int32).reshape(-1)
one_hot_targets = np.eye(nb_classes)[targets]

# Pick one at random from each class
labels_to_keep = [
    np.random.choice(np.nonzero(one_hot_targets[:, c])[0]) for c in range(nb_classes)
]

y_train = np.zeros(shape=one_hot_targets.shape, dtype=np.float32)
y_val = one_hot_targets.copy()

train_mask = np.zeros(shape=(n_nodes,), dtype=np.bool)
val_mask = np.ones(shape=(n_nodes,), dtype=np.bool)

for l in labels_to_keep:
    y_train[l, :] = one_hot_targets[l, :]
    y_val[l, :] = np.zeros(shape=(nb_classes,))
    train_mask[l] = True
    val_mask[l] = False

# GCN Preprocessing
adj = GraphConv.preprocess(adj).astype('f4')

print(f"Features: {feat_x.shape}")
print(f"Adjacency Matrix: {adj.shape}")
print(f"Train target: {y_train.shape}")
print(f"Validation target: {y_val.shape}")
print(f"Train Mask: {train_mask.shape}")
print(f"Validation Mask: {val_mask.shape}", end="\n\n")

# BUILDING 4-Layer GCN MODEL
feat_in = tf.keras.layers.Input(shape=(feat_x.shape[-1],))
adj_in = tf.keras.layers.Input(shape=(adj.shape[0],), sparse=True)
l_sizes = [4, 4, 2, nb_classes]

o_fc1 = GraphConv(l_sizes[0], tf.nn.tanh)([feat_in, adj_in])
o_fc2 = GraphConv(l_sizes[1], tf.nn.tanh)([o_fc1, adj_in])
o_fc3 = GraphConv(l_sizes[2], tf.nn.tanh)([o_fc2, adj_in])
o_fc4 = GraphConv(l_sizes[3], tf.identity)([o_fc3, adj_in])
model = tf.keras.models.Model(inputs=[feat_in, adj_in], outputs=o_fc4)
model.compile(optimizer="adam", loss="categorical_crossentropy", weighted_metrics=["acc"])

model.fit([feat_x, adj], y_train,
        sample_weight=train_mask,
        validation_data=([feat_x, adj], y_val, val_mask),
        batch_size=n_nodes,
        shuffle=False)

loss, acc = model.evaluate([feat_x, adj], y_val,
                            sample_weight=val_mask,
                            batch_size=n_nodes)
print(f"Validation Loss: {loss}\tValidation Accuracy: {acc}")
