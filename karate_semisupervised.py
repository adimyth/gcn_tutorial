import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse
import tensorflow as tf
import tensorflow.keras.backend as K

import layers.graph as lg
import utils.sparse as us

g = nx.read_graphml("R/karate.graphml")

nx.draw(
    g,
    cmap=plt.get_cmap("jet"),
    node_color=np.log(list(nx.get_node_attributes(g, "membership").values())),
)

adj = nx.adj_matrix(g)
# Get important parameters of adjacency matrix
n_nodes = adj.shape[0]

# Some preprocessing
adj_tilde = adj + np.identity(n=adj.shape[0])
d_tilde_diag = np.squeeze(np.sum(np.array(adj_tilde), axis=1))
d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -1 / 2)
d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)
adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt)
adj_norm_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(adj_norm))
adj_norm_sparse_tensor = us.tuple_to_sparsetensor(adj_norm_tuple)

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


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


print(f"Features: {feat_x.shape}")
print(f"Adjacency Matrix: {adj_norm_sparse_tensor.shape}")
print(f"Train target: {y_train.shape}")
print(f"Validation target: {y_val.shape}")
print(f"Train Mask: {train_mask.shape}")
print(f"Validation Mask: {val_mask.shape}", end="\n\n")

# BUILDING 4-Layer GCN MODEL
feat_in = tf.keras.layers.Input(shape=(feat_x.shape[-1],))
adj_in = tf.keras.layers.Input(shape=(adj_norm_sparse_tensor.shape[0],), sparse=True)
l_sizes = [4, 4, 2, nb_classes]

o_fc1 = lg.GraphConvLayer(output_dim=l_sizes[0], name="fc1", activation=tf.nn.tanh)(
    [adj_in, feat_in]
)

o_fc2 = lg.GraphConvLayer(output_dim=l_sizes[1], name="fc2", activation=tf.nn.tanh)(
    [adj_in, o_fc1]
)

o_fc3 = lg.GraphConvLayer(output_dim=l_sizes[2], name="fc3", activation=tf.nn.tanh)(
    [adj_in, o_fc2]
)

o_fc4 = lg.GraphConvLayer(output_dim=l_sizes[3], name="fc4", activation=tf.identity)(
    [adj_in, o_fc3]
)

model = tf.keras.models.Model(inputs=[adj_in, feat_in], outputs=o_fc4)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

outputs = {}
t = time.time()
for epoch in range(300):
    with tf.GradientTape() as tape:
        logits = model((adj_norm_sparse_tensor, feat_x))
        train_loss = masked_softmax_cross_entropy(
            preds=logits, labels=y_train, mask=train_mask
        )
        train_acc = masked_accuracy(preds=logits, labels=y_train, mask=train_mask)
    grads = tape.gradient(train_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    if epoch % 50 == 0:
        val_loss = masked_softmax_cross_entropy(
            preds=logits, labels=y_val, mask=val_mask
        )
        val_acc = masked_accuracy(preds=logits, labels=y_val, mask=val_mask)

        print(
            "Epoch:",
            "%04d" % (epoch + 1),
            "train_loss=",
            "{:.5f}".format(train_loss),
            "train_acc=",
            "{:.5f}".format(train_acc),
            "val_loss=",
            "{:.5f}".format(val_loss),
            "val_acc=",
            "{:.5f}".format(val_acc),
            "time=",
            "{:.5f}".format(time.time() - t),
        )
        K.clear_session()
        interim_model = tf.keras.models.Model(inputs=[adj_in, feat_in], outputs=o_fc3)
        temp_out = interim_model((adj_norm_sparse_tensor, feat_x))
        outputs[epoch] = temp_out

node_positions = {
    o: {n: tuple(outputs[o][j]) for j, n in enumerate(nx.nodes(g))} for o in outputs
}
plot_titles = {o: "epoch {o}".format(o=o) for o in outputs}

# Two subplots, unpack the axes array immediately
f, axes = plt.subplots(nrows=2, ncols=3, sharey=True, sharex=True)

e = list(node_positions.keys())

for i, ax in enumerate(axes.flat):
    pos = node_positions[e[i]]
    ax.set_title(plot_titles[e[i]])
    nx.draw(
        g,
        cmap=plt.get_cmap("jet"),
        node_color=np.log(list(nx.get_node_attributes(g, "membership").values())),
        pos=pos,
        ax=ax,
    )

plt.show()
