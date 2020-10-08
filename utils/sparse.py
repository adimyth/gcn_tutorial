import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def sparse_to_tuple(sparse_mx) -> tuple:
    """Convert sparse matrix to tuple representation."""
    # The zeroth element of the tuple contains the cell location of each
    # non-zero value in the sparse matrix
    # The first element of the tuple contains the value at each cell location
    # in the sparse matrix
    # The second element of the tuple contains the full shape of the sparse
    # matrix
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def tuple_to_sparsetensor(sparse_tuple: tuple) -> tf.sparse.SparseTensor:
    """Converts tuple containing coordintates, values & shape to tf.SparseTensor"""
    sparse_tensor = tf.sparse.SparseTensor(
        indices=sparse_tuple[0], values=sparse_tuple[1], dense_shape=sparse_tuple[2]
    )
    sparse_tensor = tf.cast(sparse_tensor, tf.float32)
    return sparse_tensor
