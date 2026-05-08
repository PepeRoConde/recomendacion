# Source - https://stackoverflow.com/a/43130239
# Posted by Dave DeCaprio
# Retrieved 2026-05-07, License - CC BY-SA 3.0

# Pero vamos que está adaptao
import tensorflow as tf
import numpy as np


def matriz_a_tensor(X):
    coo = X.tocoo()
    indices = np.asmatrix([coo.row, coo.col]).transpose()
    X = tf.sparse.SparseTensor(indices, coo.data, coo.shape)
    X.numcol = coo.shape[1]
    return X
