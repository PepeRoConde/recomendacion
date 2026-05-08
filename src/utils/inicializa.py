import tensorflow as tf
import time


def inicializa(n, dim=20, matriz="S"):
    # TODO: poner un mensaje de error que explique como se usa
    assert (matriz == "S") | (matriz == "QT")
    antes = time.time()
    if matriz == "S":
        S = tf.Variable(tf.random_normal_initializer(stddev=0.05)(shape=[n, n]))
    else:
        Q = tf.Variable(tf.random_normal_initializer(stddev=0.05)(shape=[n, dim]))
        T = tf.Variable(tf.random_normal_initializer(stddev=0.05)(shape=[dim, n]))
    print(f"[{matriz}] Tardouse {time.time() - antes:.3f}s en inicializar {matriz}")
    if matriz == "S":
        return S
    else:
        return Q, T
