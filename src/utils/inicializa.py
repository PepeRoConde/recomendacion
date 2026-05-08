import tensorflow as tf
import time


def inicializa(n, dim=20, matriz="S", positiva=False):
    # TODO: poner un mensaje de error que explique como se usa
    assert (matriz == "S") | (matriz == "QT")
    antes = time.time()
    if positiva:
        constraint = tf.keras.constraints.NonNeg()
    else:
        constraint = None
    # [articulo SSLIM] The non- negativity constraint is applied on S such that the
    # learned S corresponds to positive aggregations over items.
    if matriz == "S":
        S = tf.Variable(
            tf.random_normal_initializer(stddev=0.05)(shape=[n, n]),
            constraint=constraint,
        )
    else:
        Q = tf.Variable(
            tf.random_normal_initializer(stddev=0.05)(shape=[n, dim]),
            constraint=constraint,
        )
        T = tf.Variable(
            tf.random_normal_initializer(stddev=0.05)(shape=[dim, n]),
            constraint=constraint,
        )
    print(f"[{matriz}] Tardouse {time.time() - antes:.3f}s en inicializar {matriz}")
    if matriz == "S":
        return S
    else:
        return Q, T
