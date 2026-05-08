import argparse
import tensorflow as tf
from tqdm import tqdm

from src.utils import carga_o_construye_matriz, matriz_a_tensor, inicializa_S

parser = argparse.ArgumentParser(
    prog="Sistemas de Recomendación - Tercera Iteración\nJose Romero Conde & Marcos Grobas Martínez"
)
parser.add_argument("-m", dest="modo", default="SSLIM", choices="FISM")
parser.add_argument("-e", dest="EPOCHS", default=10)
parser.add_argument("-lr", dest="LEARNING_RATE", default=0.01)
parser.add_argument("-td", dest="TRAIN_DIR", default="data/dataset/train/")
parser.add_argument("-dd", dest="DATA_DIR", default="data/")
parser.add_argument("-mj", dest="MAX_JSONS", type=int, default=50)
parser.add_argument("-mp", dest="MAX_PLAYLISTS", type=int, default=200)

args = parser.parse_args()


@tf.function
def train_step(R, S, optiomizer, lmb1=0.5, lmb2=0.5):
    with tf.GradientTape() as t:
        RS = tf.sparse.sparse_dense_matmul(R, S)
        residuo = tf.norm(tf.sparse.add(R, -RS), ord=2)
        l1, l2 = tf.norm(S, ord=1), tf.norm(S, ord=2)
        perdida = residuo + lmb1 * l1 + lmb2 * l2

    g = t.gradient(perdida, S)
    optimizer.apply_gradients([(g, S)])
    return perdida


R, _1, _2, _3 = carga_o_construye_matriz(
    args.TRAIN_DIR, args.DATA_DIR, args.MAX_JSONS, args.MAX_PLAYLISTS
)  # devuelve csr de scipy
del _1, _2, _3

R = matriz_a_tensor(R)  # ambas dispersas, claro

S = inicializa_S(n=R.numcol)

optimizer = tf.keras.optimizers.SGD(
    learning_rate=args.LEARNING_RATE,
)

print("Entramos no bucle de adestramento, isto vai levar...")
for i in tqdm(range(args.EPOCHS)):
    perdida = train_step(R, S, optimizer)
    print(f"Época {i} || Pérdida {perdida.numpy().sum()}")
