import math


def r_precision(predicted, relevant_set):
    """
    si hay R elementos relevantes,
    ¿cuantos elementos relevantes hay en los top-R predichos?
    """
    R = len(relevant_set)
    if R == 0:
        return 0.0
    hits = sum(1 for t in predicted[:R] if t in relevant_set)
    return hits / R


def ndcg(predicted, relevant_set):
    """
    Normalized Discounted Cumulative Gain
    NDCG = DCG / IDCG
    """
    if not relevant_set:
        return 0.0

    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(predicted)
        if item in relevant_set
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(len(relevant_set)))

    return dcg / idcg if idcg > 0.0 else 0.0


def clicks(predicted, relevant_set):
    """
    ¿cuantas veces tendría que resetear las top 10 predicciones para encontrar una cancion buena?
    si la cancion buena esta en las 10 primeras, 0; si esta en las 10 siguientes, 1... como máximo 51
    """
    for i, t in enumerate(predicted):
        if t in relevant_set:
            return i // 10

    return 51
