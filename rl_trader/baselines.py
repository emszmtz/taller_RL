"""Políticas de referencia: obs = (tendencia, posición) -> acción.

Acciones: 0 = vender, 1 = mantener, 2 = comprar. Comprar con el activo ya en
cartera, o vender sin él, no hace nada en el entorno.
"""
import numpy as np


def aleatoria(seed=0):
    """La del enunciado: una acción al azar en cada paso."""
    rng = np.random.default_rng(seed)
    return lambda obs: int(rng.integers(3))


def buy_and_hold(obs):
    """Compra en el primer paso y ya no vende."""
    return 2


def momentum(obs):
    """Dentro con tendencia positiva, fuera con tendencia negativa."""
    return 2 if obs[0] > 0 else 0


def contrarian(obs):
    """Dentro con tendencia negativa, fuera con tendencia positiva."""
    return 2 if obs[0] < 0 else 0


BASELINES = {
    "aleatorio": aleatoria,
    "buy_and_hold": lambda: buy_and_hold,
    "momentum": lambda: momentum,
    "contrarian": lambda: contrarian,
}
