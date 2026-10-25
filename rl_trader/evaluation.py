"""Conjuntos de series (entrenamiento, validación y test) y métricas de evaluación.

Las series se generan con semillas de rangos disjuntos, así que ninguna serie
de validación o de test se ve durante el entrenamiento, y todos los agentes se
evalúan exactamente sobre las mismas series (comparación pareada).
"""
import numpy as np
import pandas as pd

from .env import SimpleTradingEnv

# np.random.seed admite semillas hasta 2**32 - 1. El entrenamiento usa
# semilla_agente * 10_000_000 + episodio, por debajo de 4_000_000_000.
SEMILLA_VALIDACION = 4_000_000_000
SEMILLA_TEST = 4_100_000_000
SEMILLA_DISCRETIZADOR = 4_200_000_000

N_STEPS = 500


def semilla_entrenamiento(semilla_agente, episodio):
    if not (0 <= semilla_agente < 400 and 0 <= episodio < 10_000_000):
        raise ValueError("semilla o episodio fuera del rango reservado al entrenamiento")
    return semilla_agente * 10_000_000 + episodio


def generar_series(generador, n, semilla_base, n_steps=N_STEPS):
    """n series del generador; la serie i usa la semilla semilla_base + i."""
    series = []
    for i in range(n):
        np.random.seed(semilla_base + i)
        series.append(generador(n_steps=n_steps))
    return series


def ejecutar_episodio(precios, politica):
    """Recorre una serie con una política obs -> acción.

    Devuelve los balances (incluido el inicial) y la posición mantenida en cada paso.
    """
    env = SimpleTradingEnv(precios)
    obs, _ = env.reset()
    balances, posiciones = [env.balance], []
    done = False
    while not done:
        obs, _, done, _, _ = env.step(politica(obs))
        balances.append(env.balance)
        posiciones.append(env.position)
    return np.array(balances), np.array(posiciones)


def metricas(balances, posiciones):
    rets = balances[1:] / balances[:-1] - 1
    sd = rets.std()
    return {
        "retorno_pct": 100 * (balances[-1] / balances[0] - 1),
        # Anualizado suponiendo que cada paso es un día de mercado
        "sharpe": np.sqrt(252) * rets.mean() / sd if sd > 0 else 0.0,
        "max_drawdown_pct": 100 * np.max(1 - balances / np.maximum.accumulate(balances)),
        "operaciones": int(np.abs(np.diff(posiciones, prepend=0)).sum()),
        "tiempo_dentro_pct": 100 * posiciones.mean(),
    }


def evaluar(politica, series):
    """Una fila de métricas por serie."""
    return pd.DataFrame([metricas(*ejecutar_episodio(p, politica)) for p in series])


def ic_bootstrap_pareado(x, y, n=10_000, nivel=0.95, seed=0):
    """Diferencia media x - y entre observaciones pareadas (la misma serie de test)
    y su intervalo de confianza bootstrap. Devuelve (media, inferior, superior)."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    medias = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    cola = (1 - nivel) / 2
    return d.mean(), np.quantile(medias, cola), np.quantile(medias, 1 - cola)


def resumir(evaluaciones):
    """{nombre: DataFrame de evaluar()} -> tabla con media y desviación de cada métrica.

    Incluye también la mediana del retorno: en mercados muy volátiles (el de
    ciclo) unas pocas series con retornos enormes dominan la media.
    """
    filas = {}
    for nombre, df in evaluaciones.items():
        fila = {"retorno_mediana_pct": df["retorno_pct"].median()}
        for col in df.columns:
            fila[col] = df[col].mean()
            fila[col + "_std"] = df[col].std()
        filas[nombre] = fila
    return pd.DataFrame(filas).T
