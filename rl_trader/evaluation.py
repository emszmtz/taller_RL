"""Cómo se generan las series de precios y cómo se mide a un agente.

Las series de entrenamiento, validación y test salen de rangos de semillas que
no se solapan. Así ninguna serie de validación ni de test aparece al entrenar,
y todos los agentes se comparan exactamente sobre las mismas series.
"""
import numpy as np
import pandas as pd

from .env import SimpleTradingEnv

# Las semillas de NumPy pueden llegar hasta 2**32 − 1 (unos 4.300 millones).
# El entrenamiento usa números por debajo de 4.000 millones, y cada uno de los
# otros conjuntos tiene su propio bloque a partir de aquí.
SEMILLA_VALIDACION = 4_000_000_000
SEMILLA_TEST = 4_100_000_000
SEMILLA_DISCRETIZADOR = 4_200_000_000

# Longitud de cada serie de precios, en pasos
N_STEPS = 500


def semilla_entrenamiento(semilla_agente, episodio):
    """Devuelve la semilla de la serie que ve un agente en un episodio de entrenamiento.

    Cada agente (según su semilla) tiene su propio bloque de 10 millones de
    semillas, una por episodio, todas por debajo de las reservadas a validación y
    test.
    """
    if not (0 <= semilla_agente < 400 and 0 <= episodio < 10_000_000):
        raise ValueError("semilla o episodio fuera del rango reservado al entrenamiento")
    return semilla_agente * 10_000_000 + episodio


def generar_series(generador, n, semilla_base, n_steps=N_STEPS):
    """Genera n series de precios con un generador, siempre las mismas.

    La serie número i usa la semilla semilla_base + i, así que dos llamadas con
    los mismos argumentos dan exactamente las mismas series.
    """
    series = []
    for i in range(n):
        np.random.seed(semilla_base + i)
        series.append(generador(n_steps=n_steps))
    return series


def ejecutar_episodio(precios, politica):
    """Hace que una política recorra una serie de precios y apunta qué pasa.

    La política es una función que recibe lo que ve el agente (tendencia,
    posición) y devuelve una acción. Devuelve el capital en cada paso, empezando
    por el inicial (1.000), y si el agente tenía el activo en cada paso.
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
    """Resume un episodio en cinco números.

    - retorno_pct: cuánto ha ganado o perdido el capital, en %. Es la
      rentabilidad, no el retorno G_t de los apuntes.
    - sharpe: la rentabilidad media por paso dividida por su volatilidad,
      anualizada como si cada paso fuera un día de mercado.
    - max_drawdown_pct: la mayor caída del capital desde un máximo anterior, en %.
    - operaciones: cuántas veces ha comprado o vendido.
    - tiempo_dentro_pct: qué parte del tiempo ha tenido el activo, en %.
    """
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
    """Prueba una política en varias series y devuelve una tabla con una fila de métricas por serie."""
    return pd.DataFrame([metricas(*ejecutar_episodio(p, politica)) for p in series])


def ic_bootstrap_pareado(x, y, n=10_000, nivel=0.95, seed=0):
    """Compara dos agentes sobre las mismas series y da un intervalo de confianza.

    x e y tienen una medida por serie para cada agente (por ejemplo, el log del
    capital final), en el mismo orden. La función calcula la diferencia media
    x − y y, por bootstrap (remuestreando las series muchas veces), su intervalo
    de confianza. Si el intervalo no incluye el 0, la diferencia es significativa.

    Devuelve (diferencia media, límite inferior, límite superior).
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    medias = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    cola = (1 - nivel) / 2
    return d.mean(), np.quantile(medias, cola), np.quantile(medias, 1 - cola)


def resumir(evaluaciones):
    """Junta las evaluaciones de varios agentes en una tabla resumen.

    Recibe un diccionario {nombre del agente: tabla de evaluar()} y devuelve una
    fila por agente con la media y la desviación típica de cada métrica. Añade la
    mediana de la rentabilidad, que es la métrica principal: en el mercado de
    ciclo, unas pocas series con resultados enormes disparan la media.
    """
    filas = {}
    for nombre, df in evaluaciones.items():
        fila = {"retorno_mediana_pct": df["retorno_pct"].median()}
        for col in df.columns:
            fila[col] = df[col].mean()
            fila[col + "_std"] = df[col].std()
        filas[nombre] = fila
    return pd.DataFrame(filas).T
