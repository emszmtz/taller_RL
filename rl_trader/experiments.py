"""Ejecución de experimentos, en paralelo, a partir de configuraciones sencillas.

Una configuración es un dict con 'mercado' (clave de prices.GENERADORES),
'algoritmo' ('qlearning' o 'sarsa') y, opcionalmente, 'n_intervalos' y
cualquier argumento de agents.entrenar (alpha, omega, gamma, n_episodios, seed...).
"""
import os
from concurrent.futures import ProcessPoolExecutor

from .agents import Discretizador, entrenar
from .evaluation import SEMILLA_TEST, SEMILLA_VALIDACION, evaluar, generar_series
from .prices import GENERADORES

N_VALIDACION = 20
N_TEST = 100


def ejecutar(config, en_test=False):
    """Entrena un agente y lo evalúa en validación y, si en_test, también en test.

    El test solo debe usarse con configuraciones ya elegidas en validación.
    """
    params = dict(config)
    generador = GENERADORES[params.pop("mercado")]
    algoritmo = params.pop("algoritmo")
    disc = Discretizador.ajustar(generador, n_intervalos=params.pop("n_intervalos", 10))
    validacion = generar_series(generador, N_VALIDACION, SEMILLA_VALIDACION)
    resultado = entrenar(algoritmo, generador, disc, series_validacion=validacion, **params)
    salida = {
        "config": config,
        "resultado": resultado,
        "validacion": evaluar(resultado.politica(), validacion),
    }
    if en_test:
        salida["test"] = evaluar(resultado.politica(), generar_series(generador, N_TEST, SEMILLA_TEST))
    return salida


def _ejecutar_en_test(config):
    return ejecutar(config, en_test=True)


def ejecutar_varios(configs, en_test=False, procesos=None):
    """Ejecuta varias configuraciones en paralelo (un proceso por núcleo por defecto)."""
    funcion = _ejecutar_en_test if en_test else ejecutar
    with ProcessPoolExecutor(max_workers=procesos or os.cpu_count()) as pool:
        return list(pool.map(funcion, configs))
