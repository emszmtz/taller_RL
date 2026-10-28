"""Herramientas para lanzar muchos entrenamientos a la vez.

Cada experimento se describe con un diccionario (una "configuración"), por ejemplo:

    {"mercado": "reversion", "algoritmo": "sarsa", "gamma": 0.5, "seed": 0}

- mercado: uno de los cuatro generadores ("ciclo", "browniano", "reversion" o "regimenes").
- algoritmo: "qlearning", "sarsa" o "montecarlo".
- n_intervalos: en cuántos intervalos se divide la tendencia (10 si no se indica).
- n_validacion: cuántas series de validación usar (20 si no se indica).
- Cualquier otro parámetro de agents.entrenar (alpha, gamma, n_episodios...).
"""
import os
from concurrent.futures import ProcessPoolExecutor

from .agents import Discretizador, entrenar
from .evaluation import SEMILLA_TEST, SEMILLA_VALIDACION, evaluar, generar_series
from .prices import GENERADORES

# Cuántas series de validación se usan si no se indica otra cosa, y cuántas de test
N_VALIDACION = 20
N_TEST = 100

# Los hiperparámetros que se ajustan; juntos identifican cada configuración
HIPERPARAMETROS = ["alpha", "omega", "gamma", "n_intervalos", "frac_decaimiento", "eps_final", "n_episodios"]

# La configuración de la primera pasada (notebook 01), antes de ajustar nada
CONFIG_SIN_AJUSTAR = dict(alpha=1.0, omega=0.6, gamma=0.9, n_intervalos=10, frac_decaimiento=0.7, eps_final=0.01,
                          n_episodios=2000)


def ejecutar(config, en_test=False):
    """Entrena un agente según una configuración y lo evalúa.

    Siempre lo evalúa en las series de validación. Con en_test=True también lo
    evalúa en las 100 series de test, pero eso solo debe hacerse con
    configuraciones ya elegidas: si se eligiera mirando el test, el resultado
    dejaría de ser una medida limpia.

    Devuelve un diccionario con la configuración, el resultado del entrenamiento y
    las tablas de evaluación.
    """
    params = dict(config)
    generador = GENERADORES[params.pop("mercado")]
    algoritmo = params.pop("algoritmo")
    disc = Discretizador.ajustar(generador, n_intervalos=params.pop("n_intervalos", 10))
    validacion = generar_series(generador, params.pop("n_validacion", N_VALIDACION), SEMILLA_VALIDACION)
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
    """Igual que ejecutar(config, en_test=True). Tiene que ser una función aparte
    para poder repartirla entre varios procesos."""
    return ejecutar(config, en_test=True)


def ejecutar_varios(configs, en_test=False, procesos=None, funcion=None):
    """Lanza varias configuraciones en paralelo, una por núcleo del ordenador.

    Devuelve los resultados en el mismo orden que las configuraciones. Con
    funcion se puede usar otra forma de entrenar (por ejemplo, la de datos
    reales); si no se indica, se usa ejecutar().

    Ojo: en un script, la llamada tiene que ir dentro de
    `if __name__ == "__main__":`, porque cada proceso vuelve a importar el script.
    """
    if funcion is None:
        funcion = _ejecutar_en_test if en_test else ejecutar
    with ProcessPoolExecutor(max_workers=procesos or os.cpu_count()) as pool:
        return list(pool.map(funcion, configs))


def elegir_mejores(runs, hiperparametros=HIPERPARAMETROS):
    """Elige la mejor configuración para cada mercado y algoritmo.

    Recibe una tabla con un entrenamiento por fila: mercado, algoritmo, los
    hiperparámetros y sus resultados en validación. Promedia las semillas de cada
    configuración y se queda con la de mayor mediana en validación. Si hay
    empate, gana la que opera menos y, si sigue habiéndolo, la primera de la tabla.
    """
    agregado = (runs.groupby(["mercado", "algoritmo", *hiperparametros], sort=False, as_index=False)
                [["val_mediana", "val_sharpe", "val_operaciones"]].mean())
    agregado = agregado.sort_values(["val_mediana", "val_operaciones"], ascending=[False, True], kind="mergesort")
    return agregado.groupby(["mercado", "algoritmo"], sort=False).head(1).reset_index(drop=True)
