"""Fase 3: la evaluación final, en las series de test.

Los agentes se entrenan de nuevo con 10 semillas que no se usaron en el ajuste
(de la 100 a la 109), en dos versiones:

- la ajustada: la configuración ganadora de la fase 2 (results/fase2_mejores.csv),
  entrenada durante 4000 episodios;
- la sin ajustar: la configuración de partida de la fase 1, con 2000 episodios.
  Comparar las dos muestra cuánto aporta el ajuste.

Después se prueba cada agente, junto con las estrategias de referencia
(aleatoria, buy & hold, momentum y contrarian), en las 100 series de test de
cada mercado. Las series son las mismas para todos, así que la comparación es
justa.

Para lanzarlo, desde la raíz del repositorio:  python scripts/fase3_resultados.py

Si los resultados ya están en results/, no se vuelve a calcular (hay que
borrarlos para repetirlo). Con 10 núcleos tarda unos 15 minutos.
"""
import time
from pathlib import Path

import pandas as pd

from rl_trader.agents import episodio_convergencia, episodio_estable, politica_dentro
from rl_trader.baselines import BASELINES
from rl_trader.evaluation import SEMILLA_TEST, evaluar, generar_series
from rl_trader.experiments import CONFIG_SIN_AJUSTAR, HIPERPARAMETROS, N_TEST, ejecutar_varios
from rl_trader.prices import GENERADORES

RESULTADOS = Path(__file__).resolve().parent.parent / "results"
# Semillas nuevas, distintas de las del ajuste (0, 1 y 2)
SEMILLAS = range(100, 110)
# Duración del entrenamiento de la versión ajustada
EPISODIOS = 4000
SIN_AJUSTAR = CONFIG_SIN_AJUSTAR
# Parámetros comunes a todos los entrenamientos: cada 200 episodios se mira la
# validación para dibujar la curva de aprendizaje
COMUN = dict(eps_inicial=1.0, evaluar_cada=200, n_validacion=100)


def configuraciones():
    """Prepara la lista de entrenamientos: cada agente, en sus dos versiones y con cada semilla.

    Devuelve parejas (versión, configuración), donde la versión es "ajustado" o
    "sin ajustar".
    """
    salida = []
    for f in pd.read_csv(RESULTADOS / "fase2_mejores.csv").itertuples():
        ajustada = {k: getattr(f, k) for k in HIPERPARAMETROS} | {"n_episodios": EPISODIOS}
        for variante, hp in (("ajustado", ajustada), ("sin ajustar", SIN_AJUSTAR)):
            for s in SEMILLAS:
                salida.append((variante, dict(mercado=f.mercado, algoritmo=f.algoritmo, seed=s, **COMUN, **hp)))
    return salida


def main():
    ruta_test = RESULTADOS / "fase3_test_por_serie.csv"
    if ruta_test.exists():
        print(f"ya existe {ruta_test.name}, no se repite")
        return
    variantes, configs = zip(*configuraciones())
    t0 = time.perf_counter()
    salidas = ejecutar_varios(list(configs), en_test=True)
    print(f"{len(configs)} entrenamientos en {(time.perf_counter() - t0) / 60:.1f} min")

    # De cada entrenamiento se guarda su resultado en cada serie de test y un
    # resumen: hiperparámetros, tiempo, cuándo convergió y la política aprendida
    test, runs, curvas = [], [], []
    for run, (variante, s) in enumerate(zip(variantes, salidas)):
        c, res = s["config"], s["resultado"]
        agente = c["algoritmo"] if variante == "ajustado" else f"{c['algoritmo']} (sin ajustar)"
        test.append(s["test"].assign(mercado=c["mercado"], agente=agente, seed=c["seed"], serie=range(N_TEST)))
        convergencia = episodio_convergencia(res.validacion)
        runs.append({
            "run": run, "mercado": c["mercado"], "agente": agente, "algoritmo": c["algoritmo"],
            "variante": variante, "seed": c["seed"], **{k: c[k] for k in HIPERPARAMETROS},
            "segundos": res.segundos,
            "episodio_convergencia": convergencia,
            "segundos_convergencia": res.validacion.set_index("episodio").loc[convergencia, "segundos"],
            "episodio_estable": episodio_estable(res.validacion),
            "politica": "".join(map(str, politica_dentro(res.Q))),
        })
        curvas.append(res.validacion.assign(run=run))

    # Las estrategias de referencia, en las mismas series de test. La aleatoria se
    # repite con las 10 semillas porque cada una da un resultado distinto; las
    # demás no tienen azar y basta con probarlas una vez.
    for mercado, generador in GENERADORES.items():
        series = generar_series(generador, N_TEST, SEMILLA_TEST)
        for nombre, fabrica in BASELINES.items():
            for s in (SEMILLAS if nombre == "aleatorio" else [None]):
                politica = fabrica(s) if nombre == "aleatorio" else fabrica()
                test.append(evaluar(politica, series).assign(mercado=mercado, agente=nombre, seed=s,
                                                             serie=range(N_TEST)))

    pd.DataFrame(runs).to_csv(RESULTADOS / "fase3_entrenamientos.csv", index=False)
    pd.concat(curvas).to_csv(RESULTADOS / "fase3_curvas.csv", index=False)
    pd.concat(test).to_csv(ruta_test, index=False)


if __name__ == "__main__":
    main()
