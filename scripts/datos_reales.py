"""Extra: agentes entrenados y evaluados con precios reales de EODHD.

Activos: S&P 500 (SPY, con dividendos), IBEX 35, Bitcoin y EUR/USD, con precios
diarios ajustados.
- Entrenamiento: ventanas al azar de 500 días hasta 2018 (Bitcoin, desde 2014).
- Test: desde 2019 hasta el último dato disponible, como un único episodio.

Con datos reales no se ajustan hiperparámetros: solo hay una historia de cada
activo y cualquier elección sobre ella sería sobreajuste. Se prueban dos
configuraciones fijadas de antemano, con 10 semillas, para Q-learning y Sarsa:
la sin ajustar de la fase 1 y la ajustada en la fase 2 para el mercado de
regímenes, el simulado más parecido a un mercado real.

La primera vez necesita la clave de EODHD en .env; después usa la copia de data/.
En results/reales_*.csv solo se guardan fechas, métricas y políticas, nunca precios.

Uso, desde la raíz del repositorio:  python scripts/datos_reales.py
Con 10 núcleos tarda unos 10 minutos.
"""
import json
import time
from pathlib import Path

import pandas as pd

from rl_trader.baselines import BASELINES
from rl_trader.evaluation import ejecutar_episodio, metricas
from rl_trader.experiments import CONFIG_SIN_AJUSTAR, HIPERPARAMETROS, ejecutar_varios
from rl_trader.real import ACTIVOS, FIN_ENTRENAMIENTO, INICIO_TEST, cargar, dividir, ejecutar_real

RESULTADOS = Path(__file__).resolve().parent.parent / "results"
SEMILLAS = range(100, 110)
EPISODIOS = 4000


def configuraciones(datos):
    mejores = pd.read_csv(RESULTADOS / "fase2_mejores.csv")
    regimenes = {f.algoritmo: {k: getattr(f, k) for k in HIPERPARAMETROS} | {"n_episodios": EPISODIOS}
                 for f in mejores[mejores.mercado == "regimenes"].itertuples()}
    configs = []
    for activo, (entrenamiento, test) in datos.items():
        for algoritmo in ("qlearning", "sarsa"):
            for variante, hp in (("sin ajustar", CONFIG_SIN_AJUSTAR), ("regímenes", regimenes[algoritmo])):
                for s in SEMILLAS:
                    configs.append(dict(activo=activo, algoritmo=algoritmo, variante=variante, seed=s,
                                        precios_entrenamiento=entrenamiento, precios_test=test, **hp))
    return configs


def main():
    ruta = RESULTADOS / "reales_resultados.csv"
    if ruta.exists():
        print(f"ya existe {ruta.name}, no se repite")
        return

    datos, periodos = {}, []
    for activo, info in ACTIVOS.items():
        serie = cargar(activo)
        entrenamiento, test = dividir(serie)
        datos[activo] = (entrenamiento, test)
        periodos.append({
            "activo": activo, "ticker": info["ticker"], "nombre": info["nombre"],
            "inicio_entrenamiento": serie.index[0].date(), "fin_entrenamiento": serie[:FIN_ENTRENAMIENTO].index[-1].date(),
            "dias_entrenamiento": len(entrenamiento),
            "inicio_test": serie[INICIO_TEST:].index[0].date(), "fin_test": serie.index[-1].date(),
            "dias_test": len(test),
        })

    configs = configuraciones(datos)
    t0 = time.perf_counter()
    salidas = ejecutar_varios(configs, funcion=ejecutar_real)
    print(f"{len(configs)} entrenamientos en {(time.perf_counter() - t0) / 60:.1f} min")

    filas, politicas = [], []
    for s in salidas:
        clave = {"activo": s["activo"], "agente": s["algoritmo"], "variante": s["variante"], "seed": s["seed"]}
        for periodo in ("entrenamiento", "test"):
            filas.append(clave | {"periodo": periodo} | s[periodo])
        politicas.append(clave | {"segundos": s["segundos"], "politica": s["politica"],
                                  "cortes": json.dumps(s["cortes"])})

    # Políticas de referencia en los mismos periodos
    for activo, (entrenamiento, test) in datos.items():
        for periodo, precios in (("entrenamiento", entrenamiento), ("test", test)):
            for nombre, fabrica in BASELINES.items():
                for s in (SEMILLAS if nombre == "aleatorio" else [None]):
                    politica = fabrica(s) if nombre == "aleatorio" else fabrica()
                    filas.append({"activo": activo, "agente": nombre, "variante": "", "seed": s, "periodo": periodo}
                                 | metricas(*ejecutar_episodio(precios, politica)))

    pd.DataFrame(periodos).to_csv(RESULTADOS / "reales_periodos.csv", index=False)
    pd.DataFrame(politicas).to_csv(RESULTADOS / "reales_politicas.csv", index=False)
    pd.DataFrame(filas).to_csv(ruta, index=False)


if __name__ == "__main__":
    main()
