"""Fase 2: ajuste de hiperparámetros por etapas, siempre sobre series de validación.

- Etapa A: tasa de aprendizaje (α constante o decreciente) × γ.
- Etapa B: con el mejor α y γ de cada mercado y algoritmo, nº de intervalos × calendario de ε.
- Etapa C: con la mejor configuración, un entrenamiento largo para medir la convergencia.

Cada configuración se entrena con tres semillas y se puntúa con la mediana del
retorno en 100 series de validación, promediada entre semillas (a igualdad,
gana la que opera menos). El test no se toca.

Uso, desde la raíz del repositorio:  python scripts/fase2_ajuste.py
Cada etapa guarda sus resultados en results/ y, si ya existen, no se repite
(bórralos para volver a calcularla). Con 10 núcleos tarda unos 40 minutos.
"""
import time
from pathlib import Path

import pandas as pd

from rl_trader.agents import episodio_estable
from rl_trader.experiments import HIPERPARAMETROS, ejecutar_varios, elegir_mejores
from rl_trader.prices import GENERADORES

RESULTADOS = Path(__file__).resolve().parent.parent / "results"
ALGORITMOS = ("qlearning", "sarsa")
SEMILLAS = (0, 1, 2)

BASE = dict(n_episodios=2000, alpha=1.0, omega=0.6, gamma=0.9, eps_inicial=1.0, eps_final=0.01,
            frac_decaimiento=0.7, n_intervalos=10, evaluar_cada=200, n_validacion=100)
ALPHAS = [dict(alpha=0.01, omega=0.0), dict(alpha=0.001, omega=0.0),
          dict(alpha=1.0, omega=0.6), dict(alpha=1.0, omega=0.8), dict(alpha=1.0, omega=1.0)]
GAMMAS = [0.0, 0.5, 0.9, 0.99]
INTERVALOS = [4, 6, 10, 16, 20]
EPSILONS = [dict(frac_decaimiento=0.3, eps_final=0.01), dict(frac_decaimiento=0.7, eps_final=0.01),
            dict(frac_decaimiento=0.7, eps_final=0.1)]
EPISODIOS_LARGO = 6000


def correr(nombre, variantes):
    """Entrena cada variante (cambios respecto a BASE) con todas las semillas y guarda los resultados."""
    ruta_runs = RESULTADOS / f"fase2_{nombre}.csv"
    if ruta_runs.exists():
        print(f"{nombre}: ya existe {ruta_runs.name}, no se repite")
        return pd.read_csv(ruta_runs)
    configs = [BASE | v | {"seed": s} for v in variantes for s in SEMILLAS]
    t0 = time.perf_counter()
    salidas = ejecutar_varios(configs)
    print(f"{nombre}: {len(configs)} entrenamientos en {(time.perf_counter() - t0) / 60:.1f} min")

    filas, curvas = [], []
    for run, s in enumerate(salidas):
        c, res, val = s["config"], s["resultado"], s["validacion"]
        filas.append({
            "run": run, "mercado": c["mercado"], "algoritmo": c["algoritmo"], "seed": c["seed"],
            **{k: c[k] for k in HIPERPARAMETROS},
            "val_mediana": val["retorno_pct"].median(),
            "val_media": val["retorno_pct"].mean(),
            "val_sharpe": val["sharpe"].mean(),
            "val_drawdown": val["max_drawdown_pct"].mean(),
            "val_operaciones": val["operaciones"].mean(),
            "val_dentro": val["tiempo_dentro_pct"].mean(),
            "segundos": res.segundos,
            "episodio_estable": episodio_estable(res.validacion),
        })
        curvas.append(res.validacion.assign(run=run))
    runs = pd.DataFrame(filas)
    runs.to_csv(ruta_runs, index=False)
    pd.concat(curvas).to_csv(RESULTADOS / f"fase2_{nombre}_curvas.csv", index=False)
    return runs


def main():
    RESULTADOS.mkdir(exist_ok=True)
    mercados = list(GENERADORES)

    etapa_a = correr("etapa_a", [dict(mercado=m, algoritmo=a, gamma=g, **al)
                                 for m in mercados for a in ALGORITMOS for al in ALPHAS for g in GAMMAS])
    mejor_a = elegir_mejores(etapa_a)

    etapa_b = correr("etapa_b", [dict(mercado=f.mercado, algoritmo=f.algoritmo, alpha=f.alpha, omega=f.omega,
                                      gamma=f.gamma, n_intervalos=n, **e)
                                 for f in mejor_a.itertuples() for n in INTERVALOS for e in EPSILONS])
    mejor_b = elegir_mejores(etapa_b)
    mejor_b.to_csv(RESULTADOS / "fase2_mejores.csv", index=False)

    correr("etapa_c", [dict(mercado=f.mercado, algoritmo=f.algoritmo, n_episodios=EPISODIOS_LARGO,
                            **{k: getattr(f, k) for k in HIPERPARAMETROS if k != "n_episodios"})
                       for f in mejor_b.itertuples()])
    print(mejor_b.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
