"""Fase 2: búsqueda de los mejores hiperparámetros, mirando solo la validación.

Probar todas las combinaciones a la vez sería demasiado lento, así que la
búsqueda se hace en tres etapas, y cada una parte de lo mejor de la anterior:

- Etapa A: se combinan varias tasas de aprendizaje α (fijas o que van bajando)
  con varios valores de γ.
- Etapa B: con el mejor α y γ de cada mercado y algoritmo, se prueba en cuántos
  intervalos dividir la tendencia y a qué ritmo bajar ε.
- Etapa C: la configuración ganadora se entrena durante mucho más tiempo (6000
  episodios) para ver a partir de cuándo deja de mejorar.

Cada configuración se entrena con tres semillas distintas. Su nota es la
mediana de la rentabilidad en 100 series de validación, promediada entre las
tres semillas; si hay empate, gana la que opera menos. Las series de test no se
usan en ningún momento.

Para lanzarlo, desde la raíz del repositorio:  python scripts/fase2_ajuste.py

Cada etapa guarda sus resultados en results/. Si ya están ahí, esa etapa no se
vuelve a calcular (hay que borrarlos para repetirla). Con 10 núcleos tarda unos
40 minutos.
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

# Configuración de partida: cada variante que se prueba cambia solo algunos de estos valores
BASE = dict(n_episodios=2000, alpha=1.0, omega=0.6, gamma=0.9, eps_inicial=1.0, eps_final=0.01,
            frac_decaimiento=0.7, n_intervalos=10, evaluar_cada=200, n_validacion=100)
# Valores de la etapa A. La tasa de aprendizaje es alpha / n**omega, donde n es
# cuántas veces se ha visitado ese estado y acción: con omega = 0 es fija, y con
# omega > 0 va bajando a medida que se repiten las visitas.
ALPHAS = [dict(alpha=0.01, omega=0.0), dict(alpha=0.001, omega=0.0),
          dict(alpha=1.0, omega=0.6), dict(alpha=1.0, omega=0.8), dict(alpha=1.0, omega=1.0)]
GAMMAS = [0.0, 0.5, 0.9, 0.99]
# Valores de la etapa B: en cuántos intervalos se divide la tendencia y cómo baja ε
INTERVALOS = [4, 6, 10, 16, 20]
EPSILONS = [dict(frac_decaimiento=0.3, eps_final=0.01), dict(frac_decaimiento=0.7, eps_final=0.01),
            dict(frac_decaimiento=0.7, eps_final=0.1)]
# Duración del entrenamiento largo de la etapa C
EPISODIOS_LARGO = 6000


def correr(nombre, variantes):
    """Entrena todas las variantes de una etapa, con las tres semillas, y guarda los resultados.

    Cada variante es un diccionario con los valores que cambian respecto a BASE.
    Se guardan dos ficheros en results/: uno con una fila por entrenamiento (sus
    hiperparámetros y cómo le fue en validación) y otro con las curvas de
    aprendizaje.
    """
    ruta_runs = RESULTADOS / f"fase2_{nombre}.csv"
    if ruta_runs.exists():
        print(f"{nombre}: ya existe {ruta_runs.name}, no se repite")
        return pd.read_csv(ruta_runs)
    configs = [BASE | v | {"seed": s} for v in variantes for s in SEMILLAS]
    t0 = time.perf_counter()
    salidas = ejecutar_varios(configs)
    print(f"{nombre}: {len(configs)} entrenamientos en {(time.perf_counter() - t0) / 60:.1f} min")

    # Una fila por entrenamiento: sus hiperparámetros y el resumen de sus resultados en validación
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

    # Etapa A: todas las combinaciones de α y γ, para cada mercado y algoritmo
    etapa_a = correr("etapa_a", [dict(mercado=m, algoritmo=a, gamma=g, **al)
                                 for m in mercados for a in ALGORITMOS for al in ALPHAS for g in GAMMAS])
    mejor_a = elegir_mejores(etapa_a)

    # Etapa B: partiendo del mejor α y γ, todas las combinaciones de intervalos y de ε
    etapa_b = correr("etapa_b", [dict(mercado=f.mercado, algoritmo=f.algoritmo, alpha=f.alpha, omega=f.omega,
                                      gamma=f.gamma, n_intervalos=n, **e)
                                 for f in mejor_a.itertuples() for n in INTERVALOS for e in EPSILONS])
    mejor_b = elegir_mejores(etapa_b)
    mejor_b.to_csv(RESULTADOS / "fase2_mejores.csv", index=False)

    # Etapa C: la configuración ganadora, con un entrenamiento mucho más largo
    correr("etapa_c", [dict(mercado=f.mercado, algoritmo=f.algoritmo, n_episodios=EPISODIOS_LARGO,
                            **{k: getattr(f, k) for k in HIPERPARAMETROS if k != "n_episodios"})
                       for f in mejor_b.itertuples()])
    print(mejor_b.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
