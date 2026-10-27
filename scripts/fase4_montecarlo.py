"""Fase 4 (extra): control Monte Carlo on-policy (slides 2) frente a Q-learning y Sarsa.

Mismo protocolo que las fases 2 y 3, para que la comparación sea justa:

- Etapa A (validación): tasa de aprendizaje × γ × primera visita / todas las visitas.
- Etapa B (validación): con lo mejor de A, nº de intervalos × calendario de ε.
- Test: 10 semillas nuevas (100–109) y 4000 episodios, sobre las mismas 100 series
  de test que el resto de agentes:
  - "montecarlo": la configuración elegida en validación;
  - "montecarlo (config. sarsa)": los hiperparámetros elegidos para Sarsa en la fase 2.
    Sarsa es el equivalente TD on-policy de este Monte Carlo: con todo lo demás igual,
    solo cambia el objetivo (retorno real G frente a R + γ Q(S', A')).

Uso, desde la raíz del repositorio:  python scripts/fase4_montecarlo.py
Cada etapa guarda sus resultados en results/fase4_*.csv y, si ya existen, no se
repite (bórralos para volver a calcularla).
"""
import time
from pathlib import Path

import pandas as pd

from rl_trader.agents import episodio_convergencia, episodio_estable, politica_dentro
from rl_trader.experiments import HIPERPARAMETROS, N_TEST, ejecutar_varios, elegir_mejores
from rl_trader.prices import GENERADORES

RESULTADOS = Path(__file__).resolve().parent.parent / "results"
HP_MC = HIPERPARAMETROS + ["primera_visita"]
SEMILLAS_AJUSTE = (0, 1, 2)
SEMILLAS_TEST = range(100, 110)
EPISODIOS_TEST = 4000

# Como en scripts/fase2_ajuste.py
BASE = dict(algoritmo="montecarlo", n_episodios=2000, alpha=1.0, omega=0.6, gamma=0.9, eps_inicial=1.0,
            eps_final=0.01, frac_decaimiento=0.7, n_intervalos=10, evaluar_cada=200, n_validacion=100,
            primera_visita=False)
ALPHAS = [dict(alpha=0.01, omega=0.0), dict(alpha=0.001, omega=0.0),
          dict(alpha=1.0, omega=0.6), dict(alpha=1.0, omega=0.8), dict(alpha=1.0, omega=1.0)]
GAMMAS = [0.0, 0.5, 0.9, 0.99]
VISITAS = [False, True]
INTERVALOS = [4, 6, 10, 16, 20]
EPSILONS = [dict(frac_decaimiento=0.3, eps_final=0.01), dict(frac_decaimiento=0.7, eps_final=0.01),
            dict(frac_decaimiento=0.7, eps_final=0.1)]


def ajustar(nombre, variantes):
    """Entrena cada variante (cambios respecto a BASE) con las semillas de ajuste."""
    ruta = RESULTADOS / f"fase4_mc_{nombre}.csv"
    if ruta.exists():
        print(f"{nombre}: ya existe {ruta.name}, no se repite")
        return pd.read_csv(ruta)
    configs = [BASE | v | {"seed": s} for v in variantes for s in SEMILLAS_AJUSTE]
    t0 = time.perf_counter()
    salidas = ejecutar_varios(configs)
    print(f"{nombre}: {len(configs)} entrenamientos en {(time.perf_counter() - t0) / 60:.1f} min")

    filas, curvas = [], []
    for run, s in enumerate(salidas):
        c, res, val = s["config"], s["resultado"], s["validacion"]
        filas.append({
            "run": run, "mercado": c["mercado"], "algoritmo": c["algoritmo"], "seed": c["seed"],
            **{k: c[k] for k in HP_MC},
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
    runs.to_csv(ruta, index=False)
    pd.concat(curvas).to_csv(RESULTADOS / f"fase4_mc_{nombre}_curvas.csv", index=False)
    return runs


def evaluar_en_test(mejor_mc):
    ruta_test = RESULTADOS / "fase4_mc_test_por_serie.csv"
    if ruta_test.exists():
        print(f"test: ya existe {ruta_test.name}, no se repite")
        return
    sarsa = pd.read_csv(RESULTADOS / "fase2_mejores.csv").query("algoritmo == 'sarsa'")
    variantes, configs = [], []
    for agente, tabla in (("montecarlo", mejor_mc), ("montecarlo (config. sarsa)", sarsa)):
        for f in tabla.itertuples():
            hp = {k: getattr(f, k) for k in HIPERPARAMETROS} | {"n_episodios": EPISODIOS_TEST}
            hp["primera_visita"] = bool(getattr(f, "primera_visita", False))
            for s in SEMILLAS_TEST:
                variantes.append(agente)
                configs.append(BASE | hp | {"mercado": f.mercado, "seed": s})
    t0 = time.perf_counter()
    salidas = ejecutar_varios(configs, en_test=True)
    print(f"test: {len(configs)} entrenamientos en {(time.perf_counter() - t0) / 60:.1f} min")

    test, runs, curvas = [], [], []
    for run, (agente, s) in enumerate(zip(variantes, salidas)):
        c, res = s["config"], s["resultado"]
        test.append(s["test"].assign(mercado=c["mercado"], agente=agente, seed=c["seed"], serie=range(N_TEST)))
        convergencia = episodio_convergencia(res.validacion)
        runs.append({
            "run": run, "mercado": c["mercado"], "agente": agente, "algoritmo": c["algoritmo"],
            "seed": c["seed"], **{k: c[k] for k in HP_MC},
            "segundos": res.segundos,
            "episodio_convergencia": convergencia,
            "segundos_convergencia": res.validacion.set_index("episodio").loc[convergencia, "segundos"],
            "episodio_estable": episodio_estable(res.validacion),
            "politica": "".join(map(str, politica_dentro(res.Q))),
        })
        curvas.append(res.validacion.assign(run=run))
    pd.DataFrame(runs).to_csv(RESULTADOS / "fase4_mc_entrenamientos.csv", index=False)
    pd.concat(curvas).to_csv(RESULTADOS / "fase4_mc_curvas.csv", index=False)
    pd.concat(test).to_csv(ruta_test, index=False)


def main():
    RESULTADOS.mkdir(exist_ok=True)
    mercados = list(GENERADORES)

    etapa_a = ajustar("etapa_a", [dict(mercado=m, gamma=g, primera_visita=v, **al)
                                  for m in mercados for al in ALPHAS for g in GAMMAS for v in VISITAS])
    mejor_a = elegir_mejores(etapa_a, HP_MC)

    etapa_b = ajustar("etapa_b", [dict(mercado=f.mercado, alpha=f.alpha, omega=f.omega, gamma=f.gamma,
                                       primera_visita=bool(f.primera_visita), n_intervalos=n, **e)
                                  for f in mejor_a.itertuples() for n in INTERVALOS for e in EPSILONS])
    mejor_b = elegir_mejores(etapa_b, HP_MC)
    mejor_b.to_csv(RESULTADOS / "fase4_mc_mejores.csv", index=False)
    print(mejor_b.round(3).to_string(index=False))

    evaluar_en_test(mejor_b)


if __name__ == "__main__":
    main()
