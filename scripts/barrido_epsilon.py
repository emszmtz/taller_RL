"""¿Cuánto debería explorar el agente al final del entrenamiento? Barrido del ε final.

Durante el entrenamiento, ε (la probabilidad de elegir una acción al azar) baja
de 1 hasta un valor final y después se queda ahí. Este script prueba varios
valores de ese ε final, solo en validación: el test no se toca.

Se hace para los mercados de reversión y de regímenes, con Q-learning y Sarsa.
Se parte de la configuración elegida en la fase 2 (results/fase2_mejores.csv),
con 4000 episodios, y solo se cambia el ε final. En lo que sigue, "desplegado"
es el ε final de esa configuración, el que se usó en los resultados de test.

Se usan 10 semillas nuevas (de la 200 a la 209). Con la misma semilla, todos los
valores de ε ven las mismas series de entrenamiento, así que se pueden comparar
por parejas. La evaluación usa las mismas 100 series de validación de siempre.

No se prueba ε final = 0: con la fórmula exponencial, ε caería a 0 justo
después del primer episodio y el agente casi no exploraría.

Para lanzarlo, desde la raíz del repositorio:  python scripts/barrido_epsilon.py

Con 10 núcleos tarda unos 12 minutos. Si results/eps_barrido.csv ya existe, no
se vuelve a entrenar. Guarda results/eps_barrido.csv (un entrenamiento por fila),
results/eps_barrido_resumen.csv y figures/eps_barrido.png.
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl_trader.evaluation import ic_bootstrap_pareado
from rl_trader.experiments import HIPERPARAMETROS, ejecutar_varios
from rl_trader.plots import COLORES_ALGORITMO, EJE, FONDO, NOMBRES_ALGORITMO, NOMBRES_MERCADO, aplicar_estilo

ROOT = Path(__file__).resolve().parent.parent
RESULTADOS, FIGURAS = ROOT / "results", ROOT / "figures"
MERCADOS = ("reversion", "regimenes")
ALGORITMOS = ("qlearning", "sarsa")
SEMILLAS = range(200, 210)
# Los valores de ε final que se prueban
EPS_FINAL = [0.001, 0.01, 0.03, 0.1, 0.3]
EPISODIOS = 4000


def entrenar_todo():
    """Entrena un agente por cada combinación de mercado, algoritmo, ε final y semilla.

    Devuelve una tabla con una fila por entrenamiento y sus resultados en
    validación. La columna "desplegado" marca las filas cuyo ε final es el de la
    configuración elegida en la fase 2.
    """
    mejores = pd.read_csv(RESULTADOS / "fase2_mejores.csv")
    mejores = mejores[mejores.mercado.isin(MERCADOS) & mejores.algoritmo.isin(ALGORITMOS)]
    configs, desplegado = [], {}
    for f in mejores.itertuples():
        base = {k: getattr(f, k) for k in HIPERPARAMETROS} | {"n_episodios": EPISODIOS}
        desplegado[(f.mercado, f.algoritmo)] = f.eps_final
        for eps in EPS_FINAL:
            for s in SEMILLAS:
                configs.append(dict(mercado=f.mercado, algoritmo=f.algoritmo, seed=s, evaluar_cada=EPISODIOS,
                                    n_validacion=100, **(base | {"eps_final": eps})))
    t0 = time.perf_counter()
    salidas = ejecutar_varios(configs)
    print(f"{len(configs)} entrenamientos en {(time.perf_counter() - t0) / 60:.1f} min")

    filas = []
    for s in salidas:
        c, v = s["config"], s["validacion"]
        filas.append({
            "mercado": c["mercado"], "algoritmo": c["algoritmo"], "seed": c["seed"], "eps_final": c["eps_final"],
            "frac_decaimiento": c["frac_decaimiento"], "desplegado": c["eps_final"] == desplegado[(c["mercado"], c["algoritmo"])],
            "val_mediana": v["retorno_pct"].median(), "val_media": v["retorno_pct"].mean(),
            "val_sharpe": v["sharpe"].mean(), "val_operaciones": v["operaciones"].mean(),
            "val_dentro": v["tiempo_dentro_pct"].mean(), "segundos": s["resultado"].segundos,
        })
    return pd.DataFrame(filas)


def resumir(runs):
    """Resume el barrido: una fila por mercado, algoritmo y ε final.

    Para cada ε final da la media de las 10 semillas y cuánto varía entre ellas, y
    lo compara con el ε desplegado semilla a semilla. Si el intervalo de confianza
    de esa diferencia no incluye el 0, se marca como significativa.
    """
    filas = []
    for (m, a), d in runs.groupby(["mercado", "algoritmo"]):
        base = d[d.desplegado].sort_values("seed")["val_mediana"].to_numpy()
        for eps, g in d.groupby("eps_final"):
            x = g.sort_values("seed")["val_mediana"].to_numpy()
            dif, lo, hi = ic_bootstrap_pareado(x, base)
            filas.append({"mercado": m, "algoritmo": a, "eps_final": eps, "desplegado": bool(g.desplegado.iloc[0]),
                          "val_mediana": x.mean(), "sd_semillas": x.std(ddof=1),
                          "minimo": x.min(), "maximo": x.max(),
                          "operaciones": g.val_operaciones.mean(), "dentro_pct": g.val_dentro.mean(),
                          "dif_vs_desplegado": dif, "ic_inf": lo, "ic_sup": hi,
                          "significativa": bool(lo > 0 or hi < 0) and not g.desplegado.iloc[0]})
    return pd.DataFrame(filas)


def figura(resumen):
    """Dibuja un panel por mercado y algoritmo: la mediana en validación según el ε final."""
    aplicar_estilo()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), sharex=True)
    pares = [(m, a) for m in MERCADOS for a in ALGORITMOS]
    for ax, (m, a) in zip(axes, pares):
        r = resumen[(resumen.mercado == m) & (resumen.algoritmo == a)].sort_values("eps_final")
        color = COLORES_ALGORITMO[a]
        ax.fill_between(r.eps_final, r.val_mediana - r.sd_semillas, r.val_mediana + r.sd_semillas,
                        color=color, alpha=0.12, lw=0)
        ax.plot(r.eps_final, r.val_mediana, color=color, marker="o", ms=5, mec=FONDO, mew=1.5)
        d = r[r.desplegado]
        ax.plot(d.eps_final, d.val_mediana, "o", ms=10, mfc="none", mec=EJE, mew=1.5)
        ax.axhline(0, color=EJE, lw=1)
        ax.set_xscale("log")
        ax.set_xticks(EPS_FINAL, [f"{e:g}" for e in EPS_FINAL])
        ax.minorticks_off()
        ax.set_title(f"{NOMBRES_MERCADO[m]} · {NOMBRES_ALGORITMO[a]}", loc="left", fontsize=9)
        ax.set_xlabel("ε final")
    axes[0].set_ylabel("mediana en validación (%)")
    fig.suptitle("Línea: media de 10 semillas; banda: ± 1 desviación típica entre semillas; círculo: ε desplegado",
                 x=0.01, ha="left", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIGURAS.mkdir(exist_ok=True)
    fig.savefig(FIGURAS / "eps_barrido.png", dpi=200)
    plt.close(fig)


def main():
    RESULTADOS.mkdir(exist_ok=True)
    ruta = RESULTADOS / "eps_barrido.csv"
    if ruta.exists():
        print(f"ya existe {ruta.name}, no se vuelve a entrenar")
        runs = pd.read_csv(ruta)
    else:
        runs = entrenar_todo()
        runs.to_csv(ruta, index=False)
    resumen = resumir(runs)
    resumen.to_csv(RESULTADOS / "eps_barrido_resumen.csv", index=False)
    figura(resumen)
    with pd.option_context("display.width", 220, "display.float_format", lambda x: f"{x:.1f}"):
        tabla = resumen.drop(columns=["minimo", "maximo", "ic_inf", "ic_sup"])
        print(tabla.assign(eps_final=tabla.eps_final.map("{:g}".format)).to_string(index=False))


if __name__ == "__main__":
    main()
