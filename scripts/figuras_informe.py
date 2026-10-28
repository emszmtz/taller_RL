"""Dibuja las figuras del informe, en tamaño pequeño para que quepan en una página A4.

Lee los resultados de las fases 3 y 4 (results/fase3_*.csv, results/fase4_mc_*.csv) y guarda en figures/:
- informe_capital.png: con cuánto capital acaba cada agente en cada mercado, como múltiplo del inicial.
- informe_politicas.png: qué ha aprendido Sarsa (en cada estado, en qué fracción de las semillas decide estar dentro).
- informe_curvas.png: curvas de aprendizaje en validación de Q-learning, Sarsa y Monte Carlo.

Para lanzarlo, desde la raíz del repositorio:  python scripts/figuras_informe.py  (tarda unos segundos)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from rl_trader.agents import Discretizador
from rl_trader.plots import AQUA, AZUL, EJE, GRIS, NARANJA, SECUENCIAL, TINTA_2, aplicar_estilo
from rl_trader.prices import GENERADORES

ROOT = Path(__file__).resolve().parent.parent
RESULTADOS, FIGURAS = ROOT / "results", ROOT / "figures"
BLANCO = "white"
ANCHO = 6.95  # en pulgadas: el ancho del texto en un A4 con márgenes de 1,6 cm
TITULOS = {"ciclo": "1. Tendencia + ciclo", "browniano": "2. Browniano",
           "reversion": "3. Reversión", "regimenes": "4. Regímenes"}
MERCADOS = list(TITULOS)
TITULOS_CORTOS = {"ciclo": "1. Ciclo", "browniano": "2. Browniano", "reversion": "3. Reversión",
                  "regimenes": "4. Regímenes"}
AGENTES = {"aleatorio": "Aleatorio", "buy_and_hold": "Buy & hold", "momentum": "Momentum",
           "contrarian": "Contrarian", "qlearning": "Q-learning", "sarsa": "Sarsa", "montecarlo": "Monte Carlo"}
COLORES = {"qlearning": AZUL, "sarsa": NARANJA, "montecarlo": AQUA}


def estilo_compacto():
    """El estilo común, con la letra más pequeña para que las figuras quepan en el informe."""
    aplicar_estilo()
    # Fondo blanco: sobre papel, el casi blanco de pantalla se ve como un recuadro gris
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.labelsize": 7.5, "lines.linewidth": 1.5, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})


def coma(x):
    """Escribe un número con coma decimal, como en el texto del informe: 1.5 → "1,5"."""
    return f"{x:g}".replace(".", ",")


def marcas_log(ax):
    """Elige qué marcas poner en un eje x logarítmico y cómo escribirlas.

    Si el eje abarca mucho (más de 30 veces entre el mínimo y el máximo), pone una
    marca en cada potencia de 10; si abarca poco, pone marcas intermedias como 0,8
    o 1,5.
    """
    lo, hi = ax.get_xlim()
    if hi / lo > 30:
        candidatas = [0.01, 0.1, 1, 10, 100, 1000, 10000]
        etiqueta = lambda x, _: f"{x / 1000:g}K" if x >= 1000 else coma(x)
    else:
        candidatas = [0.6, 0.8, 1, 1.2, 1.5, 2]
        etiqueta = lambda x, _: coma(x)
    ax.xaxis.set_major_locator(FixedLocator([c for c in candidatas if lo <= c <= hi]))
    ax.xaxis.set_major_formatter(FuncFormatter(etiqueta))
    ax.xaxis.set_minor_formatter(NullFormatter())


def figura_capital(test):
    """Figura 1: el capital final de cada agente en cada mercado, en las series de test.

    El punto es la mediana y la barra va del percentil 25 al 75. El eje está en
    escala logarítmica y la línea vertical marca el capital inicial (×1).
    """
    fig, axes = plt.subplots(1, 4, figsize=(ANCHO, 1.6), sharey=True, gridspec_kw={"width_ratios": [1.3, 1, 1, 1]})
    y = np.arange(len(AGENTES))[::-1]
    for ax, m in zip(axes, MERCADOS):
        ax.axvline(1, color=EJE, lw=0.8)
        for yi, agente in zip(y, AGENTES):
            multiplo = np.exp(test[(test.mercado == m) & (test.agente == agente)]["log_capital"])
            p25, p50, p75 = np.percentile(multiplo, [25, 50, 75])
            color = COLORES.get(agente, GRIS)
            ax.plot([p25, p75], [yi, yi], color=color, lw=1.5, solid_capstyle="round")
            ax.plot(p50, yi, "o", ms=5, color=color, mec=BLANCO, mew=1.5)
        ax.set_xscale("log")
        marcas_log(ax)
        ax.set_title(TITULOS[m], loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, list(AGENTES.values()))
    for ax in axes:
        ax.set_xlabel("capital final (× inicial)", labelpad=1)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIGURAS / "informe_capital.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def efectiva(cadena):
    """Convierte una política guardada (cadena de ceros y unos) en la que de verdad se aplica.

    Si en ningún estado sin el activo decide comprar, el agente nunca llega a
    tenerlo, así que lo aprendido para los estados con el activo no se usa nunca:
    se pone a 0 (fuera) para que la figura no lo muestre.
    """
    v = np.array(list(cadena), dtype=int)
    if not v[0::2].any():
        v[1::2] = 0
    return v


def figura_politicas(runs, agente="sarsa"):
    """Figura 2: lo que ha aprendido el agente (Sarsa, si no se indica otro) en cada mercado.

    Cada casilla es un estado: la fila es el intervalo de tendencia y la columna
    dice si tiene el activo ("sin" o "con"). El color indica en qué fracción de las
    10 semillas la política decide estar dentro en ese estado.
    """
    # Se dibuja a su tamaño final (78% del ancho de texto) para que la letra impresa sea la real
    fig, axes = plt.subplots(1, 4, figsize=(0.78 * ANCHO, 1.45), gridspec_kw={"wspace": 0.9})
    for ax, m in zip(axes, MERCADOS):
        r = runs[(runs.mercado == m) & (runs.agente == agente)]
        n = int(r["n_intervalos"].iloc[0])
        dentro = np.mean([efectiva(p) for p in r["politica"]], axis=0).reshape(n, 2)
        im = ax.imshow(dentro, cmap=SECUENCIAL, vmin=0, vmax=1, aspect="auto", origin="lower")
        cortes = Discretizador.ajustar(GENERADORES[m], n_intervalos=n).cortes
        paso = 1 if n <= 10 else 3
        posiciones = [k for k in range(1, n) if (k - n // 2) % paso == 0]
        ax.set_yticks(np.array(posiciones) - 0.5, [f"{100 * cortes[k - 1]:+.1f} %".replace(".", ",") for k in posiciones])
        ax.set_xticks([0, 1], ["sin", "con"])
        ax.tick_params(length=0)
        ax.grid(False)
        for lado in ax.spines.values():
            lado.set_visible(False)
        for k in range(1, n):
            ax.axhline(k - 0.5, color=BLANCO, lw=0.6 if n > 10 else 1.5)
        ax.axvline(0.5, color=BLANCO, lw=1.5)
        ax.set_title(TITULOS_CORTOS[m], loc="left")
    axes[0].set_ylabel("tendencia")
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.03, aspect=20)
    cb.set_label("semillas dentro", color=TINTA_2)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: coma(x)))
    cb.outline.set_visible(False)
    fig.savefig(FIGURAS / "informe_politicas.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def figura_curvas(curvas, runs):
    """Mediana en validación (media entre semillas) y episodio de convergencia (punto)."""
    fig, axes = plt.subplots(1, 4, figsize=(ANCHO, 1.5))
    for ax, m in zip(axes, MERCADOS):
        ax.axhline(0, color=EJE, lw=0.8)
        for agente in ("qlearning", "sarsa", "montecarlo"):
            media = curvas[(curvas.mercado == m) & (curvas.agente == agente)].groupby("episodio")["retorno_mediana_pct"].mean()
            ax.plot(media.index, media.values, color=COLORES[agente], label=AGENTES[agente])
            conv = runs[(runs.mercado == m) & (runs.agente == agente)]["episodio_convergencia"].median()
            ep = media.index[np.argmin(np.abs(media.index - conv))]
            ax.plot(ep, media.loc[ep], "o", ms=5, color=COLORES[agente], mec=BLANCO, mew=1.2)
        ax.set_title(TITULOS[m], loc="left")
        ax.set_xticks([0, 2000, 4000], ["0", "2000", "4000"])
        ax.set_xlabel("episodios", labelpad=1)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}K" if abs(x) >= 1000 else coma(x)))
    axes[0].set_ylabel("mediana val. (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=3, fontsize=7,
               handlelength=1.5, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.8)
    fig.savefig(FIGURAS / "informe_curvas.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    estilo_compacto()
    test = pd.concat([pd.read_csv(RESULTADOS / "fase3_test_por_serie.csv"),
                      pd.read_csv(RESULTADOS / "fase4_mc_test_por_serie.csv")])
    # El capital final como múltiplo del inicial, en logaritmo
    test["log_capital"] = np.log1p(test["retorno_pct"] / 100)
    runs = pd.read_csv(RESULTADOS / "fase3_entrenamientos.csv", dtype={"politica": str})
    figura_capital(test)
    figura_politicas(runs)
    runs_mc = pd.read_csv(RESULTADOS / "fase4_mc_entrenamientos.csv", dtype={"politica": str})
    runs_mc["run"] += 10_000
    curvas = pd.concat([pd.read_csv(RESULTADOS / "fase3_curvas.csv"),
                        pd.read_csv(RESULTADOS / "fase4_mc_curvas.csv").assign(run=lambda d: d["run"] + 10_000)])
    todos = pd.concat([runs, runs_mc], ignore_index=True)
    figura_curvas(curvas.merge(todos[["run", "mercado", "agente"]], on="run"), todos)
    print("figuras del informe guardadas en", FIGURAS)


if __name__ == "__main__":
    main()
