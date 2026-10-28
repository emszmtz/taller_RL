"""Estilo común para que todas las figuras se vean igual.

Colores: azul para Q-learning, naranja para Sarsa y aqua para Monte Carlo
(elegidos para que se distingan también con daltonismo), grises para textos y
ejes, y una escala de azules para los mapas de calor.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_2, GRIS, REJILLA, EJE, FONDO = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
AZULES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SECUENCIAL = LinearSegmentedColormap.from_list("azules", AZULES)

NOMBRES_MERCADO = {"ciclo": "1. Tendencia + ciclo", "browniano": "2. Browniano",
                   "reversion": "3. Reversión a la media", "regimenes": "4. Regímenes"}
NOMBRES_ALGORITMO = {"qlearning": "Q-learning", "sarsa": "Sarsa", "montecarlo": "Monte Carlo"}
COLORES_ALGORITMO = {"qlearning": AZUL, "sarsa": NARANJA, "montecarlo": AQUA}


def aplicar_estilo():
    """Aplica el estilo común a matplotlib. Basta con llamarla una vez al principio del notebook."""
    plt.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
        "axes.edgecolor": EJE, "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA,
        "xtick.color": GRIS, "ytick.color": GRIS, "axes.grid": True, "grid.color": REJILLA,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "font.size": 9, "axes.titlesize": 10,
        "legend.frameon": False,
    })


def compacto(x, _=None):
    """Escribe los números grandes de los ejes de forma corta: 150000 → 150K, 2000000 → 2M."""
    if abs(x) >= 1e6:
        return f"{x / 1e6:g}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:g}K"
    return f"{x:g}"


def formato_log(eje):
    """Pone etiquetas legibles en un eje con escala logarítmica.

    Quita la notación científica (3×10⁰ pasa a ser 3). Si el rango del eje es
    estrecho (menos de 5 veces entre el mínimo y el máximo), añade también marcas
    intermedias para que no quede una sola etiqueta. Hay que llamarla después de
    dibujar, y recibe ax.xaxis o ax.yaxis.
    """
    lo, hi = sorted(eje.get_view_interval())
    eje.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    if hi / lo < 5:
        intermedias = [0.5, 0.6, 0.7, 0.8, 0.9, 1.2, 1.5, 2, 3, 4]
        eje.set_minor_locator(FixedLocator([v for v in intermedias if lo <= v <= hi]))
        eje.set_minor_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    else:
        eje.set_minor_formatter(NullFormatter())


def corto(x):
    """Escribe un valor con 3 cifras como mucho, para anotarlo en una figura o tabla: 81234 → 81K, 26.64 → 26.6."""
    if abs(x) >= 1e4:
        return f"{x / 1e3:.0f}K"
    if abs(x) >= 100:
        return f"{x:.0f}"
    return f"{x:.1f}"


def mapa_calor(ax, tabla, destacar=None):
    """Dibuja una tabla de números como mapa de calor, con el valor escrito en cada casilla.

    El color es relativo dentro del panel: el más oscuro es el mejor valor de esa
    tabla, no un valor absoluto. Con destacar=(fila, columna) se recuadra una
    casilla, por ejemplo la configuración elegida.
    """
    v = tabla.to_numpy(float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    rel = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    ax.imshow(rel, cmap=SECUENCIAL, vmin=0, vmax=1, aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, corto(v[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if rel[i, j] > 0.55 else TINTA)
    ax.set_xticks(range(v.shape[1]), [str(c) for c in tabla.columns])
    ax.set_yticks(range(v.shape[0]), [str(r) for r in tabla.index])
    ax.tick_params(length=0)
    ax.grid(False)
    for lado in ax.spines.values():
        lado.set_visible(False)
    for k in range(1, v.shape[0]):
        ax.axhline(k - 0.5, color=FONDO, lw=2)
    for k in range(1, v.shape[1]):
        ax.axvline(k - 0.5, color=FONDO, lw=2)
    if destacar is not None:
        # Recuadro con doble borde (oscuro por fuera y claro por dentro), para que
        # se vea igual sobre casillas claras y oscuras
        i, j = destacar
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=TINTA, lw=2.5))
        ax.add_patch(Rectangle((j - 0.42, i - 0.42), 0.84, 0.84, fill=False, ec=FONDO, lw=1.5))
