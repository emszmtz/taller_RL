"""Estilo común de las figuras del informe.

Paleta: azul y naranja para Q-learning y Sarsa (validados para daltonismo),
tintas y rejilla neutras, y una escala de azules para magnitudes.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

AZUL, NARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA_2, GRIS, REJILLA, EJE, FONDO = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
AZULES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SECUENCIAL = LinearSegmentedColormap.from_list("azules", AZULES)

NOMBRES_MERCADO = {"ciclo": "1. Tendencia + ciclo", "browniano": "2. Browniano",
                   "reversion": "3. Reversión a la media", "regimenes": "4. Regímenes"}
NOMBRES_ALGORITMO = {"qlearning": "Q-learning", "sarsa": "Sarsa"}
COLORES_ALGORITMO = {"qlearning": AZUL, "sarsa": NARANJA}


def aplicar_estilo():
    plt.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
        "axes.edgecolor": EJE, "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA,
        "xtick.color": GRIS, "ytick.color": GRIS, "axes.grid": True, "grid.color": REJILLA,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "font.size": 9, "axes.titlesize": 10,
        "legend.frameon": False,
    })


def compacto(x, _=None):
    """Para ejes: 150000 -> 150K."""
    if abs(x) >= 1e6:
        return f"{x / 1e6:g}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:g}K"
    return f"{x:g}"


def corto(x):
    """Para anotar valores: 3 cifras como mucho."""
    if abs(x) >= 1e4:
        return f"{x / 1e3:.0f}K"
    if abs(x) >= 100:
        return f"{x:.0f}"
    return f"{x:.1f}"


def mapa_calor(ax, tabla, destacar=None):
    """Tabla anotada; el color es relativo dentro del panel (más oscuro = mejor).

    tabla: DataFrame filas × columnas. destacar: (fila, columna) a recuadrar.
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
        # Doble borde (claro dentro, oscuro fuera) para que se vea sobre cualquier tono
        i, j = destacar
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=TINTA, lw=2.5))
        ax.add_patch(Rectangle((j - 0.42, i - 0.42), 0.84, 0.84, fill=False, ec=FONDO, lw=1.5))
