"""Experimentos con precios reales de EODHD.

Se entrena con ventanas al azar del periodo de entrenamiento y se evalúa cada
periodo completo (entrenamiento y test) como un único episodio. En results/ solo
se guardan métricas y políticas, nunca precios: los datos de EODHD no se
redistribuyen y cada uno los descarga con su clave (ver data.py).
"""
import numpy as np

from .agents import Discretizador, entrenar, politica_dentro
from .data import descargar_precios
from .evaluation import ejecutar_episodio, metricas

ACTIVOS = {
    "sp500": dict(ticker="SPY.US", nombre="S&P 500 (SPY)", inicio="2005-01-01"),
    "ibex": dict(ticker="IBEX.INDX", nombre="IBEX 35", inicio="2005-01-01"),
    # Antes de 2014 Bitcoin era casi ilíquido (precios de céntimos)
    "bitcoin": dict(ticker="BTC-USD.CC", nombre="Bitcoin", inicio="2014-01-01"),
    "eurusd": dict(ticker="EURUSD.FOREX", nombre="EUR/USD", inicio="2005-01-01"),
}
FIN_ENTRENAMIENTO = "2018-12-31"
INICIO_TEST = "2019-01-01"


def cargar(activo):
    """Precio de cierre ajustado diario del activo (pd.Series indexada por fecha)."""
    a = ACTIVOS[activo]
    serie = descargar_precios(a["ticker"], start=a["inicio"])["adjusted_close"].astype(float)
    return serie[serie > 0].dropna()


def dividir(serie):
    """(precios de entrenamiento, precios de test) como arrays."""
    return serie[:FIN_ENTRENAMIENTO].to_numpy(), serie[INICIO_TEST:].to_numpy()


class GeneradorVentanas:
    """Hace de generador de precios para entrenar: cada llamada devuelve una
    ventana al azar de la serie. Usa el generador global de NumPy, como los del
    enunciado, así que agents.entrenar la hace reproducible con su semilla."""

    def __init__(self, precios):
        self.precios = np.asarray(precios, dtype=float)

    def __call__(self, n_steps=500):
        inicio = np.random.randint(0, len(self.precios) - n_steps + 1)
        return self.precios[inicio:inicio + n_steps]


def politica_desde_cadena(politica, cortes):
    """Reconstruye una política obs -> acción a partir de la cadena de politica_dentro
    y de los cortes del discretizador (equivale a la greedy de la tabla Q)."""
    disc = Discretizador(cortes)
    dentro = [c == "1" for c in politica]

    def actuar(obs):
        s = disc(obs)
        if s % 2 == 0:              # sin activo: dentro = comprar
            return 2 if dentro[s] else 1
        return 1 if dentro[s] else 0  # con activo: fuera = vender
    return actuar


def ejecutar_real(config):
    """Entrena un agente con ventanas del periodo de entrenamiento y lo evalúa en
    los periodos completos de entrenamiento y de test.

    config: 'activo', 'algoritmo', 'variante', 'precios_entrenamiento',
    'precios_test', 'n_intervalos' y argumentos de agents.entrenar.
    """
    params = dict(config)
    activo, algoritmo, variante = params.pop("activo"), params.pop("algoritmo"), params.pop("variante")
    entrenamiento, test = params.pop("precios_entrenamiento"), params.pop("precios_test")
    disc = Discretizador.desde_series([entrenamiento], n_intervalos=params.pop("n_intervalos", 10))
    res = entrenar(algoritmo, GeneradorVentanas(entrenamiento), disc, **params)
    politica = res.politica()
    return {
        "activo": activo, "algoritmo": algoritmo, "variante": variante, "seed": params.get("seed", 0),
        "entrenamiento": metricas(*ejecutar_episodio(entrenamiento, politica)),
        "test": metricas(*ejecutar_episodio(test, politica)),
        "segundos": res.segundos,
        "politica": "".join(map(str, politica_dentro(res.Q))),
        "cortes": disc.cortes,
    }
