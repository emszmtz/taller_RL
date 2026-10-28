"""Todo lo necesario para probar los agentes con precios reales de EODHD.

La idea es la misma que con los mercados simulados, con una diferencia
importante: de cada activo solo hay una historia. Por eso:

- para entrenar, se cortan ventanas de 500 días al azar dentro del periodo de
  entrenamiento (hasta 2018), y cada episodio usa una ventana distinta;
- para evaluar, se recorre de una sola vez todo el periodo de test (desde 2019).

En results/ solo se guardan métricas y políticas, nunca precios: los datos de
EODHD no se pueden redistribuir, y cada uno los descarga con su propia clave
(ver data.py).
"""
import numpy as np

from .agents import Discretizador, entrenar, politica_dentro
from .data import descargar_precios
from .evaluation import ejecutar_episodio, metricas

# Los cuatro activos: un índice estadounidense, uno español, una criptomoneda y una divisa
ACTIVOS = {
    "sp500": dict(ticker="SPY.US", nombre="S&P 500 (SPY)", inicio="2005-01-01"),
    "ibex": dict(ticker="IBEX.INDX", nombre="IBEX 35", inicio="2005-01-01"),
    # Bitcoin empieza en 2014: antes era casi ilíquido y cotizaba a céntimos
    "bitcoin": dict(ticker="BTC-USD.CC", nombre="Bitcoin", inicio="2014-01-01"),
    "eurusd": dict(ticker="EURUSD.FOREX", nombre="EUR/USD", inicio="2005-01-01"),
}
# Hasta esta fecha se entrena; desde la siguiente, se evalúa
FIN_ENTRENAMIENTO = "2018-12-31"
INICIO_TEST = "2019-01-01"


def cargar(activo):
    """Devuelve el precio de cierre diario de un activo, indexado por fecha.

    Es el cierre ajustado (por dividendos y splits, cuando los hay). Si el activo
    ya está descargado en data/, lo lee de ahí; si no, lo descarga de EODHD.
    """
    a = ACTIVOS[activo]
    serie = descargar_precios(a["ticker"], start=a["inicio"])["adjusted_close"].astype(float)
    return serie[serie > 0].dropna()


def dividir(serie):
    """Separa la serie en dos tramos: los precios de entrenamiento (hasta 2018) y los de test (desde 2019)."""
    return serie[:FIN_ENTRENAMIENTO].to_numpy(), serie[INICIO_TEST:].to_numpy()


class GeneradorVentanas:
    """Hace el papel de "generador de precios" con datos reales.

    Con los mercados simulados, cada episodio genera una serie nueva. Con datos
    reales eso no es posible, así que cada vez que se le pide una serie devuelve
    una ventana de n_steps días elegida al azar dentro de los precios de
    entrenamiento. Usa el azar global de NumPy, igual que los generadores del
    enunciado, así que entrenar() lo hace reproducible con su semilla.
    """

    def __init__(self, precios):
        self.precios = np.asarray(precios, dtype=float)

    def __call__(self, n_steps=500):
        inicio = np.random.randint(0, len(self.precios) - n_steps + 1)
        return self.precios[inicio:inicio + n_steps]


def politica_desde_cadena(politica, cortes):
    """Vuelve a construir una política a partir de cómo se guardó en results/.

    Las políticas se guardan como una cadena de ceros y unos (fuera o dentro en
    cada estado), junto con los cortes de la tendencia. Esta función la convierte
    otra vez en una función que recibe lo que ve el agente y devuelve una acción.
    Se comporta exactamente igual que la política original (hay un test que lo
    comprueba).
    """
    disc = Discretizador(cortes)
    dentro = [c == "1" for c in politica]

    def actuar(obs):
        s = disc(obs)
        if s % 2 == 0:
            # Sin el activo: si la política dice "dentro", compra; si no, sigue fuera
            return 2 if dentro[s] else 1
        # Con el activo: si la política dice "dentro", mantiene; si no, vende
        return 1 if dentro[s] else 0
    return actuar


def ejecutar_real(config):
    """Entrena un agente con un activo real y lo evalúa en los dos periodos.

    Recibe una configuración con el activo, el algoritmo, la variante de
    hiperparámetros, los precios de entrenamiento y de test ya separados, y los
    parámetros de entrenar(). Los intervalos de la tendencia se calculan con los
    precios de entrenamiento. Devuelve las métricas del periodo de entrenamiento
    completo y del periodo de test, y la política aprendida.
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
