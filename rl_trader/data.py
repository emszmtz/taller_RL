"""Descarga de precios reales diarios desde EODHD.

Hace falta una clave de EODHD. Se lee de la variable de entorno EODHD_API_KEY o
del fichero .env de la raíz del repositorio, que git ignora. No la escribas
nunca en el código ni en un notebook: el repositorio es público.
"""
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EOD_URL = "https://eodhd.com/api/eod/{ticker}"

load_dotenv(ROOT / ".env")


def _api_key():
    """Devuelve la clave de EODHD o da un error que explica cómo configurarla."""
    key = os.environ.get("EODHD_API_KEY")
    if not key:
        raise RuntimeError(
            "Falta la clave de EODHD: copia .env.example a .env y escribe tu clave "
            "en EODHD_API_KEY (o define esa variable de entorno)."
        )
    return key


def descargar_precios(ticker, start=None, end=None, cache=True):
    """Descarga los precios diarios de un activo y los devuelve como tabla.

    La tabla tiene una fila por día y columnas de apertura, máximo, mínimo,
    cierre, cierre ajustado y volumen.

    - ticker: el código del activo en EODHD, con el formato SÍMBOLO.MERCADO; por
      ejemplo, "SPY.US", "BTC-USD.CC" o "EURUSD.FOREX".
    - start, end: fechas de inicio y fin, "AAAA-MM-DD" (opcionales).
    - cache: si es True, guarda una copia en data/ y la próxima vez la lee de ahí
      sin volver a descargar. La carpeta data/ no se sube al repositorio.
    """
    path = DATA_DIR / f"{ticker}_{start or 'inicio'}_{end or 'fin'}.csv"
    if cache and path.exists():
        return pd.read_csv(path, index_col="date", parse_dates=True)

    params = {"api_token": _api_key(), "fmt": "json", "period": "d"}
    if start:
        params["from"] = start
    if end:
        params["to"] = end

    # Si la conexión falla, el error de requests incluye la URL completa, y con
    # ella la clave. Por eso se da un error propio sin la URL: así la clave no
    # puede acabar en la salida de un notebook.
    try:
        resp = requests.get(EOD_URL.format(ticker=ticker), params=params, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"No se pudo conectar con EODHD ({type(e).__name__}) para {ticker}") from None
    if not resp.ok:
        raise RuntimeError(f"EODHD respondió {resp.status_code} para {ticker}: {resp.text[:200]}")

    df = pd.DataFrame(resp.json())
    if df.empty:
        raise RuntimeError(f"EODHD no devolvió datos para {ticker}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    if cache:
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(path)
    return df
