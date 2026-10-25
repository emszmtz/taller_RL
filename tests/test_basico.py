"""Comprobaciones rápidas del código compartido. Ejecuta `pytest` desde la raíz del repositorio."""
import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl_trader import env as env_mod
from rl_trader import prices as prices_mod
from rl_trader.agents import (Discretizador, entrenar, episodio_estable, politica_dentro, politica_efectiva,
                              tendencias)
from rl_trader.baselines import buy_and_hold
from rl_trader.env import SimpleTradingEnv
from rl_trader.evaluation import ejecutar_episodio
from rl_trader.prices import GENERADORES

ENUNCIADO = Path(__file__).resolve().parent.parent / "notebooks" / "00_enunciado_taller_trader.ipynb"


def _definiciones(fuente):
    return {n.name: ast.dump(n) for n in ast.parse(fuente).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def test_entorno_y_generadores_identicos_al_enunciado():
    del_enunciado = {}
    for celda in json.loads(ENUNCIADO.read_text())["cells"]:
        if celda["cell_type"] == "code":
            del_enunciado.update(_definiciones("".join(celda["source"])))
    for mod in (env_mod, prices_mod):
        for nombre, dump in _definiciones(inspect.getsource(mod)).items():
            assert del_enunciado.get(nombre) == dump, f"{nombre} no coincide con el enunciado"


@pytest.mark.parametrize("nombre", GENERADORES)
def test_tendencias_coinciden_con_el_entorno(nombre):
    np.random.seed(0)
    precios = GENERADORES[nombre](n_steps=100)
    env = SimpleTradingEnv(precios)
    obs, _ = env.reset()
    vistas, done = [obs[0]], False
    while not done:
        obs, _, done, _, _ = env.step(1)
        vistas.append(obs[0])
    # El entorno devuelve float32
    np.testing.assert_allclose(vistas, tendencias(precios), rtol=1e-5, atol=1e-7)


def test_buy_and_hold_coincide_con_el_enunciado():
    np.random.seed(1)
    precios = GENERADORES["browniano"](n_steps=200)
    balances, _ = ejecutar_episodio(precios, buy_and_hold)
    acciones = 1000 * 0.999 / precios[5]
    np.testing.assert_allclose(balances[1:], acciones * precios[6:])


def test_discretizador_simetrico_con_cero_como_corte():
    disc = Discretizador.ajustar(GENERADORES["browniano"], n_intervalos=8, n_series=5)
    assert disc.n_intervalos == 8 and disc.n_estados == 16
    assert disc.cortes[3] == 0.0
    np.testing.assert_allclose(disc.cortes[:3], -np.array(disc.cortes[4:][::-1]))
    assert disc((-1e-9, 0)) == 2 * 3
    assert disc((0.0, 1)) == 2 * 4 + 1


def test_sarsa_y_qlearning_actualizan_distinto():
    g = GENERADORES["reversion"]
    disc = Discretizador.ajustar(g, n_intervalos=4, n_series=5)
    q_learning = entrenar("qlearning", g, disc, n_episodios=5).Q
    q_sarsa = entrenar("sarsa", g, disc, n_episodios=5).Q
    assert not np.allclose(q_learning, q_sarsa)


def test_politica_dentro_y_efectiva():
    Q = np.array([[0, 0, 1],   # sin activo, compra    -> dentro
                  [1, 0, 0],   # con activo, vende     -> fuera
                  [1, 0, 0],   # sin activo, "vende"   -> fuera
                  [0, 1, 0]])  # con activo, mantiene  -> dentro
    assert list(politica_dentro(Q)) == [1, 0, 0, 1]
    assert list(politica_efectiva(Q)) == [1, 0, 0, 1]
    Q[0] = [1, 0, 0]  # ya no compra en ningún estado: los estados con activo no se alcanzan
    assert list(politica_efectiva(Q)) == [0, 0, 0, 0]


def test_episodio_estable():
    v = pd.DataFrame({"episodio": [100, 200, 300, 400], "politica": ["01", "10", "11", "11"]})
    assert episodio_estable(v) == 300
    v["politica"] = ["11"] * 4
    assert episodio_estable(v) == 100
    v["politica"] = ["11", "11", "11", "10"]
    assert episodio_estable(v) == 400


def test_elegir_mejores_promedia_semillas_y_desempata_por_operaciones():
    from rl_trader.experiments import HIPERPARAMETROS, elegir_mejores
    base = dict(mercado="m", algoritmo="qlearning", **{k: 0 for k in HIPERPARAMETROS}, val_sharpe=0.0)
    runs = pd.DataFrame([
        base | dict(gamma=0.0, val_mediana=10, val_operaciones=5),  # media 6: pierde
        base | dict(gamma=0.0, val_mediana=2, val_operaciones=5),
        base | dict(gamma=0.5, val_mediana=8, val_operaciones=9),   # media 8, opera más
        base | dict(gamma=0.5, val_mediana=8, val_operaciones=9),
        base | dict(gamma=0.9, val_mediana=8, val_operaciones=1),   # media 8, opera menos: gana
        base | dict(gamma=0.9, val_mediana=8, val_operaciones=1),
    ])
    mejor = elegir_mejores(runs)
    assert len(mejor) == 1 and mejor.loc[0, "gamma"] == 0.9


def _seno(n_steps=500):
    return 100 + 10 * np.sin(2 * np.pi * np.arange(n_steps) / 40)


@pytest.mark.parametrize("algoritmo", ["qlearning", "sarsa"])
def test_aprende_momentum_en_un_mercado_determinista(algoritmo):
    """En una sinusoide sin ruido seguir la tendencia es claramente lo mejor."""
    disc = Discretizador.ajustar(_seno, n_intervalos=4, n_series=1)
    res = entrenar(algoritmo, _seno, disc, n_episodios=150, alpha=1.0, omega=0.6, gamma=0.9)
    agente, _ = ejecutar_episodio(_seno(), res.politica())
    bh, _ = ejecutar_episodio(_seno(), buy_and_hold)
    assert agente[-1] > 1500 and agente[-1] > bh[-1]
