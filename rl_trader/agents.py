"""Agentes tabulares: discretización del estado, Q-learning y Sarsa (slides 3)."""
import bisect
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .env import SimpleTradingEnv
from .evaluation import N_STEPS, SEMILLA_DISCRETIZADOR, evaluar, generar_series, semilla_entrenamiento

N_ACCIONES = 3  # 0 = vender, 1 = mantener, 2 = comprar
ALGORITMOS = ("qlearning", "sarsa")


def tendencias(precios):
    """La tendencia que observa el entorno en cada paso (misma fórmula que _get_obs)."""
    ventanas = np.lib.stride_tricks.sliding_window_view(np.asarray(precios, dtype=float), 6)
    media = ventanas.mean(axis=1)
    return np.clip((ventanas[:, -1] - media) / media, -1.0, 1.0)


class Discretizador:
    """Convierte la observación (tendencia, posición) en un índice de la tabla Q.

    Los cortes de la tendencia son simétricos, con el 0 como corte central: se
    toman cuantiles de |tendencia| en series del propio generador y se reflejan.
    Así cada intervalo recibe un número parecido de visitas (la escala de la
    tendencia cambia mucho entre mercados) y ningún intervalo mezcla tendencias
    positivas y negativas.
    """

    def __init__(self, cortes):
        self.cortes = [float(c) for c in cortes]
        self.n_intervalos = len(self.cortes) + 1
        self.n_estados = 2 * self.n_intervalos

    @classmethod
    def ajustar(cls, generador, n_intervalos=10, n_series=50, n_steps=N_STEPS):
        if n_intervalos < 2 or n_intervalos % 2:
            raise ValueError("n_intervalos debe ser par (y al menos 2) para que el 0 sea un corte")
        series = generar_series(generador, n_series, SEMILLA_DISCRETIZADOR, n_steps)
        absolutas = np.abs(np.concatenate([tendencias(p) for p in series]))
        mitad = n_intervalos // 2
        qs = np.quantile(absolutas, np.arange(1, mitad) / mitad)
        return cls(np.concatenate([-qs[::-1], [0.0], qs]))

    def intervalo(self, tendencia):
        return bisect.bisect_right(self.cortes, tendencia)

    def __call__(self, obs):
        return 2 * self.intervalo(float(obs[0])) + int(obs[1])


def epsilon_exponencial(episodio, n_episodios, eps_inicial=1.0, eps_final=0.01, frac_decaimiento=0.7):
    """ε baja exponencialmente de eps_inicial a eps_final durante la fracción
    frac_decaimiento del entrenamiento y después se queda en eps_final."""
    n_dec = max(1, int(frac_decaimiento * n_episodios))
    if episodio >= n_dec:
        return eps_final
    return eps_inicial * (eps_final / eps_inicial) ** (episodio / n_dec)


def accion_greedy(q, rng):
    """argmax con desempate aleatorio (al principio la tabla Q es todo ceros)."""
    mejores = np.flatnonzero(q == q.max())
    return int(mejores[0]) if len(mejores) == 1 else int(rng.choice(mejores))


def accion_epsilon_greedy(q, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(N_ACCIONES))
    return accion_greedy(q, rng)


@dataclass
class Resultado:
    algoritmo: str
    Q: np.ndarray
    discretizador: Discretizador
    historial: pd.DataFrame   # una fila por episodio de entrenamiento
    validacion: pd.DataFrame  # métricas en validación cada evaluar_cada episodios
    segundos: float

    def politica(self):
        """Política greedy respecto a Q, para evaluar (sin exploración)."""
        Q, disc = self.Q, self.discretizador
        return lambda obs: int(np.argmax(Q[disc(obs)]))


def entrenar(algoritmo, generador, discretizador, n_episodios=2000, n_steps=N_STEPS,
             alpha=0.05, omega=0.0, gamma=0.9, eps_inicial=1.0, eps_final=0.01,
             frac_decaimiento=0.7, seed=0, series_validacion=None, evaluar_cada=100):
    """Entrena Q-learning o Sarsa tabular; cada episodio usa una serie nueva del generador.

    La tasa de aprendizaje es α = alpha / n(s,a)**omega, con n(s,a) el número de
    visitas al par contando la actual. omega = 0 da α constante; con
    0.5 < omega <= 1 se cumplen las condiciones de convergencia Σα = ∞, Σα² < ∞.
    """
    if algoritmo not in ALGORITMOS:
        raise ValueError(f"algoritmo debe ser uno de {ALGORITMOS}")
    sarsa = algoritmo == "sarsa"
    rng = np.random.default_rng(seed)
    Q = np.zeros((discretizador.n_estados, N_ACCIONES))
    visitas = np.zeros_like(Q)
    historial, validacion = [], []
    inicio = time.perf_counter()

    for ep in range(n_episodios):
        eps = epsilon_exponencial(ep, n_episodios, eps_inicial, eps_final, frac_decaimiento)
        np.random.seed(semilla_entrenamiento(seed, ep))
        env = SimpleTradingEnv(generador(n_steps=n_steps))
        obs, _ = env.reset()
        s = discretizador(obs)
        a = accion_epsilon_greedy(Q[s], eps, rng)
        suma_recompensas, done = 0.0, False

        while not done:
            obs, r, done, _, _ = env.step(a)
            s2 = discretizador(obs)
            if sarsa:
                a2 = accion_epsilon_greedy(Q[s2], eps, rng)
                siguiente = Q[s2, a2]
            else:
                siguiente = Q[s2].max()
            # Como en las slides, Q(terminal, ·) = 0: no se hace bootstrap al acabar la serie
            objetivo = r if done else r + gamma * siguiente
            visitas[s, a] += 1
            Q[s, a] += alpha / visitas[s, a] ** omega * (objetivo - Q[s, a])
            if not sarsa:
                a2 = accion_epsilon_greedy(Q[s2], eps, rng)
            s, a = s2, a2
            suma_recompensas += r

        historial.append({
            "episodio": ep,
            "epsilon": eps,
            "suma_recompensas": suma_recompensas,
            "retorno_pct": 100 * (env.balance / env.initial_balance - 1),
        })
        if series_validacion is not None and ((ep + 1) % evaluar_cada == 0 or ep + 1 == n_episodios):
            Q_actual = Q.copy()
            df = evaluar(lambda o: int(np.argmax(Q_actual[discretizador(o)])), series_validacion)
            validacion.append({
                "episodio": ep + 1,
                "segundos": time.perf_counter() - inicio,
                "retorno_pct": df["retorno_pct"].mean(),
                "retorno_mediana_pct": df["retorno_pct"].median(),
                "sharpe": df["sharpe"].mean(),
            })

    return Resultado(algoritmo, Q, discretizador, pd.DataFrame(historial),
                     pd.DataFrame(validacion), time.perf_counter() - inicio)


def mapa_politica(resultado):
    """Tabla intervalo de tendencia × posición -> 'dentro'/'fuera' según la política greedy."""
    disc, Q = resultado.discretizador, resultado.Q
    bordes = [-np.inf] + disc.cortes + [np.inf]
    filas = []
    for i in range(disc.n_intervalos):
        fila = {"tendencia": f"[{bordes[i]:.4f}, {bordes[i + 1]:.4f})"}
        for pos, nombre in ((0, "sin_activo"), (1, "con_activo")):
            a = int(np.argmax(Q[2 * i + pos]))
            dentro = (a == 2) if pos == 0 else (a != 0)
            fila[nombre] = "dentro" if dentro else "fuera"
        filas.append(fila)
    return pd.DataFrame(filas)
