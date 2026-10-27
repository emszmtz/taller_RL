"""Agentes tabulares: discretización del estado, Q-learning y Sarsa (slides 3) y
control Monte Carlo on-policy (slides 2)."""
import bisect
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .env import SimpleTradingEnv
from .evaluation import N_STEPS, SEMILLA_DISCRETIZADOR, evaluar, generar_series, semilla_entrenamiento

N_ACCIONES = 3  # 0 = vender, 1 = mantener, 2 = comprar
ALGORITMOS = ("qlearning", "sarsa", "montecarlo")


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


def politica_dentro(Q):
    """1 donde la política greedy está dentro del mercado y 0 donde está fuera.

    Sin el activo (estados pares), dentro = comprar; con el activo (impares), dentro = no vender.
    """
    a = Q.argmax(axis=1)
    dentro = a != 0
    dentro[0::2] = a[0::2] == 2
    return dentro.astype(int)


def politica_efectiva(Q):
    """politica_dentro, pero si el agente no compra en ningún estado, los estados
    con activo no se alcanzan nunca y se marcan todos como fuera."""
    dentro = politica_dentro(Q)
    if not dentro[0::2].any():
        dentro[1::2] = 0
    return dentro


def episodio_estable(validacion):
    """Primer control a partir del cual la política efectiva ya no cambia.

    Si cambia en el último control, devuelve ese episodio: no se ha estabilizado.
    """
    politicas = validacion["politica"].tolist()
    k = len(politicas) - 1
    while k > 0 and politicas[k - 1] == politicas[-1]:
        k -= 1
    return int(validacion["episodio"].iloc[k])


def episodio_convergencia(validacion, columna="retorno_mediana_pct", tol_rel=0.1, tol_abs=1.0):
    """Primer control a partir del cual la métrica de validación ya no se aleja de
    su valor final más de max(tol_rel · |final|, tol_abs)."""
    v = validacion[columna].to_numpy()
    tolerancia = max(tol_rel * abs(v[-1]), tol_abs)
    lejos = np.flatnonzero(np.abs(v - v[-1]) > tolerancia)
    k = lejos[-1] + 1 if len(lejos) else 0
    return int(validacion["episodio"].iloc[k])


def actualizar_montecarlo(Q, visitas, trayectoria, alpha, omega, gamma, primera_visita=False):
    """Actualización de control MC on-policy al final de un episodio (slides 2).

    Recorre la trayectoria [(s, a, r), ...] hacia atrás con G ← γG + R y mueve
    Q(s, a) hacia G con α = alpha / n(s,a)**omega. Con alpha = 1 y omega = 1, Q(s, a)
    es el promedio de los retornos observados, como en el algoritmo de las slides.
    Con primera_visita solo cuenta la primera aparición de cada par en el episodio
    (first-visit); si no, todas (every-visit).
    """
    if primera_visita:
        primera = {}
        for t, (s, a, _) in enumerate(trayectoria):
            primera.setdefault((s, a), t)
    G = 0.0
    for t in range(len(trayectoria) - 1, -1, -1):
        s, a, r = trayectoria[t]
        G = gamma * G + r
        if primera_visita and primera[(s, a)] != t:
            continue
        visitas[s, a] += 1
        Q[s, a] += alpha / visitas[s, a] ** omega * (G - Q[s, a])


@dataclass
class Resultado:
    algoritmo: str
    Q: np.ndarray
    discretizador: Discretizador
    historial: pd.DataFrame   # una fila por episodio de entrenamiento
    validacion: pd.DataFrame  # métricas en validación cada evaluar_cada episodios
    segundos: float           # tiempo de entrenamiento, sin contar las evaluaciones en validación

    def politica(self):
        """Política greedy respecto a Q, para evaluar (sin exploración)."""
        Q, disc = self.Q, self.discretizador
        return lambda obs: int(np.argmax(Q[disc(obs)]))


def entrenar(algoritmo, generador, discretizador, n_episodios=2000, n_steps=N_STEPS,
             alpha=0.05, omega=0.0, gamma=0.9, eps_inicial=1.0, eps_final=0.01,
             frac_decaimiento=0.7, seed=0, series_validacion=None, evaluar_cada=100,
             primera_visita=False):
    """Entrena Q-learning, Sarsa o control Monte Carlo tabular; cada episodio usa una
    serie nueva del generador.

    La tasa de aprendizaje es α = alpha / n(s,a)**omega, con n(s,a) el número de
    visitas al par contando la actual. omega = 0 da α constante; con
    0.5 < omega <= 1 se cumplen las condiciones de convergencia Σα = ∞, Σα² < ∞.

    Monte Carlo usa la misma política ε-greedy, pero no actualiza Q hasta acabar el
    episodio, con el retorno real G en vez de un objetivo con bootstrap.
    primera_visita solo afecta a Monte Carlo.
    """
    if algoritmo not in ALGORITMOS:
        raise ValueError(f"algoritmo debe ser uno de {ALGORITMOS}")
    sarsa = algoritmo == "sarsa"
    montecarlo = algoritmo == "montecarlo"
    rng = np.random.default_rng(seed)
    Q = np.zeros((discretizador.n_estados, N_ACCIONES))
    visitas = np.zeros_like(Q)
    historial, validacion = [], []
    inicio = time.perf_counter()
    t_evaluando = 0.0

    for ep in range(n_episodios):
        eps = epsilon_exponencial(ep, n_episodios, eps_inicial, eps_final, frac_decaimiento)
        np.random.seed(semilla_entrenamiento(seed, ep))
        env = SimpleTradingEnv(generador(n_steps=n_steps))
        obs, _ = env.reset()
        s = discretizador(obs)
        a = accion_epsilon_greedy(Q[s], eps, rng)
        suma_recompensas, done = 0.0, False
        trayectoria = []  # (s, a, r) del episodio, solo para Monte Carlo

        while not done:
            obs, r, done, _, _ = env.step(a)
            s2 = discretizador(obs)
            if montecarlo:
                # Q no cambia hasta acabar el episodio: la política es fija durante él
                trayectoria.append((s, a, r))
                s, a = s2, accion_epsilon_greedy(Q[s2], eps, rng)
                suma_recompensas += r
                continue
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

        if montecarlo:
            actualizar_montecarlo(Q, visitas, trayectoria, alpha, omega, gamma, primera_visita)

        historial.append({
            "episodio": ep,
            "epsilon": eps,
            "suma_recompensas": suma_recompensas,
            "retorno_pct": 100 * (env.balance / env.initial_balance - 1),
        })
        if series_validacion is not None and ((ep + 1) % evaluar_cada == 0 or ep + 1 == n_episodios):
            t0 = time.perf_counter()
            Q_actual = Q.copy()
            df = evaluar(lambda o: int(np.argmax(Q_actual[discretizador(o)])), series_validacion)
            validacion.append({
                "episodio": ep + 1,
                "segundos": t0 - inicio - t_evaluando,
                "retorno_pct": df["retorno_pct"].mean(),
                "retorno_mediana_pct": df["retorno_pct"].median(),
                "sharpe": df["sharpe"].mean(),
                "politica": "".join(map(str, politica_efectiva(Q_actual))),
            })
            t_evaluando += time.perf_counter() - t0

    return Resultado(algoritmo, Q, discretizador, pd.DataFrame(historial),
                     pd.DataFrame(validacion), time.perf_counter() - inicio - t_evaluando)


def mapa_politica(resultado):
    """Tabla intervalo de tendencia × posición -> 'dentro'/'fuera' según la política greedy."""
    disc = resultado.discretizador
    dentro = politica_dentro(resultado.Q)
    bordes = [-np.inf] + disc.cortes + [np.inf]
    filas = []
    for i in range(disc.n_intervalos):
        fila = {"tendencia": f"[{bordes[i]:.4f}, {bordes[i + 1]:.4f})"}
        for pos, nombre in ((0, "sin_activo"), (1, "con_activo")):
            fila[nombre] = "dentro" if dentro[2 * i + pos] else "fuera"
        filas.append(fila)
    return pd.DataFrame(filas)
