"""Los agentes que aprenden a invertir.

Aquí está todo lo que necesita un agente tabular para aprender:

- El discretizador, que convierte lo que ve el agente (la tendencia del precio,
  que es un número continuo, y si tiene o no el activo) en una fila de la tabla Q.
  Los métodos tabulares solo funcionan con un número finito de estados.
- La exploración ε-greedy: casi siempre elige la mejor acción según lo aprendido,
  pero de vez en cuando prueba una al azar.
- El bucle de entrenamiento de Q-learning y Sarsa (slides 3) y del control
  Monte Carlo (slides 2, añadido por Josep).
- Utilidades para ver qué ha aprendido el agente y cuándo deja de cambiar.
"""
import bisect
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .env import SimpleTradingEnv
from .evaluation import N_STEPS, SEMILLA_DISCRETIZADOR, evaluar, generar_series, semilla_entrenamiento

# Las tres acciones del entorno: 0 = vender, 1 = mantener, 2 = comprar
N_ACCIONES = 3
# Los algoritmos que sabe entrenar la función entrenar()
ALGORITMOS = ("qlearning", "sarsa", "montecarlo")


def tendencias(precios):
    """Calcula la tendencia que ve el agente en cada paso de una serie de precios.

    La tendencia compara el precio de hoy con la media de los últimos 6 precios
    (incluido el de hoy): (precio − media) / media. Por ejemplo, 0,02 significa
    que el precio está un 2% por encima de su media reciente. Es la misma cuenta
    que hace el entorno en cada paso, pero para toda la serie de una vez.
    """
    ventanas = np.lib.stride_tricks.sliding_window_view(np.asarray(precios, dtype=float), 6)
    media = ventanas.mean(axis=1)
    return np.clip((ventanas[:, -1] - media) / media, -1.0, 1.0)


class Discretizador:
    """Traduce lo que ve el agente a un número de fila de la tabla Q.

    El agente ve dos cosas: la tendencia (un número continuo) y si tiene el
    activo (0 o 1). Como la tabla Q solo puede tener un número finito de filas,
    la tendencia se reparte en intervalos, y cada combinación de intervalo y
    posición es un estado.

    Cómo se eligen los intervalos:
    - Son simétricos y el 0 siempre es uno de los cortes, para que ningún
      intervalo mezcle subidas con bajadas.
    - Se calculan con cuantiles de la tendencia observada en cada mercado, para
      que todos los intervalos reciban un número parecido de visitas. Hace falta
      porque la escala cambia mucho de un mercado a otro: en el de ciclo la
      tendencia se mueve entre −30% y +40%, y en los demás casi siempre entre
      −6% y +6%.

    Con k intervalos hay 2k estados: el estado 2·i es "intervalo i, sin el
    activo" y el 2·i + 1 es "intervalo i, con el activo".
    """

    def __init__(self, cortes):
        self.cortes = [float(c) for c in cortes]
        self.n_intervalos = len(self.cortes) + 1
        self.n_estados = 2 * self.n_intervalos

    @classmethod
    def ajustar(cls, generador, n_intervalos=10, n_series=50, n_steps=N_STEPS):
        """Calcula los cortes a partir de varias series del generador (50 por defecto).

        Usa semillas reservadas solo para esto, distintas de las de entrenamiento,
        validación y test.
        """
        return cls.desde_series(generar_series(generador, n_series, SEMILLA_DISCRETIZADOR, n_steps), n_intervalos)

    @classmethod
    def desde_series(cls, series, n_intervalos=10):
        """Calcula los cortes a partir de series de precios que ya tienes.

        Sirve, por ejemplo, para los activos reales, donde no hay un generador: se
        usan los precios del periodo de entrenamiento. El número de intervalos
        tiene que ser par, para que el 0 quede como corte.
        """
        if n_intervalos < 2 or n_intervalos % 2:
            raise ValueError("n_intervalos debe ser par (y al menos 2) para que el 0 sea un corte")
        absolutas = np.abs(np.concatenate([tendencias(p) for p in series]))
        mitad = n_intervalos // 2
        qs = np.quantile(absolutas, np.arange(1, mitad) / mitad)
        return cls(np.concatenate([-qs[::-1], [0.0], qs]))

    def intervalo(self, tendencia):
        """Devuelve en qué intervalo cae una tendencia (el 0 es el más bajista)."""
        return bisect.bisect_right(self.cortes, tendencia)

    def __call__(self, obs):
        """Devuelve el número de estado que corresponde a lo que ve el agente: (tendencia, posición)."""
        return 2 * self.intervalo(float(obs[0])) + int(obs[1])


def epsilon_exponencial(episodio, n_episodios, eps_inicial=1.0, eps_final=0.01, frac_decaimiento=0.7):
    """Devuelve el ε (la probabilidad de explorar) que toca en un episodio.

    ε empieza en eps_inicial (con 1, todas las acciones son al azar) y baja de
    forma exponencial hasta eps_final durante la primera parte del entrenamiento
    (frac_decaimiento; por ejemplo, el 70% de los episodios). A partir de ahí se
    queda fijo en eps_final.
    """
    n_dec = max(1, int(frac_decaimiento * n_episodios))
    if episodio >= n_dec:
        return eps_final
    return eps_inicial * (eps_final / eps_inicial) ** (episodio / n_dec)


def accion_greedy(q, rng):
    """Elige la acción con más valor en la tabla Q (la acción "greedy").

    Si hay empate, elige al azar entre las empatadas. Pasa mucho al principio,
    cuando toda la tabla vale cero, y así el agente no se queda siempre con la
    primera acción.
    """
    mejores = np.flatnonzero(q == q.max())
    return int(mejores[0]) if len(mejores) == 1 else int(rng.choice(mejores))


def accion_epsilon_greedy(q, eps, rng):
    """Elige una acción con la estrategia ε-greedy de los apuntes.

    Con probabilidad ε elige una acción al azar (explora); si no, la mejor según
    la tabla Q (explota).
    """
    if rng.random() < eps:
        return int(rng.integers(N_ACCIONES))
    return accion_greedy(q, rng)


def politica_dentro(Q):
    """Resume la política aprendida en "dentro" (1) o "fuera" (0) para cada estado.

    El agente tiene tres acciones, pero en la práctica solo decide si quiere estar
    dentro del mercado o fuera:
    - sin el activo, "dentro" significa que compra;
    - con el activo, "dentro" significa que no vende.
    Es lo que se dibuja en los mapas de política.
    """
    a = Q.argmax(axis=1)
    dentro = a != 0
    dentro[0::2] = a[0::2] == 2
    return dentro.astype(int)


def politica_efectiva(Q):
    """Como politica_dentro, pero sin contar los estados a los que nunca se llega.

    Si el agente no compra en ningún estado, nunca llega a tener el activo, así
    que lo que diga la tabla para los estados "con activo" no importa. En ese caso
    se marcan todos como fuera, para que no parezca que la política cambia por
    estados que no se usan.
    """
    dentro = politica_dentro(Q)
    if not dentro[0::2].any():
        dentro[1::2] = 0
    return dentro


def episodio_estable(validacion):
    """Calcula en qué episodio dejó de cambiar la política del agente.

    Recibe la tabla de controles en validación que guarda entrenar() (uno cada
    cierto número de episodios) y devuelve el primer control a partir del cual la
    política ya no cambió. Si cambió en el último control, devuelve ese mismo
    episodio, lo que quiere decir que todavía no se había estabilizado.
    """
    politicas = validacion["politica"].tolist()
    k = len(politicas) - 1
    while k > 0 and politicas[k - 1] == politicas[-1]:
        k -= 1
    return int(validacion["episodio"].iloc[k])


def episodio_convergencia(validacion, columna="retorno_mediana_pct", tol_rel=0.1, tol_abs=1.0):
    """Calcula en qué episodio se estabilizó el resultado del agente.

    Es el criterio de convergencia del informe: el primer control a partir del
    cual la mediana en validación ya no se aleja de su valor final más de un 10%
    (tol_rel) o, si eso es muy poco, más de 1 punto (tol_abs).
    """
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
    """Todo lo que devuelve un entrenamiento.

    - algoritmo: "qlearning", "sarsa" o "montecarlo".
    - Q: la tabla aprendida, con una fila por estado y una columna por acción.
    - discretizador: el que se usó; hace falta para volver a usar la tabla.
    - historial: una fila por episodio de entrenamiento.
    - validacion: los resultados en validación cada cierto número de episodios.
    - segundos: el tiempo de entrenamiento, sin contar esas evaluaciones.
    """
    algoritmo: str
    Q: np.ndarray
    discretizador: Discretizador
    historial: pd.DataFrame
    validacion: pd.DataFrame
    segundos: float

    def politica(self):
        """Devuelve la política aprendida, lista para usar en test.

        Es la política greedy: en cada estado elige la acción con más valor en la
        tabla Q, sin explorar.
        """
        Q, disc = self.Q, self.discretizador
        return lambda obs: int(np.argmax(Q[disc(obs)]))


def entrenar(algoritmo, generador, discretizador, n_episodios=2000, n_steps=N_STEPS,
             alpha=0.05, omega=0.0, gamma=0.9, eps_inicial=1.0, eps_final=0.01,
             frac_decaimiento=0.7, seed=0, series_validacion=None, evaluar_cada=100,
             primera_visita=False):
    """Entrena un agente y devuelve un Resultado.

    Cada episodio usa una serie de precios nueva del generador, así que el agente
    aprende cómo se comporta el mercado y no una serie concreta. Al entrenar elige
    las acciones con ε-greedy y, después de cada paso, actualiza la tabla Q:

    - Q-learning usa como objetivo la recompensa más el mejor valor del estado
      siguiente.
    - Sarsa usa la recompensa más el valor de la acción que de verdad va a tomar
      en el estado siguiente.
    - Monte Carlo no actualiza nada hasta acabar el episodio, y entonces usa el
      retorno realmente obtenido en lugar de una estimación.

    Parámetros principales:
    - alpha y omega: la tasa de aprendizaje es alpha / n**omega, donde n es cuántas
      veces se ha visitado ese par estado-acción. Con omega = 0 es constante; con
      omega entre 0,5 y 1 va bajando poco a poco y cumple las condiciones de
      convergencia de los apuntes.
    - gamma: el factor de descuento, es decir, cuánto pesan las recompensas futuras.
    - eps_inicial, eps_final y frac_decaimiento: cómo baja la exploración (ver
      epsilon_exponencial).
    - seed: fija la serie que toca en cada episodio y las decisiones al azar, para
      que el resultado se pueda repetir.
    - series_validacion y evaluar_cada: si se dan, cada evaluar_cada episodios se
      prueba la política del momento en esas series y se guarda el resultado.
    - primera_visita: solo para Monte Carlo; cuenta solo la primera visita a cada
      par estado-acción dentro del episodio.
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

    # Cada episodio recorre una serie de precios nueva de principio a fin
    for ep in range(n_episodios):
        eps = epsilon_exponencial(ep, n_episodios, eps_inicial, eps_final, frac_decaimiento)
        # La semilla decide qué serie de precios toca en este episodio
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
            # Valor del estado siguiente: Sarsa usa la acción que va a tomar de
            # verdad; Q-learning, la mejor posible
            if sarsa:
                a2 = accion_epsilon_greedy(Q[s2], eps, rng)
                siguiente = Q[s2, a2]
            else:
                siguiente = Q[s2].max()
            # Al acabar la serie no hay estado siguiente: el objetivo es solo la
            # recompensa (en los apuntes, Q(terminal, ·) = 0)
            objetivo = r if done else r + gamma * siguiente
            # Acerca el valor de (s, a) al objetivo, más o menos según la tasa de aprendizaje
            visitas[s, a] += 1
            Q[s, a] += alpha / visitas[s, a] ** omega * (objetivo - Q[s, a])
            # Q-learning elige la siguiente acción con la tabla ya actualizada
            if not sarsa:
                a2 = accion_epsilon_greedy(Q[s2], eps, rng)
            s, a = s2, a2
            suma_recompensas += r

        if montecarlo:
            actualizar_montecarlo(Q, visitas, trayectoria, alpha, omega, gamma, primera_visita)

        # Lo que pasó en este episodio, para poder seguir el entrenamiento
        historial.append({
            "episodio": ep,
            "epsilon": eps,
            "suma_recompensas": suma_recompensas,
            "retorno_pct": 100 * (env.balance / env.initial_balance - 1),
        })
        # Cada evaluar_cada episodios se prueba la política del momento (sin explorar)
        # en las series de validación. Ese tiempo no cuenta como entrenamiento.
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
    """Devuelve la política aprendida como una tabla fácil de leer.

    Tiene una fila por intervalo de tendencia, con sus límites, y dos columnas:
    qué hace el agente sin el activo y con el activo ("dentro" o "fuera").
    """
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
