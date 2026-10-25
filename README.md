# Taller RL: agente de trading con Q-learning y Sarsa

Taller de Aprendizaje por Refuerzo del MIAX-14. Entrenamos agentes tabulares (Q-learning, Sarsa y algún extra) que operan un único activo en el entorno `SimpleTradingEnv` del enunciado, y los comparamos con un agente aleatorio y con *buy & hold* en cuatro mercados simulados y en datos reales de EODHD.

El enunciado original está en [notebooks/00_enunciado_taller_trader.ipynb](notebooks/00_enunciado_taller_trader.ipynb).

## Fechas

| Cuándo | Qué |
|---|---|
| Martes 27/10 | Un representante del grupo envía un email al profesor para el orden de presentación |
| Jueves 29/10, antes de clase | Primera versión del PDF por email al profesor; presentación de 15 min en clase |
| Jueves 29/10, 23:59 | Entrega final: PDF de **2 páginas como máximo** |

## Puesta en marcha

Hace falta Python 3.10 o superior (usamos 3.12; gymnasium aún no siempre va bien con 3.14).

```bash
git clone https://github.com/emszmtz/miax14-taller-rl.git
cd miax14-taller-rl

# Con uv
uv venv --python 3.12
uv pip install -e ".[notebooks]"

# O con pip
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[notebooks]"
```

`pip install -e .` instala el paquete `rl_trader` en modo editable, así que los notebooks pueden hacer `from rl_trader import ...` desde cualquier carpeta.

### Clave de EODHD (solo para datos reales)

```bash
cp .env.example .env
```

Escribe tu clave en `.env` (`EODHD_API_KEY=...`). Ese fichero está en `.gitignore` y **no se sube nunca**. El repositorio es público: no pongas claves en el código, en los notebooks ni en sus salidas.

Los datos descargados se guardan en `data/`, que tampoco se sube: cada uno los descarga con su propia clave.

## Uso

```python
from rl_trader.env import SimpleTradingEnv
from rl_trader.prices import GENERADORES

precios = GENERADORES["regimenes"](n_steps=500)
env = SimpleTradingEnv(precios)
obs, info = env.reset()
```

```python
from rl_trader.data import descargar_precios

df = descargar_precios("SPY.US", start="2005-01-01")
env = SimpleTradingEnv(df["adjusted_close"].to_numpy())
```

## Estructura

```
rl_trader/          Código compartido (importable)
  env.py            Entorno del enunciado, sin modificar
  prices.py         Los 4 generadores de precios del enunciado, sin modificar
  data.py           Descarga de precios reales desde EODHD
notebooks/          Experimentos (uno por tema)
```

Iremos añadiendo `rl_trader/agents.py` (Q-learning, Sarsa...), la evaluación, `results/` (CSV), `figures/` y `report/` (el PDF).

## Cómo colaborar

- Una rama por tarea y *pull request* a `main`; nadie hace push directo a `main`.
- La lógica va en `rl_trader/`; los notebooks solo llaman a esas funciones y muestran resultados. Así los cambios se revisan bien y hay menos conflictos.
- Los notebooks dan muchos conflictos al fusionar: que cada uno trabaje en su propio notebook y borre las salidas pesadas antes de hacer commit.
- Fija semillas (`np.random.seed`) en todo lo que se vaya a comparar o a citar en el informe.

## Plan de trabajo

- [ ] **Fase 1. Núcleo:** discretización de la tendencia, Q-learning, Sarsa, ε con decaimiento, baselines (aleatorio, buy & hold, reglas de momentum y contrarian), evaluación y métricas.
- [ ] **Fase 2. Ajuste:** α, ε, γ, número de intervalos y de episodios, evaluados sobre series de validación.
- [ ] **Fase 3. Resultados:** varias semillas, 100 series de test por mercado, tabla comparativa, curvas de aprendizaje y mapas de política.
- [ ] **Fase 4. Extras:** datos reales (EODHD), sensibilidad a comisiones frente a γ, Expected Sarsa o control Monte Carlo, estado con más información.
- [ ] **Fase 5. Informe:** PDF de 2 páginas con diseño, resultados y análisis crítico.
