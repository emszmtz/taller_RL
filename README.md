# Taller RL: agente de trading con Q-learning y Sarsa

Taller de Aprendizaje por Refuerzo del MIAX-14. Entrenamos agentes tabulares (Q-learning, Sarsa y, como extra, control Monte Carlo) que operan un único activo en el entorno `SimpleTradingEnv` del enunciado, y los comparamos con un agente aleatorio y con *buy & hold* en cuatro mercados simulados y en datos reales de EODHD.

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
uv pip install -e ".[notebooks,dev]"

# O con pip
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[notebooks,dev]"

# Comprobar que todo funciona (unos segundos)
pytest
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

Entrenar y evaluar un agente:

```python
from rl_trader.agents import Discretizador, entrenar
from rl_trader.evaluation import SEMILLA_TEST, evaluar, generar_series

g = GENERADORES["reversion"]
disc = Discretizador.ajustar(g, n_intervalos=10)
res = entrenar("sarsa", g, disc, n_episodios=2000, alpha=1.0, omega=0.6, gamma=0.9)
evaluar(res.politica(), generar_series(g, 100, SEMILLA_TEST)).median()
```

Varios entrenamientos en paralelo: `rl_trader.experiments.ejecutar_varios(configs)`. En un notebook funciona directamente. En un script `.py`, la llamada tiene que ir dentro de `if __name__ == "__main__":`, porque en macOS y Windows cada proceso hijo vuelve a importar el script.

### Series de entrenamiento, validación y test

Cada conjunto sale de un rango de semillas distinto (ver `rl_trader/evaluation.py`):

- **Entrenamiento:** una serie nueva en cada episodio.
- **Validación:** 20 series, para curvas y para elegir hiperparámetros.
- **Test:** 100 series, solo para los resultados finales.

Todos los agentes se evalúan sobre las mismas series de test. **No elijáis hiperparámetros mirando el test.**

## Estructura

```
rl_trader/          Código compartido (importable)
  env.py            Entorno del enunciado, sin modificar
  prices.py         Los 4 generadores de precios del enunciado, sin modificar
  agents.py         Discretización del estado, Q-learning, Sarsa y control Monte Carlo
  baselines.py      Aleatorio, buy & hold, momentum y contrarian
  evaluation.py     Conjuntos de series y métricas
  experiments.py    Ejecución de configuraciones en paralelo y elección de la mejor
  plots.py          Estilo común de las figuras (paleta, mapas de calor)
  data.py           Descarga de precios reales desde EODHD
  real.py           Activos reales, ventanas de entrenamiento y evaluación en test
scripts/            Cálculos largos que guardan resultados en results/
tests/              Comprobaciones rápidas (pytest)
notebooks/          Análisis (uno por fase o tema)
results/            Tablas de resultados (CSV)
figures/            Figuras para el informe
report/             Informe en Markdown (fuente única del PDF)
entregable/         PDF final para entregar
```

Los cálculos largos van en `scripts/` y guardan CSV en `results/`; los notebooks solo leen esos CSV, así que se ejecutan en segundos.

| Script | Tiempo (10 núcleos) | Genera | Lo analiza |
|---|---|---|---|
| `python scripts/fase2_ajuste.py` | ~40 min | `results/fase2_*.csv` | [notebooks/02_ajuste_hiperparametros.ipynb](notebooks/02_ajuste_hiperparametros.ipynb) |
| `python scripts/fase3_resultados.py` | ~15 min | `results/fase3_*.csv` | [notebooks/03_resultados_test.ipynb](notebooks/03_resultados_test.ipynb) |
| `python scripts/fase4_montecarlo.py` | ~47 min (16 procesos) | `results/fase4_mc_*.csv` | [notebooks/04_montecarlo.ipynb](notebooks/04_montecarlo.ipynb) |
| `python scripts/datos_reales.py` | ~10 min | `results/reales_*.csv` | [notebooks/05_datos_reales.ipynb](notebooks/05_datos_reales.ipynb) |
| `python scripts/figuras_informe.py` | segundos | `figures/informe_*.png` | el informe |
| `python scripts/informe_pdf.py` (sin LaTeX) | segundos | `report/informe_sin_latex.pdf` | — |

Los scripts de las fases 3 y 4 y el de datos reales leen `results/fase2_mejores.csv`, así que dependen del de la fase 2; el de las figuras del informe lee los resultados de las fases 3 y 4. El de datos reales necesita la clave de EODHD en `.env` la primera vez (descarga los precios a `data/`); en `results/` solo guarda fechas, métricas y políticas, nunca precios.

## Informe

El informe (máximo 2 páginas) está en [report/informe.md](report/informe.md) y se convierte a PDF con pandoc y XeLaTeX:

```bash
pandoc report/informe.md -o report/informe.pdf --pdf-engine=xelatex --resource-path=.
cp report/informe.pdf entregable/informe_taller_RL.pdf    # el PDF que se entrega
```

Hace falta [pandoc](https://pandoc.org) y una distribución de LaTeX con los paquetes `enumitem`, `caption` y `float`. Las fuentes son TeX Gyre Pagella y TeX Gyre Heros (clones de Palatino y Helvetica), que vienen con MacTeX, MiKTeX y TeX Live, así que el PDF sale igual en macOS, Windows y Linux. Después de editar, comprobad que sigue ocupando 2 páginas (`pdfinfo report/informe.pdf`).

En **Windows**, con MiKTeX (instala solo, sin administrador, los paquetes de LaTeX que falten) y pandoc desde pip:

```bash
winget install --id MiKTeX.MiKTeX -e --scope user
initexmf --set-config-value "[MPM]AutoInstall=1"
pip install pypandoc_binary    # pandoc queda en python -c "import pypandoc; print(pypandoc.get_pandoc_path())"
```

**Sin LaTeX**, `python scripts/informe_pdf.py` genera una versión aproximada a partir del mismo `informe.md` (pandoc a HTML e impresión con Edge o Chrome) en `report/informe_sin_latex.pdf`. La maquetación cambia un poco: comprobad que también ocupa 2 páginas.

## Cómo colaborar

- Una rama por tarea y *pull request* a `main`; nadie hace push directo a `main`.
- La lógica va en `rl_trader/`; los notebooks solo llaman a esas funciones y muestran resultados. Así los cambios se revisan bien y hay menos conflictos.
- Los notebooks dan muchos conflictos al fusionar: que cada uno trabaje en su propio notebook y borre las salidas pesadas antes de hacer commit.
- Fija semillas (`np.random.seed`) en todo lo que se vaya a comparar o a citar en el informe.

## Plan de trabajo

- [x] **Fase 1. Núcleo:** discretización de la tendencia, Q-learning, Sarsa, ε con decaimiento, baselines (aleatorio, buy & hold, reglas de momentum y contrarian), evaluación y métricas. Primera pasada en [notebooks/01_primera_pasada.ipynb](notebooks/01_primera_pasada.ipynb).
- [x] **Fase 2. Ajuste:** α, ε, γ, número de intervalos y de episodios, evaluados sobre series de validación. Configuraciones elegidas en `results/fase2_mejores.csv`; usar 4000 episodios. Análisis en [notebooks/02_ajuste_hiperparametros.ipynb](notebooks/02_ajuste_hiperparametros.ipynb).
- [x] **Fase 3. Resultados:** 10 semillas nuevas, 100 series de test por mercado, tabla comparativa (`results/fase3_tabla.csv`), comparaciones con intervalos de confianza (`results/fase3_comparaciones.csv`), curvas de aprendizaje y mapas de política. Análisis en [notebooks/03_resultados_test.ipynb](notebooks/03_resultados_test.ipynb).
- [x] **Fase 4. Extra: control Monte Carlo on-policy** (slides 2), primera visita y todas las visitas, con el mismo protocolo de ajuste y test que Q-learning y Sarsa, y un experimento controlado con los hiperparámetros de Sarsa. Análisis en [notebooks/04_montecarlo.ipynb](notebooks/04_montecarlo.ipynb).
- [x] **Extra: datos reales (EODHD).** S&P 500, IBEX 35, Bitcoin y EUR/USD; entrenamiento con ventanas de 500 días hasta 2018 y test con todo el periodo desde 2019. Sin ajuste con datos reales: configuraciones sin ajustar y de regímenes, 10 semillas. Análisis en [notebooks/05_datos_reales.ipynb](notebooks/05_datos_reales.ipynb).
- [ ] **Otros extras posibles:** sensibilidad a comisiones frente a γ, estado con más información.
- [ ] **Fase 5. Informe:** PDF de 2 páginas con diseño, resultados y análisis crítico. Fuente en [report/informe.md](report/informe.md); PDF para entregar en [entregable/informe_taller_RL.pdf](entregable/informe_taller_RL.pdf).
