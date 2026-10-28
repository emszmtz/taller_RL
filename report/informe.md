---
# Para generar el PDF, desde la raíz del repositorio:
#   pandoc report/informe.md -o report/informe.pdf --pdf-engine=xelatex --resource-path=.
#   cp report/informe.pdf entregable/informe_taller_RL.pdf
#   (el título y los autores están en el bloque \begin{center} del principio del texto)
# Sin LaTeX (versión aproximada):  python scripts/informe_pdf.py  ->  report/informe_sin_latex.pdf
title-meta: "Q-learning, Sarsa y Monte Carlo para invertir en un activo"
author-meta: "Josep Perez Segura, Natalia SanJosé Ortega, Emilio Sánchez Martínez"
lang: es
fontsize: 10pt
geometry: "a4paper, top=1.3cm, bottom=1.3cm, left=1.6cm, right=1.6cm"
# TeX Gyre Pagella y Heros (clones de Palatino y Helvetica) vienen con MacTeX, MiKTeX y TeX Live:
# cargadas por fichero, el PDF sale igual en macOS, Windows y Linux
mainfont: texgyrepagella
mainfontoptions: [Extension=.otf, UprightFont=*-regular, BoldFont=*-bold, ItalicFont=*-italic, BoldItalicFont=*-bolditalic]
sansfont: texgyreheros
sansfontoptions: [Extension=.otf, UprightFont=*-regular, BoldFont=*-bold, ItalicFont=*-italic, BoldItalicFont=*-bolditalic]
mathfont: texgyrepagella-math.otf
colorlinks: true
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
  - \setlength{\intextsep}{3pt plus 1pt minus 2pt}
  - \makeatletter
  - \renewcommand\section{\@startsection{section}{1}{\z@}{5pt}{2pt}{\normalsize\bfseries\sffamily}}
  - \makeatother
  - \setlength{\parskip}{2pt}
  - \setlength{\parindent}{0pt}
  - \usepackage{enumitem}
  - \setlist{itemsep=0.5pt, topsep=1pt, leftmargin=1.2em}
  - \setlist[itemize,1]{label=\textbullet}
  - \setlist[itemize,2]{label=--}
  - \usepackage{caption}
  - \captionsetup{font=small, labelfont=bf, skip=2pt}
  - \AtBeginDocument{\renewcommand{\tablename}{Tabla}}
  - \pagenumbering{gobble}
---

\begin{center}
{\large\bfseries\sffamily Q-learning, Sarsa y Monte Carlo para invertir en un activo: cuatro mercados simulados}\par\vspace{3pt}
{\small Josep Perez Segura · Natalia SanJosé Ortega · Emilio Sánchez Martínez}\par
{\small\itshape Taller de Aprendizaje por Refuerzo · MIAX-14 · octubre de 2026}
\end{center}

**Resumen.** Entrenamos agentes tabulares Q-learning y Sarsa y, como extra, control Monte Carlo (MC) en el entorno del enunciado, sin modificarlo, con los cuatro generadores de precios. Los comparamos con el agente aleatorio, *buy & hold* (B&H) y dos reglas fijas (*momentum* y *contrarian*) en 100 series de test nuevas por mercado. Los agentes aprenden la estrategia esperada: seguir la tendencia donde hay inercia, ir contra ella donde hay reversión y no operar donde no hay señal. Superan a todas las referencias, salvo en el browniano, donde no hay nada que explotar. Con precios reales (2019–2026), en cambio, no baten a B&H.

# 1. Diseño experimental

- **Estado.** La tendencia es continua y su escala cambia mucho entre mercados (ruido del 10% por paso en el ciclo y del 1–2,5% en el resto). La dividimos en $k$ intervalos **simétricos, con el 0 como corte**, según cuantiles de |tendencia| de cada mercado: visitas parecidas por intervalo y ninguno mezcla signos; con la posición, $2k$ estados.
- **Acciones y $\gamma$.** El agente solo decide si está dentro o fuera, y no influye en la tendencia. Sin comisiones bastaría con $\gamma = 0$; $\gamma$ sirve para anticipar lo que costará entrar y salir.
- **Algoritmos.** Q-learning, Sarsa y, como extra, control MC *on-policy* de primera visita o de todas las visitas (slides 2). Todos son $\varepsilon$-greedy, con $\varepsilon$ decreciendo exponencialmente de 1 a $\varepsilon_f$, y usan $\alpha$ constante o $\alpha = 1/n(s,a)^{\omega}$ con $0{,}5 < \omega \le 1$, que cumple $\sum \alpha = \infty$ y $\sum \alpha^2 < \infty$.
- **Datos y métricas.** Cada episodio usa una serie nueva de 500 pasos; validación y test (100 series cada uno) usan semillas disjuntas y son comunes a todos los agentes. Métrica principal: la mediana del retorno (en el ciclo, unas pocas series dominan la media). **Convergencia:** episodio desde el que la mediana en validación queda a menos de un 10% (o 1 punto) de su valor final.

**Estrategia esperada.** Antes de entrenar medimos el retorno medio del paso siguiente en cada intervalo de tendencia: mucha inercia en el ciclo (de −2,5% a +4,5%), nada en el browniano, señal contraria en la reversión (solo los extremos superan la comisión) y una señal débil en regímenes. Esperamos *momentum*, no operar, *contrarian* selectivo y, en regímenes, poco que ganar.

# 2. Ajuste de hiperparámetros

Por etapas y solo en validación (3 semillas × 100 series; 864 entrenamientos para TD y 660 para MC):

- **Etapa A:** $\alpha \in \{0{,}01;\ 0{,}001;\ 1/n^{0,6};\ 1/n^{0,8};\ 1/n\}$ × $\gamma \in \{0;\ 0{,}5;\ 0{,}9;\ 0{,}99\}$ (× primera visita o todas, en MC).
- **Etapa B:** con lo mejor de A, $k \in \{4, 6, 10, 16, 20\}$ × tres calendarios de $\varepsilon$.
- **Etapa C:** con 6000 episodios, la validación se estabiliza hacia los 4000 en todos los mercados: entrenamos con **4000** (el ajuste usó 2000 por coste, lo que puede penalizar a los $\alpha$ pequeños).

**Elegidos:** $\gamma = 0$ en ciclo y browniano (la señal es de un paso y un $\gamma$ alto solo añade ruido); $\gamma = 0{,}5$ y $k$ = 16–20 en reversión; $\alpha = 0{,}001$ constante, $\gamma \geq 0{,}9$ y $k$ = 16–20 en regímenes, donde los valores por defecto casi no operaban. MC elige lo mismo, salvo $\gamma = 0{,}5$ en regímenes.

# 3. Resultados en test

\small

| Agente | Tendencia + ciclo | Browniano | Reversión | Regímenes |
|:--------------------------|--------------------:|-----------------:|------------------:|------------------:|
| Aleatorio | +456 (1,3) | −14,7 (−0,3) | −15,2 (−0,3) | −23,9 (−0,6) |
| Buy & hold | +3.425 (1,9) | +0,7 (0,1) | −1,1 (0,1) | −20,4 (−0,3) |
| Momentum | +48.189 (3,1) | −11,8 (−0,2) | −28,7 (−0,6) | −18,4 (−0,4) |
| Contrarian | −92,9 (−0,7) | −14,3 (−0,2) | +6,9 (0,3) | −22,6 (−0,5) |
| **Q-learning** | **+64.989 (3,2)** | 0 (−0,1) | **+22,0 (0,5)** | **+1,7 (0,3)** |
| **Sarsa** | **+67.861 (3,2)** | 0 (0,0) | **+22,2 (0,5)** | **+4,7 (0,2)** |
| **Monte Carlo** | **+53.896 (3,0)** | 0 (0,0) | **+24,9 (0,6)** | 0 (0,2) |
| *Convergencia Q-learning* | 2600 ep. · 24 s | 500 ep. · 5 s | 3300 ep. · 30 s | 2900 ep. · 28 s |
| *Convergencia Sarsa* | 600 ep. · 4 s | 200 ep. · 2 s | 3700 ep. · 29 s | 2000 ep. · 16 s |
| *Convergencia MC* | 3100 ep. | 600 ep. | 3500 ep. | 3800 ep. |

: Mediana del retorno porcentual en test y Sharpe medio; 100 series por mercado (× 10 semillas nuevas para RL). Convergencia: mediana entre semillas; sin segundos para MC, que se entrenó en otro equipo (el coste por episodio es similar).

\normalsize

![Capital final en test por agente y mercado: mediana (punto) y percentiles 25–75 (línea).](figures/informe_capital.png){width=94%}

"×": cociente geométrico medio del capital final entre dos agentes en las mismas series de test, con IC bootstrap del 95%. Los agentes de RL superan al aleatorio en todos los mercados (fuera del ciclo, este pierde un 15–24%).

- **Ciclo: *momentum* con histéresis, el esperado.** Superan a *momentum* (Sarsa ×1,37 [1,15; 1,61]) y a B&H (×18) con un tercio de sus operaciones: compran con tendencia muy positiva y venden solo si es muy negativa (Fig. 2).
- **Browniano: no operar, el esperado (con un matiz, §4).** Salvo 1 semilla de Q-learning, no compran, y evitan lo que pierden las reglas en comisiones (−12% a −15%). Frente a B&H, ×1,01 [0,93; 1,10].
- **Reversión: *contrarian* selectivo, el esperado.** Compran en los intervalos más bajistas y mantienen hasta los más alcistas: ×1,14 [1,09; 1,20] frente a la regla *contrarian*, con menos de un cuarto de sus operaciones.
- **Regímenes: mejor de lo esperado.** Compran con tendencia positiva moderada y venden ante movimientos extremos en cualquier sentido: el tamaño de la tendencia aproxima la volatilidad, que separa el régimen alcista tranquilo del bajista volátil. Ganan a no operar (Q-learning ×1,06 [1,03; 1,10]) y con holgura a B&H y *momentum* (×1,38 y ×1,30).

![Políticas de Sarsa: fracción de las 10 semillas dentro del mercado en cada estado (tendencia × posición, sin o con el activo). Las de Q-learning y MC son muy parecidas, salvo MC en regímenes, que apenas compra.](figures/informe_politicas.png){width=68%}

**Q-learning, Sarsa y Monte Carlo.** Q-learning y Sarsa no difieren de forma significativa: con $\gamma = 0$ hacen la misma actualización, y explorar cuesta como mucho una comisión (no hay un "acantilado" como en *Cliff Walking*). Para MC aislamos el efecto del objetivo, el retorno real $G_t$ frente a $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$, entrenándolo con los hiperparámetros de Sarsa. Con $\gamma = 0$ dan la misma política (×1,00). Con $\gamma = 0{,}5$ (reversión), MC mejora (×1,04 [1,02; 1,05]) y varía menos entre semillas (medianas del 22% al 28%, frente al 13–25% de Sarsa): no hace *bootstrap* sobre un estado no markoviano, al que le falta la distancia a la media. Con $\gamma = 0{,}99$ (regímenes) empeora (×0,94 [0,91; 0,97]), porque $G_t$ suma decenas de retornos ruidosos. Es el compromiso sesgo-varianza entre MC y TD visto en clase.

![Curvas de aprendizaje: mediana del retorno en validación (media de 10 semillas), cada 200 episodios; el punto marca la convergencia.](figures/informe_curvas.png){width=82%}

**Convergencia (Figura 3).** Con un coste por episodio similar, el tiempo es proporcional a los episodios: TD converge en 200–3700 (2–30 s) y MC necesita más en tres mercados, y en regímenes aún mejora a los 4000 (otra cara de su varianza). El ajuste ralentiza a TD ($\alpha$ menores), pero mejora la configuración por defecto en ciclo (×1,18–1,19) y regímenes (×1,05).

# 4. Análisis crítico y conclusiones

- **Conclusión principal.** Con señal, los tres algoritmos encuentran la estrategia esperada y la mejoran con histéresis; sin señal, no operan. Las diferencias entre ellos se explican por la teoría (sesgo y varianza), no por el azar.
- **Lo "esperado" depende de la métrica.** En el browniano, la recompensa del entorno (retorno simple) tiene esperanza positiva (+0,01% por paso), y para ella, con $\gamma$ alto, lo óptimo es B&H. Pero el log-retorno la tiene negativa $(\mu - \sigma^2/2 = -0{,}01\%)$ y la mediana del capital cae. Con $\gamma = 0{,}99$, 13 de 30 entrenamientos aprendieron B&H (media en validación de +9%, mediana negativa): elegir por mediana favorece no operar.
- **El ajuste en validación es optimista.** En reversión, la mejora en validación (26% frente a 19–21%) desaparece en test (×1,01 y ×1,03, no significativas), y en ciclo el MC elegido rinde menos que con la configuración de Sarsa (×0,87): elegir entre muchas configuraciones con 3 semillas sobreajusta, y la semilla pesa mucho (Q-learning en reversión: del 1% al 29%).
- **Datos reales (EODHD).** S&P 500, IBEX 35, Bitcoin y EUR/USD: entrenamiento hasta 2018 y test de 2019 a 2026 como un único episodio, con la configuración de regímenes. Ningún agente bate con claridad a B&H: Q-learning lo iguala en el S&P 500 (+228% frente a +232%) comprando en las caídas, pero queda muy atrás en el IBEX (+9% frente a +117%) y en Bitcoin (+767% frente a +1.850%), aunque en 2005–2018 superaba a B&H en los cuatro. Con una sola historia por activo, la tabla Q memoriza un pasado que no se repite.
- **Limitaciones.** Al estado le faltan la distancia a la media y la volatilidad; la comisión es fija.
- **Además de lo pedido:** control MC, datos reales, análisis de la señal, dos reglas más, ajuste por etapas, IC pareados y tests.
