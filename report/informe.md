---
# Para generar el PDF, desde la raíz del repositorio:
#   pandoc report/informe.md -o report/informe.pdf --pdf-engine=xelatex --resource-path=.
#   (el título y los autores están en el bloque \begin{center} del principio del texto)
title-meta: "Q-learning y Sarsa para invertir en un activo"
author-meta: "Josep Perez Segura, Natalia SanJosé Ortega, Emilio Sánchez Martínez"
lang: es
fontsize: 10pt
geometry: "a4paper, margin=1.6cm"
mainfont: "Palatino"
sansfont: "Helvetica Neue"
colorlinks: true
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
  - \setlength{\intextsep}{4pt plus 2pt minus 2pt}
  - \makeatletter
  - \renewcommand\section{\@startsection{section}{1}{\z@}{5pt}{2pt}{\normalsize\bfseries\sffamily}}
  - \makeatother
  - \setlength{\parskip}{2.5pt}
  - \setlength{\parindent}{0pt}
  - \usepackage{enumitem}
  - \setlist{itemsep=1pt, topsep=1pt, leftmargin=1.2em}
  - \setlist[itemize,1]{label=\textbullet}
  - \setlist[itemize,2]{label=--}
  - \usepackage{caption}
  - \captionsetup{font=small, labelfont=bf, skip=3pt}
  - \AtBeginDocument{\renewcommand{\tablename}{Tabla}}
  - \pagenumbering{gobble}
---

\begin{center}
{\large\bfseries\sffamily Q-learning y Sarsa para invertir en un activo: cuatro mercados simulados}\par\vspace{3pt}
{\small Josep Perez Segura · Natalia SanJosé Ortega · Emilio Sánchez Martínez}\par
{\small\itshape Taller de Aprendizaje por Refuerzo · MIAX-14 · octubre de 2026}
\end{center}

**Resumen.** Entrenamos agentes tabulares Q-learning y Sarsa en el entorno del enunciado, sin modificarlo, con los cuatro generadores de precios. Los comparamos con el agente aleatorio, *buy & hold* y dos reglas fijas (*momentum* y *contrarian*) en 100 series de test nuevas por mercado. Los agentes aprenden la estrategia esperada: seguir la tendencia donde hay inercia, ir contra ella donde hay reversión y no operar donde no hay señal. Superan a todas las referencias, salvo en el browniano, donde no hay nada que explotar.

# 1. Diseño experimental

- **Estado.** La tendencia es continua y su escala cambia mucho entre mercados (ruido del 10% por paso en el de ciclo y del 1–2,5% en los demás). La dividimos en $k$ intervalos **simétricos, con el 0 como corte**, a partir de cuantiles de $|$tendencia$|$ en series de entrenamiento de cada mercado. Así cada intervalo recibe un número parecido de visitas y ninguno mezcla tendencias de signo distinto. Con la posición, el estado toma $2k$ valores.
- **Acciones y $\gamma$.** Vender sin el activo o comprar teniéndolo no hace nada: en la práctica el agente decide si está dentro o fuera. Además, no influye en la tendencia, solo en su posición. Sin comisiones bastaría con $\gamma = 0$; $\gamma$ sirve para que tenga en cuenta lo que costará entrar y salir.
- **Aprendizaje.** $\varepsilon$ decrece exponencialmente de 1 a $\varepsilon_f$. La tasa es constante o decreciente, $\alpha = 1/n(s,a)^{\omega}$ con $0{,}5 < \omega \le 1$, que cumple $\sum \alpha = \infty$ y $\sum \alpha^2 < \infty$. Con $\alpha = 0{,}05$ constante, el ruido de los retornos (1–10% por paso) tapaba señales del orden de la comisión (0,1%).
- **Datos y métricas.** Cada episodio usa una serie nueva de 500 pasos. Entrenamiento, validación (100 series) y test (100 series) salen de semillas disjuntas, y todos los agentes se evalúan sobre las mismas series de test (comparación pareada). La métrica principal es la mediana del retorno, porque en el mercado de ciclo la media la dominan unas pocas series. La **convergencia** es el episodio desde el que la mediana en validación no se aleja más de un 10% (o 1 punto) de su valor final.

**Hipótesis.** Antes de entrenar medimos el retorno medio del paso siguiente en cada intervalo de tendencia. En el ciclo hay mucha inercia (de −2,5% a +4,5% según el intervalo); en el browniano, nada; en la reversión, señal contraria, pero solo los intervalos extremos superan la comisión; en regímenes, una señal débil, porque las tendencias extremas aparecen sobre todo en el régimen bajista, que es más volátil.

# 2. Ajuste de hiperparámetros

Búsqueda por etapas, solo en validación (3 semillas × 100 series; 864 entrenamientos):

- **Etapa A:** $\alpha \in \{0{,}01;\ 0{,}001;\ 1/n^{0,6};\ 1/n^{0,8};\ 1/n\}$ × $\gamma \in \{0;\ 0{,}5;\ 0{,}9;\ 0{,}99\}$.
- **Etapa B:** con el mejor $\alpha$ y $\gamma$, $k \in \{4, 6, 10, 16, 20\}$ × tres calendarios de $\varepsilon$.
- **Etapa C:** 6000 episodios para ver cuándo se estabiliza el aprendizaje. De aquí sale el uso de **4000 episodios**.

Lo más relevante por mercado:

- **Ciclo:** $\gamma = 0$ y $k = 4$. La señal de un paso es enorme y un $\gamma$ alto solo añade ruido al objetivo. Con $\gamma = 0$, Q-learning y Sarsa hacen exactamente la misma actualización.
- **Reversión:** $\gamma = 0{,}5$ y $k$ = 16–20. **Browniano:** ninguna configuración mejora a no operar.
- **Regímenes:** $\alpha = 0{,}001$ constante, $\gamma \geq 0{,}9$ y $k$ = 16–20. Con la configuración por defecto ($\alpha = 1/n^{0,6}$, $\gamma = 0{,}9$, $k = 10$) el agente casi no operaba.

# 3. Resultados en test

\small

| Agente | Tendencia + ciclo | Browniano | Reversión | Regímenes |
|:--------------------------|--------------------:|-----------------:|------------------:|------------------:|
| Aleatorio | +456 (1,3) | −14,7 (−0,3) | −15,2 (−0,3) | −23,9 (−0,6) |
| Buy & hold | +3.425 (1,9) | +0,7 (0,1) | −1,1 (0,1) | −20,4 (−0,3) |
| Momentum | +48.189 (3,1) | −11,8 (−0,2) | −28,7 (−0,6) | −18,4 (−0,4) |
| Contrarian | −92,9 (−0,7) | −14,3 (−0,2) | +6,9 (0,3) | −22,6 (−0,5) |
| Q-learning sin ajustar | +46.432 (3,0) | 0 (0,0) | +19,9 (0,5) | 0 (0,1) |
| Sarsa sin ajustar | +46.432 (3,1) | 0 (0,0) | +17,2 (0,6) | 0 (0,0) |
| **Q-learning** | **+64.989 (3,2)** | 0 (−0,1) | **+22,0 (0,5)** | **+1,7 (0,3)** |
| **Sarsa** | **+67.861 (3,2)** | 0 (0,0) | **+22,2 (0,5)** | **+4,7 (0,2)** |
| *Convergencia Q-learning* | 2600 ep. · 24 s | 500 ep. · 5 s | 3300 ep. · 30 s | 2900 ep. · 28 s |
| *Convergencia Sarsa* | 600 ep. · 4 s | 200 ep. · 2 s | 3700 ep. · 29 s | 2000 ep. · 16 s |

: Mediana del retorno en test (%) y Sharpe medio entre paréntesis. 100 series por mercado; para los agentes de RL, 10 semillas nuevas (1000 episodios). *Sin ajustar*: configuración por defecto con 2000 episodios. Convergencia: mediana entre semillas, en episodios y segundos de entrenamiento.

\normalsize

![Capital final en test por agente y mercado: mediana (punto) y percentiles 25–75 (línea).](figures/informe_capital.png){width=100%}

En adelante, "×" es el cociente geométrico medio del capital final entre dos agentes en las mismas series de test, con su intervalo de confianza bootstrap del 95%.

- **Ciclo: momentum con histéresis.** Q-learning y Sarsa superan a la regla de *momentum* (×1,37 [1,15; 1,61] para Sarsa) con un tercio de sus operaciones (31–33 frente a 92). Compran solo con tendencia muy positiva y no venden hasta que es muy negativa (Figura 2).
- **Browniano: no operar.** Salvo 1 de las 10 semillas de Q-learning, los agentes no compran y evitan lo que pierden por comisiones las reglas (entre −12% y −15%). Frente a *buy & hold* no hay diferencia significativa (×1,01 [0,93; 1,10]).
- **Reversión: contrarian selectivo.** Compran solo en los intervalos más bajistas y mantienen hasta los más alcistas: ×1,14 [1,09; 1,20] frente a la regla *contrarian*, con menos de un cuarto de sus operaciones.
- **Regímenes: la volatilidad a través de la tendencia.** Compran con tendencia positiva moderada y venden solo ante movimientos extremos en cualquier dirección. El tamaño de la tendencia actúa como aproximación de la volatilidad, que separa el régimen alcista tranquilo del bajista volátil. La ventaja frente a no operar es pequeña pero significativa (Q-learning ×1,06 [1,03; 1,10]); frente a *buy & hold* y *momentum* es amplia (×1,38 y ×1,30).

![Políticas aprendidas por Sarsa: fracción de las 10 semillas que está dentro del mercado en cada estado (intervalo de tendencia × posición). Las de Q-learning son muy parecidas.](figures/informe_politicas.png){width=78%}

**Q-learning frente a Sarsa.** No hay diferencias atribuibles al algoritmo. Con $\gamma = 0$ (ciclo y browniano) son el mismo, y sus pequeñas diferencias (×0,98) vienen del $\alpha$ elegido o del azar de una semilla. Con $\gamma > 0$ (reversión y regímenes) no son significativas. Una acción exploratoria cuesta como mucho una comisión: no hay un "acantilado" como en *Cliff Walking* que lleve a Sarsa a una política más prudente.

**Convergencia.** Se alcanza entre los 200 y los 3700 episodios (2–30 s), según el mercado. Las configuraciones ajustadas convergen más despacio que las sin ajustar, porque su $\alpha$ es más pequeño. A cambio, terminan mejor en ciclo (×1,18–1,19) y en regímenes (×1,05).

# 4. Análisis crítico y conclusiones

- **Los agentes aprenden lo que hay que aprender.** Cuando hay señal, encuentran la estrategia esperada y la mejoran con histéresis. Cuando no la hay, no operan, que es lo correcto con comisiones.
- **El ajuste en validación es optimista.** En reversión, la mejora vista en validación (26% frente a 19–21%) desaparece en test (×1,01 y ×1,03, no significativas): elegir la mejor de 35 configuraciones con 3 semillas sobreajusta a la validación. Además, el resultado depende mucho de la semilla: en reversión, la mediana de Q-learning va del 1% al 29%.
- **Limitaciones.** Al estado le falta información clave: la distancia a la media de largo plazo (en reversión) y la volatilidad (en regímenes). Los mercados son simulados con parámetros y comisión fijos, y el ruido del 10% por paso del ciclo da retornos poco realistas.
- **Además de lo pedido:** análisis previo de la señal, dos reglas de referencia más, ajuste por etapas, 10 semillas nuevas en test, intervalos de confianza pareados y criterio de convergencia explícito; código reproducible y con tests.
