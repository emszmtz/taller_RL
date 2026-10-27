"""PDF del informe sin LaTeX: report/informe.md -> HTML (pandoc) -> PDF (Edge o Chrome headless).

Alternativa a la orden de pandoc + xelatex de la cabecera de informe.md, para
equipos sin LaTeX (p. ej. Windows). El texto sale del mismo informe.md: los
bloques LaTeX se ignoran y la cabecera se lee del bloque \\begin{center}.

Requisitos: pip install pypandoc_binary, y Microsoft Edge o Google Chrome.
Uso, desde la raíz del repositorio:  python scripts/informe_pdf.py [salida.pdf]
Por defecto escribe entregable/informe_taller_RL.pdf.
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parent.parent
INFORME = ROOT / "report" / "informe.md"
SALIDA = ROOT / "entregable" / "informe_taller_RL.pdf"
NAVEGADORES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "msedge", "google-chrome", "chromium",
]

CSS = """
@page { size: A4; margin: 1.4cm 1.6cm; }
html { font-size: 9.5pt; }
body { font-family: "Palatino Linotype", Palatino, "Book Antiqua", serif; line-height: 1.24; color: #000;
       margin: 0; text-align: justify; hyphens: auto; counter-reset: figura tabla; }
.cabecera { text-align: center; margin-bottom: 5pt; }
.cabecera .titulo { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-weight: 600; font-size: 12.5pt; }
.cabecera .autores { font-size: 9pt; margin-top: 2pt; }
.cabecera .lugar { font-size: 9pt; font-style: italic; }
h1 { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-weight: 600; font-size: 10pt;
     margin: 5pt 0 1.5pt; break-after: avoid; }
p { margin: 0 0 2pt; }
ul { margin: 1pt 0 2pt; padding-left: 1.2em; }
li { margin: 0 0 0.5pt; }
li > p { margin: 0; }
math { font-size: 100%; }
figure { margin: 3pt 0; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 8.3pt; text-align: justify; margin-top: 2pt; }
figcaption::before { counter-increment: figura; content: "Figura " counter(figura) ": "; font-weight: bold; }
table { border-collapse: collapse; width: 100%; font-size: 8.3pt; margin: 3pt 0 5pt; break-inside: avoid;
        border-top: 1.2px solid #000; border-bottom: 1.2px solid #000; }
caption { caption-side: top; font-size: 8.3pt; text-align: justify; margin-bottom: 3pt; }
caption::before { counter-increment: tabla; content: "Tabla " counter(tabla) ": "; font-weight: bold; }
thead th { border-bottom: 0.8px solid #000; font-weight: normal; padding: 1.5pt 4pt; }
td { padding: 0.6pt 4pt; }
td, th { font-variant-numeric: tabular-nums; }
"""


def cabecera(md):
    """Título, autores y lugar a partir del bloque \\begin{center} de informe.md."""
    # Anclado al principio de línea: el comentario de la cabecera YAML también nombra el bloque
    bloque = re.search(r"^\\begin\{center\}(.*?)^\\end\{center\}", md, re.S | re.M).group(1)
    lineas = []
    for linea in bloque.strip().splitlines():
        linea = re.sub(r"\\par|\\vspace\{[^}]*\}", "", linea).strip()
        texto = re.sub(r"^\{(\\[a-zA-Z]+\s*)+", "", linea).rstrip("}").strip()
        lineas.append(texto)
    titulo, autores, lugar = lineas
    return (f'<div class="cabecera"><div class="titulo">{titulo}</div>'
            f'<div class="autores">{autores}</div><div class="lugar">{lugar}</div></div>')


def comas_decimales(html):
    """En TeX, 0{,}01 es un decimal; en MathML sale como 0 <mo>,</mo> 01, con el
    espacio de un separador. Se junta solo donde la fuente TeX usaba {,}."""
    def arreglar(m):
        formula = m.group(0)
        for a, b in re.findall(r"(\d+)\{,\}(\d+)", formula):
            formula = formula.replace(f"<mn>{a}</mn><mo>,</mo><mn>{b}</mn>", f"<mn>{a},{b}</mn>", 1)
        return formula
    return re.sub(r"<math\b.*?</math>", arreglar, html, flags=re.S)


def navegador():
    for ruta in NAVEGADORES:
        encontrado = shutil.which(ruta) or (ruta if Path(ruta).exists() else None)
        if encontrado:
            return encontrado
    raise SystemExit("No encuentro Edge ni Chrome para imprimir el PDF")


def main():
    salida = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else SALIDA
    salida.parent.mkdir(parents=True, exist_ok=True)
    md = INFORME.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "estilo.css").write_text(CSS, encoding="utf-8")
        (tmp / "cabecera.html").write_text(cabecera(md), encoding="utf-8")
        html = tmp / "informe.html"
        pypandoc.convert_file(
            str(INFORME), "html5", outputfile=str(html),
            extra_args=["--standalone", "--mathml", "--embed-resources", f"--resource-path={ROOT}",
                        f"--css={tmp / 'estilo.css'}", f"--include-before-body={tmp / 'cabecera.html'}",
                        "--metadata", "pagetitle=Informe taller RL", "--metadata", "lang=es"])
        html.write_text(comas_decimales(html.read_text(encoding="utf-8")), encoding="utf-8")
        subprocess.run([navegador(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={tmp / 'perfil'}", f"--print-to-pdf={salida}", html.as_uri()],
                       check=True, capture_output=True, timeout=120)
    print("PDF guardado en", salida)


if __name__ == "__main__":
    main()
