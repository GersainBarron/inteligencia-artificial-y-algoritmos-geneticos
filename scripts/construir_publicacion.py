"""Arma la carpeta publica para GitHub: codigo, visor web estatico (docs/) y licencias.

Uso:  python scripts/construir_publicacion.py
Se puede volver a correr: reemplaza el codigo y docs/, no toca README ni .git del destino.
"""
import gzip, json, shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DEST = RAIZ.parent / "inteligencia artificial y algoritmos geneticos"
RUNS = Path(r"D:\redes neuronales\runs")
CORRIDAS_WEB = ["A-GA_s01", "B-GA_s01", "C-GA_s01", "R-GA_s01", "A-ES_s01", "B-ES_s01", "C-ES_s01"]
GENS_POR_CORRIDA = 6
IGNORAR = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.blend", "*.blend1", "*.blend@",
                                 "*.png", "*.pdf", "referencias", "*.pkl")


def copiar(rel):
    o, d = RAIZ / rel, DEST / rel
    if d.exists():
        shutil.rmtree(d) if d.is_dir() else d.unlink()
    if o.is_dir():
        shutil.copytree(o, d, ignore=IGNORAR)
    else:
        d.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(o, d)


def parchar_visor(docs):
    js = (docs / "visor.js").read_text(encoding="utf-8")
    cambios = [
        ('json("/api/corridas")', 'json("datos/corridas.json")'),
        ("json(`/api/corridas/${id}/replays`)", "json(`datos/${id}/replays.json`)"),
        ("`/runs/${corrida}/replays/${archivo}`", "`datos/${corrida}/${archivo}`"),
        ('json("/" + archivo)', "json(archivo)"),
        ('"/modelos/', '"modelos/'), ("`/modelos/", "`modelos/"),
    ]
    for a, b in cambios:
        if a not in js:
            raise SystemExit(f"no encontre en visor.js: {a}")
        js = js.replace(a, b)
    (docs / "visor.js").write_text(js, encoding="utf-8")
    html = (docs / "index.html").read_text(encoding="utf-8").replace('href="/"', 'href="./"')
    marca = '<a href="./" class="marca">evocarros</a>'
    if marca not in html:
        raise SystemExit("no encontre la marca en index.html")
    html = html.replace(marca, marca + ' <a href="metodo.html" style="color:#ff6b35;margin-left:1rem;'
                        'text-decoration:none;font-weight:600">Método y matemáticas</a>')
    shutil.copy2(RAIZ / "web" / "metodo.html", docs / "metodo.html")
    (docs / "index.html").write_text(html, encoding="utf-8")
    (docs / "panel.html").unlink(missing_ok=True)   # el panel necesita el servidor local
    (docs / ".nojekyll").write_text("", encoding="utf-8")


def construir_docs():
    docs = DEST / "docs"
    if docs.exists():
        shutil.rmtree(docs)
    shutil.copytree(RAIZ / "visor", docs, ignore=IGNORAR)
    parchar_visor(docs)
    datos = docs / "datos"; datos.mkdir()
    lista, pistas = [], set()
    for c in CORRIDAS_WEB:
        gens = sorted((RUNS / c / "replays").glob("gen_*.json*"))
        if not gens:
            continue
        paso = max(1, (len(gens) - 1) / (GENS_POR_CORRIDA - 1))
        elegidas = sorted({gens[min(len(gens) - 1, round(i * paso))] for i in range(GENS_POR_CORRIDA)})
        (datos / c).mkdir()
        for g in elegidas:
            shutil.copy2(g, datos / c / g.name)
            with (gzip.open(g, "rt", encoding="utf-8") if g.suffix == ".gz" else open(g, encoding="utf-8")) as f:
                pistas.add(json.load(f)["pista_archivo"])
        (datos / c / "replays.json").write_text(json.dumps([g.name for g in elegidas]), encoding="utf-8")
        lista.append({"id": c, "n_replays": len(elegidas)})
    (datos / "corridas.json").write_text(json.dumps(lista, indent=1), encoding="utf-8")
    for p in pistas:
        (docs / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RAIZ / p, docs / p)
    usados = {"hermanos_rodriguez.glb" if Path(p).parent.name == "mexico" else Path(p).parent.name + ".glb"
              for p in pistas} | {"carro.glb"}
    for glb in (docs / "modelos").glob("*.glb"):
        if glb.name not in usados:
            glb.unlink()
    mb = sum(f.stat().st_size for f in docs.rglob("*") if f.is_file()) / 2**20
    print(f"docs/: {len(lista)} corridas, pistas {sorted(pistas)}, {mb:.1f} MB")


def main():
    DEST.mkdir(exist_ok=True)
    for rel in ["evocarros", "scripts", "sandbox", "configs", "tests", "pistas", "blender", "resultados", "web",
                "visor", "panel.bat", "manejar.bat", "ANALISIS_FASE8.md", "REFERENCIAS_BLENDER.md"]:
        if (RAIZ / rel).exists():
            copiar(rel)
    construir_docs()
    (DEST / ".gitignore").write_text(
        "__pycache__/\n*.pyc\n.pytest_cache/\nruns/\n*.pkl\n*.mp4\n*.blend1\n*.blend@\n", encoding="utf-8")
    print("listo:", DEST)


if __name__ == "__main__":
    main()
