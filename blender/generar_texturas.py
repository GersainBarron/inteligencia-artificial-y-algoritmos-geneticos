"""Genera las texturas de anuncios y banderas del autódromo (requiere Pillow).

    conda activate evocarros
    python blender/generar_texturas.py

Salidas en `blender/texturas/`:
  anuncios.png        mosaico 16:1 que se repite a lo largo de muros y gradas:
                      nombre del autor y dirección de su página
  cartel.png          espectacular 2:1 con ambos textos
  dirigible.png       costado del dirigible, 5:1, con ambos textos
  publico.png         mosaico de espectadores (figuras genéricas de colores) para las gradas
  pancartas.png       cuatro pancartas de aficionados apiladas, 4:1 cada una
  bandera_mexico.png  bandera de México con el escudo simplificado (no es el
                      escudo oficial: es una figura esquemática para baja resolución)

Los textos son del autor del proyecto; no se usa ninguna marca de terceros.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

NOMBRE = "Gersain Barron"
WEB = "www.gersadesign.com"
SALIDA = Path(__file__).resolve().parent / "texturas"
VERDE, BLANCO, ROJO = (0, 104, 71), (255, 255, 255), (206, 17, 38)
AZUL, AMARILLO = (10, 40, 130), (255, 205, 0)


def fuente(tam: int, negrita: bool = True):
    for nombre in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if negrita else ("arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(nombre, tam)
        except OSError:
            continue
    return ImageFont.load_default()


def texto_centrado(d: ImageDraw.ImageDraw, caja, texto, color, alto_max):
    """Escribe el texto lo más grande posible dentro de la caja (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = caja
    tam = alto_max
    while tam > 8:
        f = fuente(tam)
        l, t, r, b = d.textbbox((0, 0), texto, font=f)
        if r - l <= (x1 - x0) * 0.92 and b - t <= (y1 - y0) * 0.80:
            break
        tam -= 2
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), texto, font=f, fill=color, anchor="mm")


def bandera(ancho: int, alto: int) -> Image.Image:
    im = Image.new("RGB", (ancho, alto), BLANCO)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, ancho // 3, alto), fill=VERDE)
    d.rectangle((ancho - ancho // 3, 0, ancho, alto), fill=ROJO)
    cx, cy, r = ancho / 2, alto / 2, alto * 0.20
    d.arc((cx - r * 1.25, cy - r * 0.9, cx + r * 1.25, cy + r * 1.35), 20, 160, fill=VERDE, width=max(2, alto // 28))  # guirnalda
    d.ellipse((cx - r * 0.75, cy - r * 0.95, cx + r * 0.75, cy + r * 0.55), fill=(122, 82, 40))                      # águila (esquema)
    d.ellipse((cx - r * 0.30, cy - r * 1.15, cx + r * 0.25, cy - r * 0.60), fill=(96, 62, 28))
    return im


def main() -> None:
    SALIDA.mkdir(exist_ok=True)
    # mosaico 16:1: mitad azul con el nombre, mitad blanca con la página; una banderita en cada mitad
    w, h = 2048, 128
    im = Image.new("RGB", (w, h), AZUL)
    d = ImageDraw.Draw(im)
    d.rectangle((w // 2, 0, w, h), fill=BLANCO)
    for x0, fondo, texto, color in ((0, AZUL, NOMBRE, BLANCO), (w // 2, BLANCO, WEB, AZUL)):
        b = bandera(126, 72)
        im.paste(b, (x0 + 28, (h - 72) // 2))
        d.rectangle((x0 + 28, (h - 72) // 2, x0 + 28 + 126, (h - 72) // 2 + 72), outline=(40, 40, 40))
        texto_centrado(d, (x0 + 170, 0, x0 + w // 2 - 10, h), texto, color, 110)
    d.rectangle((0, 0, w - 1, 5), fill=VERDE); d.rectangle((0, h - 6, w - 1, h - 1), fill=ROJO)
    im.save(SALIDA / "anuncios.png")

    # espectacular 2:1
    w, h = 1024, 512
    im = Image.new("RGB", (w, h), AZUL)
    d = ImageDraw.Draw(im)
    texto_centrado(d, (0, 40, w, 270), NOMBRE, BLANCO, 190)
    texto_centrado(d, (0, 270, w, 420), WEB, AMARILLO, 110)
    for k, c in enumerate((VERDE, BLANCO, ROJO)):
        d.rectangle((k * w // 3, h - 60, (k + 1) * w // 3, h), fill=c)
    d.rectangle((0, 0, w - 1, h - 1), outline=BLANCO, width=8)
    im.save(SALIDA / "cartel.png")

    bandera(512, 292).save(SALIDA / "bandera_mexico.png")

    # costado del dirigible, 5:1
    w, h = 1280, 256
    im = Image.new("RGB", (w, h), AZUL)
    d = ImageDraw.Draw(im)
    texto_centrado(d, (0, 6, w, 150), NOMBRE, BLANCO, 150)
    texto_centrado(d, (0, 140, w, 236), WEB, AMARILLO, 90)
    for k, c in enumerate((VERDE, BLANCO, ROJO)):
        d.rectangle((k * w // 3, h - 18, (k + 1) * w // 3, h), fill=c)
    im.save(SALIDA / "dirigible.png")

    # público: mosaico de 12.8 x 1.7 m (un escalón de grada, dos filas de butacas) visto desde arriba
    import random
    azar = random.Random(2026)
    w, h = 512, 64
    im = Image.new("RGB", (w, h), (15, 23, 51))
    d = ImageDraw.Draw(im)
    ropa = [(214, 40, 40), (245, 245, 245), (20, 120, 70), (240, 190, 30), (40, 90, 200), (230, 120, 30), (150, 60, 160), (60, 60, 60), (0, 104, 71), (206, 17, 38)]
    piel = [(224, 172, 135), (198, 134, 96), (141, 85, 56), (241, 194, 160)]
    for fila in range(2):
        for col in range(26):
            x, y = col * w / 26, fila * h / 2
            d.rectangle((x + 2, y + 3, x + w / 26 - 3, y + h / 2 - 3), fill=(26, 38, 80))          # butaca
            if azar.random() < 0.72:
                d.rectangle((x + 4, y + 9, x + w / 26 - 5, y + h / 2 - 5), fill=azar.choice(ropa))   # persona
                d.ellipse((x + 6, y + 4, x + w / 26 - 7, y + 15), fill=azar.choice(piel))
    im.save(SALIDA / "publico.png")
    # pancartas de los aficionados: cuatro de 4:1 apiladas (el modelo elige una por grupo)
    w, h = 512, 128
    im = Image.new("RGB", (w, 4 * h), BLANCO)
    d = ImageDraw.Draw(im)
    for k, (texto, fondo, color) in enumerate((("¡VAMOS GERSAIN!", VERDE, BLANCO), ("gersadesign.com", AMARILLO, AZUL),
                                                ("¡VIVA MÉXICO!", BLANCO, VERDE), ("GERSAIN #1", ROJO, BLANCO))):
        y = k * h
        d.rectangle((0, y, w, y + h), fill=fondo)
        if fondo == BLANCO:
            d.rectangle((0, y, 40, y + h), fill=VERDE); d.rectangle((w - 40, y, w, y + h), fill=ROJO)
        d.rectangle((0, y, w - 1, y + h - 1), outline=(30, 30, 30), width=3)
        texto_centrado(d, (44, y, w - 44, y + h), texto, color, 96)
    im.save(SALIDA / "pancartas.png")
    print("texturas en", SALIDA)


if __name__ == "__main__":
    main()
