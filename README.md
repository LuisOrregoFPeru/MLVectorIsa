# 🖊️ Imagen a Líneas — Convertidor con Streamlit

App para convertir cualquier imagen a color en un dibujo de líneas (line art),
con opción de **vectorización a SVG** para escalar sin pixelarse.
Útil para lograr un estilo tipo logo/tatuaje, libro para colorear, corte láser o vinilo.

## 📁 Archivos

- `app.py` — Código de la aplicación Streamlit.
- `requirements.txt` — Dependencias necesarias.

## 🚀 Cómo correrla en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Luego abre el navegador en `http://localhost:8501`.

## ☁️ Cómo desplegarla en Streamlit Community Cloud (gratis)

1. Crea un repositorio en GitHub y sube estos archivos (`app.py` y `requirements.txt`).
2. Ve a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
3. Haz clic en **"New app"**.
4. Selecciona tu repositorio, la rama (`main`) y el archivo principal: `app.py`.
5. Haz clic en **"Deploy"**. En un par de minutos tendrás tu URL pública.

## 🎛️ Funciones incluidas

### Preprocesamiento
- **Convertir fondo negro a blanco** antes de procesar (con sensibilidad y suavizado ajustables).

### Métodos de conversión a líneas
| Método | Resultado |
|---|---|
| Boceto a lápiz (Dodge Blend) | Líneas suaves con sombreado, estilo dibujo a mano |
| Bordes Canny | Contornos definidos, buen detalle de bordes |
| Umbral Adaptativo | Líneas gruesas, estilo libro para colorear |
| XDoG | Líneas tipo logo/tatuaje (recomendado para diseños como el del dragón) |

### Modo Nítido
Convierte a blanco y negro puro, elimina manchas de ruido y cierra micro-huecos.

### 🔷 Vectorización avanzada (SVG)
Usa **vtracer** para convertir las líneas raster en trazos vectoriales:
- Modo binario (blanco/negro) o color.
- Trazos tipo spline (curvas suaves), polígono (rectas) o pixelado.
- Controles de filtrado de manchas, suavidad de esquinas y ampliación previa.
- Descarga en `.svg` editable en Illustrator / Inkscape, escalable a cualquier tamaño.

## 🛠️ Requisitos

- Python 3.9+
- Ver `requirements.txt` para las librerías exactas.

## 💡 Flujo recomendado (para diseños tipo logo/tatuaje)

1. Activa "Convertir fondo negro a blanco" (sensibilidad 40–60).
2. Método **XDoG**.
3. Activa **Modo Nítido** (umbral 150–190).
4. Activa **Vectorización SVG** (modo binario, spline, ampliar x2).
5. Descarga el SVG.
