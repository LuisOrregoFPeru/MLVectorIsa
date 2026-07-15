# 🖊️ Imagen a Líneas — Convertidor con Streamlit

App para convertir cualquier imagen a color en un dibujo de líneas (line art),
útil para lograr un estilo tipo logo/tatuaje o libro para colorear.

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

## 🎛️ Métodos de conversión incluidos

| Método | Resultado |
|---|---|
| Boceto a lápiz (Dodge Blend) | Líneas suaves con sombreado, estilo dibujo a mano |
| Bordes Canny | Contornos definidos, buen detalle de bordes |
| Umbral Adaptativo | Líneas gruesas, estilo libro para colorear |
| XDoG | Líneas tipo logo/tatuaje (recomendado para diseños como el del dragón) |

## 🛠️ Requisitos

- Python 3.9+
- Ver `requirements.txt` para las librerías exactas.
