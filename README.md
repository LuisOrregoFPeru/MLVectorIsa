🖊️ Imagen a Líneas — Convertidor con Streamlit

App para convertir fotos o diseños a color en dibujos de líneas limpios,
con remoción de fondo por IA y vectorización a SVG.
Ideal para logos, tatuajes, libros para colorear, impresión en grande o corte en vinilo.

📁 Archivos


app.py — Aplicación Streamlit (herramientas en la barra izquierda, resultado a la derecha).
requirements.txt — Dependencias.


🚀 Correr en local

bashpip install -r requirements.txt
streamlit run app.py

☁️ Desplegar en Streamlit Community Cloud


Sube app.py y requirements.txt a un repo de GitHub.
En share.streamlit.io → New app → elige el repo y app.py.
IMPORTANTE: en Advanced settings, selecciona Python 3.12
(la versión 3.14 causa un error de tipo Segmentation fault con estas librerías).
Deploy.


Si ya tienes la app desplegada: menú (⋮) → Settings → cambia Python version a 3.12.

🧰 Herramientas incluidas (en lenguaje sencillo)


Subir imagen — tu foto o diseño de partida.
Limpiar el fondo:

Quitar fondo con IA — un modelo inteligente reconoce tu figura y borra lo de atrás.
Fondo oscuro → blanco — convierte tonos negros en blanco (rápido).



Convertir a líneas — elige el estilo: tatuaje/logo, libro para colorear, contornos o lápiz.
Limpieza final — deja las líneas negras puras y borra los puntitos de ruido.
Vectorizar (SVG) — crea un archivo que se agranda sin pixelarse.


🛠️ Requisitos


Python 3.12 (recomendado)
Ver requirements.txt.


⚠️ Notas


La primera vez que uses "Quitar fondo con IA", la app descarga el modelo (~176 MB); tarda un poco solo esa vez.
El SVG a color puede pesar varios MB; usa modo "Blanco y negro" para archivos más livianos.
