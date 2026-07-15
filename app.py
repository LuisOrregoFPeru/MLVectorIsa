import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(page_title="Imagen a Líneas", page_icon="🖊️", layout="wide")

st.title("🖊️ Convertidor de Imagen a Dibujo de Líneas")
st.caption("Sube una imagen a color y conviértela en line art / dibujo para colorear")

# ---------- Funciones de conversión ----------

def pencil_sketch(img_bgr, blur_ksize=21, sigma=0):
    """Método clásico 'dodge blend' -> boceto a lápiz / line art suave."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (blur_ksize, blur_ksize), sigma)
    inverted_blur = 255 - blurred
    # Evitar división por cero
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    return sketch


def canny_lines(img_bgr, low_thresh=50, high_thresh=150, blur_ksize=5):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    # Invertir para que las líneas sean negras sobre fondo blanco
    lines = 255 - edges
    return lines


def adaptive_threshold_lines(img_bgr, block_size=9, c_value=7, blur_ksize=5):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, blur_ksize)
    if block_size % 2 == 0:
        block_size += 1  # debe ser impar
    edges = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size, c_value
    )
    return edges


def xdog_lines(img_bgr, sigma=0.5, k=1.6, gamma=0.98, epsilon=0.1, phi=10):
    """Método XDoG: líneas más artísticas, similar a comic/tattoo linework."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    diff = g1 - gamma * g2

    xdog = np.where(
        diff >= epsilon,
        1.0,
        1.0 + np.tanh(phi * diff)
    )
    xdog = (np.clip(xdog, 0, 1) * 255).astype(np.uint8)
    return xdog


def sharpen_lines(line_img, strength=1.0):
    """Aumenta el contraste de las líneas resultantes."""
    result = cv2.convertScaleAbs(line_img, alpha=1 + strength, beta=0)
    return result


# ---------- Interfaz ----------

uploaded_file = st.file_uploader("Sube tu imagen", type=["png", "jpg", "jpeg", "webp"])

col_a, col_b = st.columns(2)

with col_a:
    metodo = st.selectbox(
        "Método de conversión",
        [
            "Boceto a lápiz (Dodge Blend)",
            "Bordes Canny",
            "Umbral Adaptativo",
            "XDoG (estilo artístico/tattoo)"
        ]
    )

with col_b:
    invertir_fondo = st.checkbox("Fondo negro / líneas blancas (como logo tattoo)", value=False)

st.subheader("Parámetros")

if metodo == "Boceto a lápiz (Dodge Blend)":
    blur_ksize = st.slider("Suavizado (impar)", 3, 51, 21, step=2)
elif metodo == "Bordes Canny":
    low_t = st.slider("Umbral bajo", 0, 255, 50)
    high_t = st.slider("Umbral alto", 0, 255, 150)
    blur_ksize = st.slider("Suavizado previo (impar)", 1, 15, 5, step=2)
elif metodo == "Umbral Adaptativo":
    block_size = st.slider("Tamaño de bloque (impar)", 3, 51, 9, step=2)
    c_value = st.slider("Constante C", -20, 20, 7)
    blur_ksize = st.slider("Mediana blur (impar)", 1, 15, 5, step=2)
else:  # XDoG
    sigma = st.slider("Sigma", 0.1, 3.0, 0.5)
    k = st.slider("k (multiplicador sigma)", 1.0, 3.0, 1.6)
    gamma = st.slider("Gamma", 0.8, 1.2, 0.98)
    epsilon = st.slider("Epsilon", 0.0, 1.0, 0.1)
    phi = st.slider("Phi (contraste de línea)", 1, 50, 10)

grosor_extra = st.slider("Reforzar contraste de líneas", 0.0, 2.0, 0.5, step=0.1)

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with st.spinner("Procesando imagen..."):
        if metodo == "Boceto a lápiz (Dodge Blend)":
            result = pencil_sketch(img_bgr, blur_ksize=blur_ksize)
        elif metodo == "Bordes Canny":
            result = canny_lines(img_bgr, low_t, high_t, blur_ksize)
        elif metodo == "Umbral Adaptativo":
            result = adaptive_threshold_lines(img_bgr, block_size, c_value, blur_ksize)
        else:
            result = xdog_lines(img_bgr, sigma, k, gamma, epsilon, phi)

        if grosor_extra > 0:
            result = sharpen_lines(result, grosor_extra)

        if invertir_fondo:
            result = 255 - result

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Imagen original", use_container_width=True)
    with col2:
        st.image(result, caption="Resultado en líneas", use_container_width=True, clamp=True)

    # Preparar descarga
    result_pil = Image.fromarray(result)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="⬇️ Descargar imagen de líneas (PNG)",
        data=byte_im,
        file_name="imagen_lineas.png",
        mime="image/png"
    )
else:
    st.info("Sube una imagen para comenzar (JPG, PNG o WEBP).")

st.markdown("---")
st.caption(
    "Tip: 'XDoG' suele dar el look más parecido a un diseño de logo/tatuaje en línea "
    "(como el de tu dragón emplumado). 'Umbral Adaptativo' funciona bien para líneas "
    "gruesas tipo libro para colorear."
)
