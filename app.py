import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

try:
    import vtracer
    VTRACER_OK = True
except ImportError:
    VTRACER_OK = False

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


def black_bg_to_white(img_bgr, threshold=30, feather=0):
    """
    Detecta píxeles oscuros/negros (fondo) y los convierte a blanco puro,
    antes de aplicar la conversión a líneas. Útil cuando la imagen original
    tiene fondo negro y se quiere partir de un fondo blanco limpio.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray <= threshold).astype(np.uint8) * 255

    if feather > 0:
        # Suaviza el borde de la máscara para que la transición no se vea dura
        k = feather if feather % 2 == 1 else feather + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    mask_f = mask.astype(np.float32) / 255.0
    mask_f = mask_f[..., None]  # (H, W, 1) para aplicar a los 3 canales

    white = np.full_like(img_bgr, 255)
    result = (img_bgr.astype(np.float32) * (1 - mask_f) + white.astype(np.float32) * mask_f)
    return result.astype(np.uint8)


def sharpen_lines(line_img, strength=1.0):
    """Aumenta el contraste de las líneas resultantes."""
    result = cv2.convertScaleAbs(line_img, alpha=1 + strength, beta=0)
    return result


def clean_sharp_lines(line_img, bin_threshold=180, remove_specks=True,
                       min_speck_size=15, close_gaps=True, close_ksize=2):
    """
    Convierte una imagen de líneas (con grises/ruido) en blanco y negro puro,
    nítido, sin manchas sueltas. Ideal para lograr el look tipo 'coloring book'.
    """
    img = line_img.copy()

    # 1. Binarización dura: todo pixel queda 0 (negro) o 255 (blanco), sin grises
    _, binary = cv2.threshold(img, bin_threshold, 255, cv2.THRESH_BINARY)

    # 2. Cerrar pequeños huecos en las líneas (líneas más continuas y limpias)
    if close_ksize > 0 and close_gaps:
        # Trabajamos sobre las líneas en negro -> invertimos para usar operaciones morfológicas
        inverted = 255 - binary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        inverted = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)
        binary = 255 - inverted

    # 3. Eliminar manchas/puntitos de ruido sueltos (componentes pequeños)
    if remove_specks:
        inverted = 255 - binary  # líneas = blanco (255) sobre fondo negro (0)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            inverted, connectivity=8
        )
        clean_inverted = np.zeros_like(inverted)
        for label_id in range(1, num_labels):  # 0 es el fondo
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area >= min_speck_size:
                clean_inverted[labels == label_id] = 255
        binary = 255 - clean_inverted

    return binary


def vectorize_to_svg(line_img, colormode="binary", mode="spline",
                     filter_speckle=4, corner_threshold=60,
                     path_precision=6, length_threshold=4.0,
                     upscale=1):
    """
    Vectoriza una imagen de líneas (raster) a SVG usando vtracer.
    Devuelve el string SVG. El SVG escala sin pixelarse.

    - colormode: 'binary' (blanco/negro) o 'color'.
    - mode: 'spline' (curvas suaves), 'polygon' (rectas), 'none' (píxel).
    - filter_speckle: elimina manchas menores a este tamaño.
    - corner_threshold: ángulo para detectar esquinas (más alto = más suave).
    - path_precision: decimales de precisión de los trazos.
    - length_threshold: descarta segmentos muy cortos.
    - upscale: factor de ampliación previo (mejora el detalle del trazado).
    """
    # Asegurar 3 canales (vtracer trabaja sobre imagen tipo PNG)
    if len(line_img.shape) == 2:
        rgb = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = line_img

    if upscale > 1:
        rgb = cv2.resize(
            rgb, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
        )

    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    svg_str = vtracer.convert_raw_image_to_svg(
        img_bytes,
        img_format="png",
        colormode=colormode,
        mode=mode,
        filter_speckle=int(filter_speckle),
        corner_threshold=int(corner_threshold),
        path_precision=int(path_precision),
        length_threshold=float(length_threshold),
    )
    return svg_str


# ---------- Interfaz ----------

uploaded_file = st.file_uploader("Sube tu imagen", type=["png", "jpg", "jpeg", "webp"])

st.subheader("🎨 Preprocesamiento")
convertir_fondo_negro = st.checkbox(
    "Convertir fondo negro a blanco (antes de procesar)", value=False
)

if convertir_fondo_negro:
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        umbral_negro = st.slider(
            "Sensibilidad de negro (más alto = detecta más tonos oscuros)",
            0, 150, 30
        )
    with col_p2:
        suavizado_borde = st.slider(
            "Suavizar borde de transición (0 = corte duro)", 0, 21, 0, step=1
        )

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

st.markdown("---")
st.subheader("✨ Modo Nítido (blanco y negro puro, sin ruido)")
st.caption(
    "Actívalo si tu resultado sale grisáceo/con textura. Convierte todo a blanco "
    "o negro puro y elimina las manchitas sueltas, como en un dibujo para colorear limpio."
)

modo_nitido = st.checkbox("Activar modo nítido", value=True)

if modo_nitido:
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        bin_threshold = st.slider(
            "Umbral blanco/negro (más alto = líneas más finas/estrictas)",
            0, 255, 180
        )
        remove_specks = st.checkbox("Eliminar manchas/puntitos de ruido", value=True)
    with col_n2:
        min_speck_size = st.slider(
            "Tamaño mínimo de mancha a conservar (px)", 1, 200, 15
        )
        close_gaps = st.checkbox("Cerrar micro-huecos en las líneas", value=True)
    close_ksize = st.slider("Grosor de cierre de huecos", 1, 7, 2)

st.markdown("---")
st.subheader("🔷 Vectorización avanzada (SVG)")
st.caption(
    "Convierte las líneas en trazos vectoriales que escalan sin pixelarse. "
    "Ideal para imprimir en grande, tatuajes, corte láser/vinilo o editar en Illustrator/Inkscape."
)

if not VTRACER_OK:
    st.warning(
        "La librería 'vtracer' no está instalada. Agrega `vtracer` a tu "
        "requirements.txt para habilitar la vectorización SVG."
    )
    vectorizar = False
else:
    vectorizar = st.checkbox("Generar versión vectorial (SVG)", value=False)

if VTRACER_OK and vectorizar:
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        v_colormode = st.selectbox(
            "Modo de color",
            ["binary", "color"],
            help="'binary' = blanco y negro (líneas). 'color' = conserva colores."
        )
        v_mode = st.selectbox(
            "Estilo de trazo",
            ["spline", "polygon", "none"],
            help="'spline' = curvas suaves (recomendado). 'polygon' = rectas. 'none' = pixelado."
        )
        v_upscale = st.slider(
            "Ampliar antes de vectorizar (mejora detalle)", 1, 4, 2,
            help="Amplía la imagen antes de trazar; captura mejor los detalles finos."
        )
    with col_v2:
        v_filter_speckle = st.slider(
            "Filtrar manchas pequeñas", 0, 30, 4,
            help="Elimina puntitos/ruido menores a este tamaño."
        )
        v_corner_threshold = st.slider(
            "Suavidad de esquinas", 0, 180, 60,
            help="Más alto = curvas más suaves; más bajo = esquinas más marcadas."
        )
        v_length_threshold = st.slider(
            "Descartar trazos muy cortos", 0.0, 10.0, 4.0, step=0.5
        )

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    if convertir_fondo_negro:
        img_bgr = black_bg_to_white(img_bgr, threshold=umbral_negro, feather=suavizado_borde)

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

        if modo_nitido:
            result = clean_sharp_lines(
                result,
                bin_threshold=bin_threshold,
                remove_specks=remove_specks,
                min_speck_size=min_speck_size,
                close_gaps=close_gaps,
                close_ksize=close_ksize
            )

        if invertir_fondo:
            result = 255 - result

    col1, col2 = st.columns(2)
    with col1:
        preview_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        caption_original = (
            "Imagen con fondo convertido a blanco" if convertir_fondo_negro
            else "Imagen original"
        )
        st.image(preview_img, caption=caption_original, use_container_width=True)
    with col2:
        st.image(result, caption="Resultado en líneas", use_container_width=True, clamp=True)

    # Preparar descarga PNG
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

    # ---------- Vectorización SVG ----------
    if VTRACER_OK and vectorizar:
        st.markdown("### 🔷 Resultado vectorial (SVG)")
        with st.spinner("Vectorizando... (puede tardar unos segundos)"):
            try:
                # Para modo color se vectoriza la imagen a color; para binario, las líneas
                if v_colormode == "color":
                    source_for_vector = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                else:
                    source_for_vector = result

                svg_str = vectorize_to_svg(
                    source_for_vector,
                    colormode=v_colormode,
                    mode=v_mode,
                    filter_speckle=v_filter_speckle,
                    corner_threshold=v_corner_threshold,
                    length_threshold=v_length_threshold,
                    upscale=v_upscale,
                )

                # Vista previa del SVG dentro de un contenedor
                st.components.v1.html(
                    f'<div style="background:white; padding:10px; border-radius:8px;">{svg_str}</div>',
                    height=520,
                    scrolling=True,
                )

                st.download_button(
                    label="⬇️ Descargar vector (SVG)",
                    data=svg_str.encode("utf-8"),
                    file_name="imagen_vectorizada.svg",
                    mime="image/svg+xml"
                )
            except Exception as e:
                st.error(f"No se pudo vectorizar: {e}")
else:
    st.info("Sube una imagen para comenzar (JPG, PNG o WEBP).")

st.markdown("---")
st.caption(
    "Tip: 'XDoG' suele dar el look más parecido a un diseño de logo/tatuaje en línea "
    "(como el de tu dragón emplumado). 'Umbral Adaptativo' funciona bien para líneas "
    "gruesas tipo libro para colorear."
)
