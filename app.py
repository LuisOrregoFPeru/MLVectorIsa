import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
 
# ---------- Librerías opcionales (IA y vectorización) ----------
try:
    import vtracer
    VTRACER_OK = True
except ImportError:
    VTRACER_OK = False
 
try:
    from rembg import remove, new_session
    REMBG_OK = True
except ImportError:
    REMBG_OK = False
 
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_OK = True
except ImportError:
    CANVAS_OK = False
 
 
st.set_page_config(page_title="Imagen a Líneas", page_icon="🖊️", layout="wide")
 
st.title("🖊️ Convertidor de Imagen a Líneas")
st.caption(
    "Sube una foto o diseño a color y conviértelo en un dibujo de líneas limpio, "
    "listo para colorear, imprimir en grande o usar como logo/tatuaje."
)
 
 
# ============================================================
#   FUNCIONES DE PROCESAMIENTO
# ============================================================
 
@st.cache_resource(show_spinner=False)
def get_rembg_session(model_name="u2net"):
    """Carga (una sola vez) el modelo de IA para remover fondos."""
    return new_session(model_name)
 
 
def ai_remove_background(img_bgr, model_name="u2net", bg_color=(255, 255, 255)):
    """Usa IA para separar la figura del fondo y pintar el fondo de un color."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    session = get_rembg_session(model_name)
    cut = remove(pil_img, session=session)  # RGBA con fondo transparente
    bg = Image.new("RGBA", cut.size, bg_color + (255,))
    composed = Image.alpha_composite(bg, cut).convert("RGB")
    result_rgb = np.array(composed)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
 
 
def black_bg_to_white(img_bgr, threshold=30, feather=0):
    """Convierte los tonos oscuros (fondo negro) en blanco puro."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray <= threshold).astype(np.uint8) * 255
    if feather > 0:
        k = feather if feather % 2 == 1 else feather + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask_f = mask.astype(np.float32) / 255.0
    mask_f = mask_f[..., None]
    white = np.full_like(img_bgr, 255)
    result = (img_bgr.astype(np.float32) * (1 - mask_f) +
              white.astype(np.float32) * mask_f)
    return result.astype(np.uint8)
 
 
def pencil_sketch(img_bgr, blur_ksize=21, sigma=0):
    """Boceto a lápiz (líneas suaves con sombreado)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (blur_ksize, blur_ksize), sigma)
    inverted_blur = 255 - blurred
    return cv2.divide(gray, inverted_blur, scale=256.0)
 
 
def canny_lines(img_bgr, low_thresh=50, high_thresh=150, blur_ksize=5):
    """Detección de bordes clásica (contornos definidos)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    return 255 - edges
 
 
def adaptive_threshold_lines(img_bgr, block_size=9, c_value=7, blur_ksize=5):
    """Líneas gruesas estilo libro para colorear."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, blur_ksize)
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, block_size, c_value
    )
 
 
def xdog_lines(img_bgr, sigma=0.5, k=1.6, gamma=0.98, epsilon=0.1, phi=10):
    """Líneas artísticas estilo logo/tatuaje."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    diff = g1 - gamma * g2
    xdog = np.where(diff >= epsilon, 1.0, 1.0 + np.tanh(phi * diff))
    return (np.clip(xdog, 0, 1) * 255).astype(np.uint8)
 
 
def sharpen_lines(line_img, strength=1.0):
    return cv2.convertScaleAbs(line_img, alpha=1 + strength, beta=0)
 
 
def clean_sharp_lines(line_img, bin_threshold=180, remove_specks=True,
                      min_speck_size=15, close_gaps=True, close_ksize=2):
    """Deja blanco y negro puro y elimina manchas de ruido."""
    img = line_img.copy()
    _, binary = cv2.threshold(img, bin_threshold, 255, cv2.THRESH_BINARY)
    if close_ksize > 0 and close_gaps:
        inverted = 255 - binary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        inverted = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)
        binary = 255 - inverted
    if remove_specks:
        inverted = 255 - binary
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            inverted, connectivity=8)
        clean_inverted = np.zeros_like(inverted)
        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] >= min_speck_size:
                clean_inverted[labels == label_id] = 255
        binary = 255 - clean_inverted
    return binary
 
 
def erase_strokes_at_points(line_img, points_xy, tolerance=8):
    """
    Borra el trazo (línea) completo que está debajo de cada punto donde el
    usuario hizo clic. Usa 'componentes conectados': entiende qué píxeles
    forman una misma línea y la borra entera.
    'points_xy' son coordenadas (x, y) en la resolución de 'line_img'.
    """
    if len(line_img.shape) == 3:
        gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = line_img.copy()
 
    # Binarizamos para identificar las líneas (negro) sobre el fondo (blanco)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    inv = 255 - binary  # ahora las líneas valen 255
    num, labels = cv2.connectedComponents(inv, connectivity=8)
 
    ys, xs = np.where(inv > 0)
    if len(xs) == 0:
        return line_img  # no hay líneas
 
    edited = gray.copy()
    h, w = gray.shape[:2]
    for (px, py) in points_xy:
        px = int(np.clip(px, 0, w - 1))
        py = int(np.clip(py, 0, h - 1))
        # Buscar el píxel de línea más cercano al clic (por si no cae exacto encima)
        d2 = (xs - px) ** 2 + (ys - py) ** 2
        idx = int(np.argmin(d2))
        # Solo borrar si el clic está razonablemente cerca de una línea
        if d2[idx] <= (tolerance * max(h, w) / 100.0) ** 2 or d2[idx] <= tolerance ** 2:
            lbl = labels[ys[idx], xs[idx]]
            if lbl != 0:
                edited[labels == lbl] = 255
    return edited
    if len(line_img.shape) == 2:
        rgb = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = line_img
    if upscale > 1:
        rgb = cv2.resize(rgb, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return vtracer.convert_raw_image_to_svg(
        buf.getvalue(), img_format="png",
        colormode=colormode, mode=mode,
        filter_speckle=int(filter_speckle),
        corner_threshold=int(corner_threshold),
        path_precision=int(path_precision),
        length_threshold=float(length_threshold),
    )
 
 
# ============================================================
#   BARRA LATERAL IZQUIERDA = TODAS LAS HERRAMIENTAS
# ============================================================
 
with st.sidebar:
    st.header("🛠️ Herramientas")
 
    uploaded_file = st.file_uploader(
        "1) Sube tu imagen", type=["png", "jpg", "jpeg", "webp"]
    )
 
    # ---------- PASO 2: Limpiar el fondo ----------
    st.markdown("### 2) Limpiar el fondo")
    st.caption("Sirve para quitar lo que está detrás de tu figura y dejarlo blanco.")
 
    metodo_fondo = st.radio(
        "¿Cómo quieres limpiar el fondo?",
        ["No tocar el fondo", "Quitar fondo con IA (recomendado)", "Fondo oscuro → blanco"],
        help=(
            "• No tocar: deja la imagen tal cual.\n"
            "• Quitar fondo con IA: un modelo inteligente reconoce tu figura "
            "y borra todo lo de atrás automáticamente (lo mejor si el fondo es complicado).\n"
            "• Fondo oscuro → blanco: convierte simplemente los tonos negros en blanco "
            "(rápido, ideal si el fondo ya es negro plano)."
        )
    )
 
    if metodo_fondo == "Quitar fondo con IA (recomendado)":
        if not REMBG_OK:
            st.warning("Falta instalar 'rembg' (agrégalo a requirements.txt).")
        else:
            st.caption("💡 La primera vez tarda un poco porque descarga el modelo de IA.")
    elif metodo_fondo == "Fondo oscuro → blanco":
        umbral_negro = st.slider(
            "Qué tan oscuro cuenta como fondo", 0, 150, 30,
            help="Súbelo si quedan restos grises del fondo; bájalo si borra partes de tu figura."
        )
        suavizado_borde = st.slider(
            "Suavizar el borde", 0, 21, 0, step=1,
            help="Evita que el corte del fondo se vea duro alrededor de la figura."
        )
 
    st.divider()
 
    # ---------- PASO 3: Convertir a líneas ----------
    st.markdown("### 3) Convertir a líneas")
    st.caption("Elige el 'estilo' de dibujo con el que se dibujarán las líneas.")
 
    metodo = st.selectbox(
        "Estilo de líneas",
        [
            "Estilo tatuaje/logo (XDoG)",
            "Estilo libro para colorear (grueso)",
            "Contornos definidos (Canny)",
            "Dibujo a lápiz (con sombras)",
        ],
        help=(
            "• Tatuaje/logo: líneas limpias y artísticas (lo mejor para diseños como logos).\n"
            "• Libro para colorear: líneas gruesas y espacios blancos amplios.\n"
            "• Contornos: solo los bordes principales, bien marcados.\n"
            "• A lápiz: conserva sombreado, parece dibujo hecho a mano."
        )
    )
 
    with st.expander("Ajustes finos del estilo"):
        if metodo == "Dibujo a lápiz (con sombras)":
            blur_ksize = st.slider("Suavizado", 3, 51, 21, step=2,
                help="Más alto = dibujo más suave y con menos detalle.")
        elif metodo == "Contornos definidos (Canny)":
            low_t = st.slider("Sensibilidad de bordes (mín.)", 0, 255, 50,
                help="Más bajo = detecta más bordes (más líneas).")
            high_t = st.slider("Sensibilidad de bordes (máx.)", 0, 255, 150,
                help="Más alto = solo bordes muy marcados.")
            blur_ksize = st.slider("Suavizado previo", 1, 15, 5, step=2)
        elif metodo == "Estilo libro para colorear (grueso)":
            block_size = st.slider("Tamaño de detalle", 3, 51, 9, step=2,
                help="Más alto = líneas más grandes y menos detalle fino.")
            c_value = st.slider("Intensidad de líneas", -20, 20, 7)
            blur_ksize = st.slider("Suavizado", 1, 15, 5, step=2)
        else:  # XDoG
            sigma = st.slider("Grosor base de línea", 0.1, 3.0, 0.5,
                help="Controla qué tan gruesas salen las líneas.")
            k = st.slider("Detalle", 1.0, 3.0, 1.6)
            gamma = st.slider("Contraste", 0.8, 1.2, 0.98)
            epsilon = st.slider("Umbral de líneas", 0.0, 1.0, 0.1)
            phi = st.slider("Dureza del trazo", 1, 50, 10)
 
        grosor_extra = st.slider("Reforzar líneas", 0.0, 2.0, 0.5, step=0.1,
            help="Oscurece y marca más las líneas resultantes.")
 
    st.divider()
 
    # ---------- PASO 4: Modo nítido ----------
    st.markdown("### 4) Limpieza final")
    st.caption("Deja las líneas 100% negras sobre blanco y borra los puntitos sueltos.")
 
    modo_nitido = st.checkbox("Activar limpieza nítida", value=True,
        help="Recomendado. Convierte todo a blanco y negro puro, sin grises ni manchas.")
 
    if modo_nitido:
        with st.expander("Ajustes de limpieza"):
            bin_threshold = st.slider("Punto de corte blanco/negro", 0, 255, 180,
                help="Más alto = líneas más finas; más bajo = líneas más gruesas.")
            remove_specks = st.checkbox("Borrar manchas de ruido", value=True)
            min_speck_size = st.slider("Tamaño mínimo de mancha", 1, 200, 15,
                help="Súbelo para borrar más puntitos sueltos del fondo.")
            close_gaps = st.checkbox("Unir líneas cortadas", value=True)
            close_ksize = st.slider("Fuerza de unión", 1, 7, 2)
 
    st.divider()
 
    # ---------- PASO 5: Edición manual ----------
    st.markdown("### 5) Edición manual (borrar a mano)")
    st.caption(
        "Corrige el resultado borrando líneas o zonas que no quieres. "
        "Puedes acercar la imagen (zoom) para trabajar con precisión."
    )
 
    if not CANVAS_OK:
        st.warning("Falta instalar 'streamlit-drawable-canvas' (agrégalo a requirements.txt).")
        edicion_manual = False
    else:
        edicion_manual = st.checkbox("Activar edición manual", value=False)
 
    if CANVAS_OK and edicion_manual:
        modo_borrado = st.radio(
            "¿Cómo quieres borrar?",
            ["Brocha (pintar encima)", "Borrar trazo con un clic", "Borrar zona (rectángulo)"],
            help=(
                "• Brocha: pintas con el mouse encima de lo que quieras borrar "
                "(como un borrador). Ajusta el grosor abajo.\n"
                "• Borrar trazo con un clic: haces clic sobre una línea y se borra "
                "esa línea completa de una vez.\n"
                "• Borrar zona: dibujas un rectángulo y se borra todo lo que quede dentro."
            )
        )
        zoom = st.slider(
            "🔍 Zoom (acercar imagen)", 0.5, 3.0, 1.0, step=0.1,
            help="Agranda la imagen en pantalla para poder borrar con más precisión."
        )
        if modo_borrado == "Brocha (pintar encima)":
            grosor_borrador = st.slider("Grosor del borrador", 3, 60, 20)
        st.caption(
            "↩️ Usa los íconos debajo del lienzo para deshacer, rehacer o limpiar todo."
        )
 
    st.divider()
 
    # ---------- PASO 6: Vectorizar ----------
    st.markdown("### 6) Vectorizar (opcional)")
    st.caption(
        "Convierte el dibujo en un archivo que se puede agrandar infinitamente "
        "sin que se vea pixelado. Ideal para imprimir grande o cortar en vinilo."
    )
 
    if not VTRACER_OK:
        st.warning("Falta instalar 'vtracer' (agrégalo a requirements.txt).")
        vectorizar = False
    else:
        vectorizar = st.checkbox("Generar archivo vectorial (SVG)", value=False)
 
    if VTRACER_OK and vectorizar:
        with st.expander("Ajustes del vector"):
            v_colormode = st.selectbox("¿Blanco y negro o a color?", ["binary", "color"],
                format_func=lambda x: "Blanco y negro" if x == "binary" else "A color",
                help="Blanco y negro para líneas; a color si quieres conservar los colores.")
            v_mode = st.selectbox("Tipo de trazo", ["spline", "polygon", "none"],
                format_func=lambda x: {"spline": "Curvas suaves", "polygon": "Líneas rectas",
                                       "none": "Pixelado"}[x])
            v_upscale = st.slider("Mejorar detalle antes de vectorizar", 1, 4, 2,
                help="Amplía la imagen antes de trazar para capturar detalles finos.")
            v_filter_speckle = st.slider("Quitar manchas pequeñas", 0, 30, 4)
            v_corner_threshold = st.slider("Suavidad de esquinas", 0, 180, 60)
            v_length_threshold = st.slider("Ignorar trazos cortos", 0.0, 10.0, 4.0, step=0.5)
 
 
# ============================================================
#   ÁREA PRINCIPAL DERECHA = IMÁGENES / RESULTADO
# ============================================================
 
if uploaded_file is None:
    st.info("👈 Sube una imagen en la barra lateral para comenzar.")
    st.markdown(
        "**Guía rápida:**\n"
        "1. Sube tu imagen.\n"
        "2. Limpia el fondo (la IA es la opción más cómoda).\n"
        "3. Elige el estilo de líneas.\n"
        "4. Deja la limpieza nítida activada.\n"
        "5. Si quieres, borra a mano líneas o zonas que no te gusten (con zoom y clic).\n"
        "6. Si quieres un archivo escalable, activa la vectorización.\n\n"
        "Los resultados aparecerán aquí a la derecha y podrás descargarlos."
    )
else:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
 
    # Paso 2: fondo
    with st.spinner("Preparando la imagen..."):
        if metodo_fondo == "Quitar fondo con IA (recomendado)" and REMBG_OK:
            img_bgr = ai_remove_background(img_bgr)
        elif metodo_fondo == "Fondo oscuro → blanco":
            img_bgr = black_bg_to_white(img_bgr, threshold=umbral_negro, feather=suavizado_borde)
 
    # Paso 3: líneas
    with st.spinner("Dibujando las líneas..."):
        if metodo == "Dibujo a lápiz (con sombras)":
            result = pencil_sketch(img_bgr, blur_ksize=blur_ksize)
        elif metodo == "Contornos definidos (Canny)":
            result = canny_lines(img_bgr, low_t, high_t, blur_ksize)
        elif metodo == "Estilo libro para colorear (grueso)":
            result = adaptive_threshold_lines(img_bgr, block_size, c_value, blur_ksize)
        else:
            result = xdog_lines(img_bgr, sigma, k, gamma, epsilon, phi)
 
        if grosor_extra > 0:
            result = sharpen_lines(result, grosor_extra)
 
        # Paso 4: nítido
        if modo_nitido:
            result = clean_sharp_lines(
                result, bin_threshold=bin_threshold, remove_specks=remove_specks,
                min_speck_size=min_speck_size, close_gaps=close_gaps, close_ksize=close_ksize
            )
 
    # Mostrar original vs resultado
    col1, col2 = st.columns(2)
    with col1:
        preview_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        cap = ("Imagen preparada" if metodo_fondo != "No tocar el fondo" else "Imagen original")
        st.image(preview_img, caption=cap, width="stretch")
    with col2:
        st.image(result, caption="Dibujo de líneas", width="stretch", clamp=True)
 
    # ---------- Paso 5: Edición manual con lienzo interactivo ----------
    if CANVAS_OK and edicion_manual:
        st.divider()
        st.subheader("✏️ Edición manual")
        st.caption(
            "Trabaja sobre el lienzo de abajo. Lo que borres aquí se usará también "
            "para la descarga y el vector."
        )
 
        # Preparar imagen base en RGB para el lienzo
        if len(result.shape) == 2:
            base_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        else:
            base_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        h0, w0 = base_rgb.shape[:2]
 
        # Tamaño de visualización según zoom (con un límite para no saturar)
        base_fit = min(w0, 700)
        disp_w = int(min(base_fit * zoom, 1400))
        disp_h = int(disp_w * h0 / w0)
        base_disp = cv2.resize(base_rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        base_pil = Image.fromarray(base_disp)
 
        # Configurar el modo del lienzo
        if modo_borrado == "Brocha (pintar encima)":
            drawing_mode = "freedraw"
            stroke_w = grosor_borrador
        elif modo_borrado == "Borrar zona (rectángulo)":
            drawing_mode = "rect"
            stroke_w = 2
        else:  # Borrar trazo con un clic
            drawing_mode = "point"
            stroke_w = 3
 
        canvas_result = st_canvas(
            fill_color="rgba(255,255,255,1)",   # blanco = borrar
            stroke_color="rgba(255,255,255,1)", # blanco
            stroke_width=stroke_w,
            background_image=base_pil,
            update_streamlit=True,
            height=disp_h,
            width=disp_w,
            drawing_mode=drawing_mode,
            point_display_radius=4 if drawing_mode == "point" else 0,
            display_toolbar=True,
            key="lienzo_edicion",
        )
 
        # Aplicar el borrado sobre el resultado a resolución completa
        edited = result.copy()
        scale_x = w0 / disp_w
        scale_y = h0 / disp_h
 
        if drawing_mode == "point":
            # Borrar trazos completos en cada punto donde se hizo clic
            puntos = []
            if canvas_result.json_data is not None:
                for obj in canvas_result.json_data.get("objects", []):
                    left = obj.get("left", 0)
                    top = obj.get("top", 0)
                    radius = obj.get("radius", 0)
                    cx = (left + radius) * scale_x
                    cy = (top + radius) * scale_y
                    puntos.append((cx, cy))
            if puntos:
                edited = erase_strokes_at_points(edited, puntos)
        else:
            # Brocha o rectángulo: usar lo pintado como máscara de borrado
            if canvas_result.image_data is not None:
                overlay = canvas_result.image_data  # RGBA en tamaño de display
                alpha = overlay[:, :, 3]
                mask = (alpha > 0).astype(np.uint8) * 255
                mask_full = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
                if len(edited.shape) == 2:
                    edited[mask_full > 0] = 255
                else:
                    edited[mask_full > 0] = (255, 255, 255)
 
        result = edited  # todo lo de abajo usa la versión editada
 
        st.image(result, caption="Resultado editado", width="stretch", clamp=True)
 
    st.divider()
    # Descargar PNG
    result_pil = Image.fromarray(result)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    st.download_button("⬇️ Descargar dibujo (PNG)", data=buf.getvalue(),
                       file_name="imagen_lineas.png", mime="image/png")
 
    # Paso 5: SVG
    if VTRACER_OK and vectorizar:
        st.divider()
        st.subheader("🔷 Versión vectorial (SVG)")
        with st.spinner("Vectorizando... (puede tardar unos segundos)"):
            try:
                if v_colormode == "color":
                    source_for_vector = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                else:
                    source_for_vector = result
                svg_str = vectorize_to_svg(
                    source_for_vector, colormode=v_colormode, mode=v_mode,
                    filter_speckle=v_filter_speckle, corner_threshold=v_corner_threshold,
                    length_threshold=v_length_threshold, upscale=v_upscale,
                )
                st.components.v1.html(
                    f'<div style="background:white;padding:10px;border-radius:8px;">{svg_str}</div>',
                    height=520, scrolling=True,
                )
                st.download_button("⬇️ Descargar vector (SVG)",
                                   data=svg_str.encode("utf-8"),
                                   file_name="imagen_vectorizada.svg",
                                   mime="image/svg+xml")
            except Exception as e:
                st.error(f"No se pudo vectorizar: {e}")
