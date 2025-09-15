import streamlit as st
from PIL import Image
import torch, io, requests, os
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ---------- CONFIGURACIÓN DE PÁGINA ----------
st.set_page_config(
    page_title="Clasificador de Frutas y Vegetales",
    page_icon="🍏",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------- ESTILO PERSONALIZADO ----------
st.markdown("""
    <style>
    /* Fondo azul noche degradado */
    .stApp {
        background: linear-gradient(135deg, #0b1d3a 0%, #1a2a5b 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Animación flotante para el logo */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
        margin-bottom: 20px;
        animation: float 4s ease-in-out infinite;
    }

    .title {
        font-family: 'Segoe UI Semibold', sans-serif;
        color: #f1f1f1;
        font-size: 42px;
        text-align: center;
        padding-top: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #a3c1f0;  
        padding-bottom: 30px;
    }

    /* Botones */
    .stButton>button {
        background-color: #1e3d70;
        color: #a3c1f0;
        height: 40px;
        width: 220px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #16315a;
        color: #ffffff;
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #1a2a5b 0%, #0b1d3a 100%);
        color: #f1f1f1;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sidebar .sidebar-content h3, .sidebar .sidebar-content h4 {
        color: #f1f1f1;
    }
    .sidebar img {
        border-radius: 8px;
        margin-bottom: 15px;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 14px;
        color: #cfd9ee;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- BARRA LATERAL ----------
escudo_path = "escudo.jpg"
st.sidebar.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
if os.path.exists(escudo_path):
    st.sidebar.image(escudo_path, width=150)

st.sidebar.markdown("""
    <h3>Universidad Nacional Agraria La Molina</h3>
    <h4>Curso: Ciencias de Datos 2</h4>
    <h4>Profesor: Aldo Richard Meza Rodriguez</h4>
    <hr style='border:1px solid #a3c1f0'>
    <h4>Integrantes:</h4>
    <ul>
        <li>Juárez Castro, Andre Saul - 20200396</li>
        <li>Medrano Alania, Sebastian Rodrigo - 20211821</li>
        <li>Quispe Cuadros, Arthur Jesus Abrahan - 20211827</li>
        <li>Vargas Céspedes, Jose Emmanuel - 20191317</li>
        <li>Yacila Aramburu, Pedro David - 20200342</li>
    </ul>
""", unsafe_allow_html=True)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# ---------- LOGO PRINCIPAL ----------
logo_path = "logo2.png"
if os.path.exists(logo_path):
    st.markdown(f"<div class='logo-container'><img src='{logo_path}' width='220'></div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='title'>🍏 Clasificador de Frutas y Vegetales</div>", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Sube una imagen o pega la URL de una imagen para clasificarla</div>", unsafe_allow_html=True)

# ---------- CARGAR MODELO ----------
@st.cache_resource
def load_model():
    modelo_path = r"C:\Users\PC\Desktop\pc1_2025\CD2\fruits-veggies"
    processor = AutoImageProcessor.from_pretrained(modelo_path)
    model = AutoModelForImageClassification.from_pretrained(modelo_path)
    model.eval()
    return processor, model

processor, model = load_model()

# ---------- FUNCIÓN DE PREDICCIÓN ----------
def predict_image(img: Image.Image):
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1)
    clase_id = probs.argmax(dim=1).item()
    clase_nombre = model.config.id2label[clase_id]
    confianza = probs[0, clase_id].item() * 100
    return clase_nombre, confianza

# ---------- INTERFAZ ----------
tab1, tab2 = st.tabs(["📁 Subir Imagen", "🌐 URL de Imagen"])

with tab1:
    uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg","jpeg","png","bmp"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Imagen subida", use_column_width=True)
        if st.button("Clasificar imagen subida"):
            clase, conf = predict_image(img)
            st.success(f"**Predicción:** {clase}  \n**Confianza:** {conf:.2f}%")

with tab2:
    url = st.text_input("Pega la URL de la imagen")
    if st.button("Clasificar imagen desde URL"):
        if not url.strip():
            st.warning("Por favor ingresa una URL.")
        else:
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content))
                st.image(img, caption="Imagen desde URL", use_column_width=True)
                clase, conf = predict_image(img)
                st.success(f"**Predicción:** {clase}  \n**Confianza:** {conf:.2f}%")
            except Exception as e:
                st.error(f"No se pudo cargar la imagen desde la URL:\n{e}")

st.markdown("<div class='footer'>© 2025 – Clasificador IA con Streamlit</div>", unsafe_allow_html=True)
