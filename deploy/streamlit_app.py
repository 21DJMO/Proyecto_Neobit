import streamlit as st
import os
import sys
import numpy as np
import time

# Añadir el directorio src al path para poder importar los módulos originales
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from audio_capture import record_continuous
    from vad_silero import load_vad_model, detect_speech
    from transcribe_whisper import load_transcription_model, transcribe_segments
    from chat_ollama import get_chat_response
    from tts_azure import init_speech_synthesizer, speak_text
except ImportError as e:
    st.error(f"Error al importar los módulos de src: {e}")
    st.stop()

# Configuración de la página
st.set_page_config(page_title="Neobit - Asistente de Idiomas", page_icon="🚀", layout="centered")

# Estilos personalizados para que se vea premium
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #262730;
        color: white;
        border: 1px solid #4a4a4a;
    }
    .stButton>button:hover {
        border-color: #ff4b4b;
        color: #ff4b4b;
    }
    .chat-bubble {
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 10px;
    }
    .user-bubble {
        background-color: #1e3a8a;
        color: white;
        text-align: right;
    }
    .ai-bubble {
        background-color: #2d2d2d;
        color: white;
        border-left: 5px solid #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

# Inicializar estado de la sesión
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'difficulty' not in st.session_state:
    st.session_state.difficulty = "intermedio"
if 'recorder' not in st.session_state:
    from audio_capture import StreamlitRecorder
    st.session_state.recorder = StreamlitRecorder()
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False

# Título y Header
st.title("🚀 Neobit AI Assistant")
st.subheader("Tu tutor personal de idiomas con IA local")

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.session_state.difficulty = st.selectbox(
        "Nivel de inglés:",
        options=["básico", "intermedio", "avanzado"],
        index=1,
        format_func=lambda x: x.capitalize()
    )
    
    st.divider()
    st.info("Este asistente usa Whisper y Ollama (Llama 3.2) de forma local para proteger tu privacidad.")

# Carga de modelos
if not st.session_state.models_loaded:
    with st.status("Cargando motores de IA...", expanded=True) as status:
        st.write("Cargando VAD (Silero)...")
        st.session_state.vad_model, st.session_state.vad_utils = load_vad_model()
        
        st.write("Cargando Whisper (Base)...")
        st.session_state.whisper_model = load_transcription_model("base")
        
        st.write("Inicializando Azure Voice...")
        st.session_state.tts_synthesizer = init_speech_synthesizer()
        
        st.session_state.models_loaded = True
        status.update(label="¡Sistemas listos!", state="complete", expanded=False)

# Contenedor de chat
chat_container = st.container()

def display_chat():
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-bubble user-bubble">🗣️ {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble ai-bubble">{message["content"]}</div>', unsafe_allow_html=True)
                if "note" in message and message["note"]:
                    with st.expander("Ver correcciones y tips", expanded=False):
                        st.markdown(message["note"])

def procesar_entrada(texto):
    if not texto:
        return
    
    st.session_state.chat_history.append({"role": "user", "content": texto})
    
    with st.spinner("IA pensando..."):
        respuesta, _ = get_chat_response(
            user_text=texto,
            model="llama3.2",
            chat_history=[m for m in st.session_state.chat_history if m["role"] != "system"],
            difficulty=st.session_state.difficulty
        )
        texto_respuesta = respuesta["spoken"]
        nota_correccion = respuesta["note"]
        st.session_state.chat_history.append({"role": "assistant", "content": texto_respuesta, "note": nota_correccion})
    
    if st.session_state.tts_synthesizer:
        speak_text(texto_respuesta, st.session_state.tts_synthesizer)
    
    st.rerun()

display_chat()

# Área de Entrada Conjunta
st.divider()
input_col, mic_col, send_col = st.columns([0.6, 0.25, 0.15])

with input_col:
    with st.form(key="chat_form", clear_on_submit=True):
        prompt = st.text_input("Escribe algo en inglés...", placeholder="Hello, how are you?", label_visibility="collapsed")
        submit_text = st.form_submit_button("🚀 Enviar")

with mic_col:
    if not st.session_state.is_recording:
        if st.button("🎤 Iniciar Grabación", use_container_width=True):
            st.session_state.is_recording = True
            st.session_state.recorder.start()
            st.rerun()
    else:
        if st.button("🛑 Detener y Procesar", use_container_width=True, type="primary"):
            st.session_state.is_recording = False
            audio = st.session_state.recorder.stop()
            
            if audio is not None:
                with st.spinner("Procesando audio largo..."):
                    timestamps = detect_speech(audio, st.session_state.vad_model, st.session_state.vad_utils, fs=16000)
                    if timestamps:
                        resultados = transcribe_segments(audio, timestamps, st.session_state.whisper_model, fs=16000)
                        texto_usuario = " ".join([r["text"] for r in resultados]).strip()
                        if texto_usuario:
                            procesar_entrada(texto_usuario)
                    else:
                        st.warning("No se detectó voz clara.")
            st.rerun()

# Lógica del formulario de texto
if submit_text and prompt:
    procesar_entrada(prompt)
