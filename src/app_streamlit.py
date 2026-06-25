import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import time
import os
import textwrap

# Asegurar que el directorio src está en el path para importaciones limpias
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar pipelines existentes
from audio_capture import StreamlitRecorder
from vad_silero import load_vad_model, detect_speech
from transcribe_whisper import load_transcription_model, transcribe_segments
from chat_ollama import get_chat_response
from tts_azure import init_speech_synthesizer, speak_text
from mission_manager import get_available_missions, get_mission_by_id

# ---------- Evaluación y feedback específico para la misión `order_food` ----------
import re


def evaluate_order_food_objectives(messages):
    """Detecta los micro-objetivos completados por el usuario en la conversación.
    Examina TODOS los mensajes, tanto del usuario como del asistente.
    Retorna un dict con booleanos para cada objetivo.
    """
    user_texts = [m.get("content", "").lower() for m in messages if m.get("role") == "user"]
    assistant_texts = [m.get("spoken", "").lower() for m in messages if m.get("role") == "assistant"]
    joined_user = " ".join(user_texts)
    joined_assistant = " ".join(assistant_texts)
    joined_all = joined_user + " " + joined_assistant

    # 1. Food selected: burger and fries are mentioned in the conversation
    food_selected = ("burger" in joined_all or "hamburger" in joined_all) and ("fries" in joined_all)

    # 2. Drink selected: a drink is mentioned in the conversation
    drink_triggers = ["drink", "tea", "coffee", "water", "cola", "coke", "soda", "juice", "lemonade", "beverage"]
    drink_selected = any(d in joined_all for d in drink_triggers)

    # 3. Order confirmed: waiter confirms order details
    confirm_triggers = [
        "your order is", "so that's", "so that is", "confirm", "that will be",
        "you ordered", "is that correct", "correct?", "is this correct", "so you want",
        # afirmaciones del waiter sin signo de pregunta
        "that's correct", "that is correct", "that's right", "that is right",
        "perfect, your order", "so your order", "your new order",
    ]
    order_confirmed = any(c in joined_user for c in confirm_triggers) and ("burger" in joined_user or "hamburger" in joined_user or "fries" in joined_user or "drink" in joined_user or "tea" in joined_user or "coke" in joined_user or "soda" in joined_user or "water" in joined_user or "juice" in joined_user)

    return {
        "food_selected": food_selected,
        "drink_selected": drink_selected,
        "order_confirmed": order_confirmed
    }


def check_client_confirmation(spoken_text):
    text = spoken_text.lower()
    confirmation_phrases = ["correct", "right", "perfect", "thank you", "thanks", "yes, that"]
    return any(p in text for p in confirmation_phrases)


def check_mission_end(spoken_text):
    """
    Detecta si Neobit dio una respuesta de cierre real en este turno exacto.
    Solo evalúa el turno actual — no el historial previo — para evitar
    falsos positivos por frases acumuladas en turnos anteriores.
    """
    text = spoken_text.lower()
    end_phrases = [
        "goodbye", "bye", "see you",
        "enjoy your meal", "enjoy your food",
        "i'll wait", "great, i'll wait", "great, thank you",
        "thank you for confirming", "i will wait",
    ]
    return any(p in text for p in end_phrases)


def has_natural_closure(messages, lookback=3):
    """Detecta si la conversación tiene un cierre natural.
    Revisa los últimos mensajes del usuario Y del asistente.
    No depende de frases exactas - usa matching flexible.
    """
    recent = messages[-lookback:] if len(messages) >= lookback else messages
    user_recent = " ".join([
        m.get("content", "").lower() for m in recent if m.get("role") == "user"
    ])
    assistant_recent = " ".join([
        m.get("spoken", "").lower() for m in recent if m.get("role") == "assistant"
    ])
    combined = user_recent + " " + assistant_recent

    # Frases de cierre naturales - flexibles
    closure_phrases = [
        "thank you", "thanks", "thank you so much",
        "have a nice day", "have a good day", "have a good one",
        "goodbye", "bye", "bye bye",
        "see you later", "see you", "see you soon",
        "you're welcome", "you are welcome", "youre welcome",
        "enjoy your meal", "enjoy", "enjoy your food",
        "take care",
        "you too",
        "come again", "come back soon",
        "have fun", "talk to you later",
        "that will be all", "that's all", "i'm done", "i am done",
        "perfect", "perfect thank you", "thanks for your help"
    ]
    return any(phrase in combined for phrase in closure_phrases)


def analyze_conversation(messages, vocab_goals=None):
    """Analiza la conversación completa y prepara un feedback final.
    Revisa CADA intervención del usuario individualmente.
    Reporta correcciones claras con alternativas más naturales y explicaciones breves.
    """
    user_texts = [m.get("content", "") for m in messages if m.get("role") == "user"]
    joined = " \n".join(user_texts).strip()

    corrections = []
    better_alternatives = []
    vocab_used = []
    strengths = []
    suggestions = []

    lower_joined = joined.lower()

    # Detectar vocabulario usado
    if vocab_goals:
        for v in vocab_goals:
            if re.search(rf"\b{re.escape(v)}\b", lower_joined, re.IGNORECASE):
                vocab_used.append(v)

    # ── Reglas de corrección gramatical ──
    pattern_rules = [
        {
            "regex": r"\bwould you like to some\b",
            "corrected": "Would you like some",
            "explanation": "After 'Would you like' we do not use 'to'."
        },
        {
            "regex": r"\bthat's sounds\b",
            "corrected": "That sounds",
            "explanation": "Use 'That sounds' instead of 'That's sounds'."
        },
        {
            "regex": r"\bhamburguer\b",
            "corrected": "Hamburger",
            "explanation": "The correct spelling is 'hamburger'."
        },
        {
            "regex": r"\bin \d+ minutes you received\b",
            "corrected": "In X minutes you will receive",
            "explanation": "Use future tense for an action that will happen later."
        },
        {
            "regex": r"\bhello, you see her\b",
            "corrected": "Hello, did you see her?",
            "explanation": "Use 'did you see her' for a clearer question structure."
        },
        {
            "regex": r"\bwould you like to a drink\b",
            "corrected": "Would you like a drink?",
            "explanation": "Use 'Would you like a drink' instead of 'Would you like to a drink'."
        },
        {
            "regex": r"\byou is\b",
            "corrected": "You are",
            "explanation": "Use 'you are' instead of 'you is'."
        },
        {
            "regex": r"\bhe don't\b",
            "corrected": "He doesn't",
            "explanation": "Use 'doesn't' with third person singular (he/she/it)."
        },
        {
            "regex": r"\bshe don't\b",
            "corrected": "She doesn't",
            "explanation": "Use 'doesn't' with third person singular (he/she/it)."
        },
        {
            "regex": r"\bit's sound\b",
            "corrected": "It sounds",
            "explanation": "Use 'It sounds' instead of 'It's sound'."
        },
    ]

    common_errors = [
        {
            "regex": r"\bto some fries\b",
            "corrected": "some fries",
            "explanation": "No 'to' is needed before 'some fries'."
        },
        {
            "regex": r"\bwould you like to a\b",
            "corrected": "would you like a",
            "explanation": "Use 'Would you like a ...' instead of 'Would you like to a ...'."
        },
        {
            "regex": r"\byou see her\b",
            "corrected": "did you see her",
            "explanation": "Use a question structure with 'did' for clarity."
        },
        {
            "regex": r"\bI want\b",
            "corrected": "I would like",
            "explanation": "'I would like' is more polite in a restaurant."
        },
        {
            "regex": r"\bgive me\b",
            "corrected": "Could I have",
            "explanation": "'Could I have' is more polite than 'Give me'."
        },
    ]

    misspellings = {
        "hamburguer": "hamburger",
        "friees": "fries",
        "recieve": "receive",
        "restaurnt": "restaurant",
        "waite": "waiter",
        "drinck": "drink",
        "pleez": "please",
        "thak you": "thank you",
    }

    # ── Mejores alternativas para sonar más natural ──
    alternative_rules = [
        {
            "regex": r"\bwhat do you drink\b",
            "alternative": "What would you like to drink?",
            "reason": "This sounds more natural and polite in a restaurant."
        },
        {
            "regex": r"\bwhat you want\b",
            "alternative": "What would you like?",
            "reason": "More polite and natural for a waiter."
        },
        {
            "regex": r"\byou want what\b",
            "alternative": "What would you like?",
            "reason": "More natural word order for a question."
        },
        {
            "regex": r"\bhow long you want\b",
            "alternative": "How long would you like to wait?",
            "reason": "Sounds more natural and complete."
        },
        {
            "regex": r"\bI want burger\b",
            "alternative": "I would like a burger.",
            "reason": "Using 'I would like' is more polite."
        },
        {
            "regex": r"\bgive me burger\b",
            "alternative": "Could I have a burger, please?",
            "reason": "Much more polite and natural."
        },
    ]

    # Revisar cada mensaje individualmente
    for msg in user_texts:
        original = msg.strip()
        lower_msg = original.lower()
        if not original:
            continue

        # Correcciones gramaticales
        for rule in pattern_rules:
            if re.search(rule["regex"], lower_msg, re.IGNORECASE):
                # Generar corrección específica para este caso
                corrected = re.sub(rule["regex"], rule["corrected"], original, flags=re.IGNORECASE)
                if corrected != original:
                    corrections.append({
                        "original": original,
                        "corrected": corrected,
                        "explanation": rule["explanation"]
                    })

        for err in common_errors:
            if re.search(err["regex"], lower_msg, re.IGNORECASE):
                corrected = re.sub(err["regex"], err["corrected"], original, flags=re.IGNORECASE)
                if corrected != original:
                    corrections.append({
                        "original": original,
                        "corrected": corrected,
                        "explanation": err["explanation"]
                    })

        # Ortografía
        for bad, good in misspellings.items():
            if re.search(rf"\b{re.escape(bad)}\b", lower_msg):
                corrected = re.sub(rf"\b{re.escape(bad)}\b", good, original, flags=re.IGNORECASE)
                if corrected != original:
                    corrections.append({
                        "original": original,
                        "corrected": corrected,
                        "explanation": f"Spelling: '{good}' is the correct form."
                    })

        # Mejores alternativas
        for alt in alternative_rules:
            if re.search(alt["regex"], lower_msg, re.IGNORECASE):
                better_alternatives.append({
                    "original": original,
                    "alternative": alt["alternative"],
                    "reason": alt["reason"]
                })

    # Eliminar duplicados de correcciones
    unique_corrections = {}
    for c in corrections:
        key = (c["original"], c["corrected"])
        unique_corrections[key] = c
    corrections = list(unique_corrections.values())

    # Eliminar duplicados de alternativas
    unique_alternatives = {}
    for a in better_alternatives:
        key = (a["original"], a["alternative"])
        unique_alternatives[key] = a
    better_alternatives = list(unique_alternatives.values())

    # Fortalezas
    if vocab_used:
        strengths.append("Used mission vocabulary naturally.")
    if len(user_texts) >= 3:
        strengths.append("Maintained a multi-turn interaction.")
    if len(user_texts) >= 5:
        strengths.append("Extended conversation with multiple exchanges.")
    if not corrections:
        strengths.append("Excellent grammar throughout the conversation.")
    if better_alternatives:
        strengths.append("Good effort in communicating your needs.")
    if not strengths:
        strengths.append("The interaction was concise and focused.")

    # Cálculo de puntuación
    real_corrections = [c for c in corrections if c.get("original") and c.get("corrected")]
    missing_vocab = max(0, 5 - len(vocab_used))
    score = 100 - len(real_corrections) * 10 - len(better_alternatives) * 5 - missing_vocab * 3
    score = max(55, min(100, score))

    # Sugerencias
    suggestions = []
    if real_corrections:
        suggestions.append("Review the grammar corrections above and practice the correct forms.")
    if better_alternatives:
        suggestions.append("Try using the suggested alternatives to sound more natural.")
    if missing_vocab > 0 and vocab_goals:
        target_words = [v for v in vocab_goals if v not in vocab_used]
        suggestions.append(f"Try using these target words: {', '.join(target_words)}.")
    if not real_corrections and not better_alternatives:
        suggestions.append("Great work! Keep practicing to build confidence and fluency.")
    if not suggestions:
        suggestions.append("Continue practicing to improve your fluency.")

    return {
        "score": score,
        "corrections": corrections,
        "better_alternatives": better_alternatives,
        "vocab_used": vocab_used,
        "strengths": strengths,
        "suggestions": suggestions
    }

# ─────────────────────────────────────────────────────────────────────────────
# FAREWELL PHRASES — detección de despedida del usuario para cierre de misión
# ─────────────────────────────────────────────────────────────────────────────
FAREWELL_PHRASES = [
    # Despedidas directas
    "goodbye", "bye", "bye bye", "see you", "see you later", "see you soon",
    "take care", "have a nice day", "have a good day", "have a good one",
    "have a great day", "have a nice evening", "have a wonderful day",
    "have a wonderful evening",
    # Cierre del servicio — waiter
    "thank you for your order", "thanks for your order",
    "thank you for your purchase", "thank you for visiting",
    "thank you for coming", "thanks for coming",
    "enjoy your meal", "enjoy your food", "enjoy your dinner", "enjoy your lunch",
    "your food will be ready", "we will bring it", "we'll bring it",
    "your order is on its way", "come back soon", "come again",
    "see you next time", "that's all for now", "that is all for now",
    # Cierre natural de conversación
    "have a good rest", "have a good evening", "all done", "we're all set",
    "we are all set", "you're all set", "you are all set",
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN DE LA PÁGINA Y ESTILOS PREMIUM (CSS CUSTOM)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neobit — English Roleplay Coach",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de estilos CSS premium
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Configuración de Tipografía Global */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Título principal con degradado premium */
    .premium-title {
        background: linear-gradient(135deg, #1e3a8a 0%, #4f46e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 5px;
        text-align: left;
    }
    
    .premium-subtitle {
        color: #475569;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    
    /* Tarjeta de Briefing */
    .briefing-card {
        background-color: #ffffff;
        border: 2px solid #cbd5e1;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    
    .briefing-title {
        color: #4f46e5;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .briefing-meta {
        font-size: 1rem;
        margin-bottom: 8px;
        color: #1e293b;
        line-height: 1.5;
    }
    
    .briefing-label {
        font-weight: 700;
        color: #0f172a;
    }
    
    /* Badges para vocabulario */
    .vocab-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e4ed8;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    /* Modal de Misión Completada */
    .mission-complete-modal {
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
        border: 3px solid #10b981;
        border-radius: 24px;
        padding: 32px;
        margin-bottom: 24px;
        box-shadow: 0 20px 40px rgba(16, 185, 129, 0.15);
        text-align: center;
    }
    .mission-complete-modal h1 {
        color: #065f46;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .mission-complete-modal .subtitle {
        color: #047857;
        font-size: 1.1rem;
        margin-bottom: 16px;
    }
    .mission-complete-modal .score-badge {
        display: inline-block;
        background: #10b981;
        color: white;
        font-size: 2.5rem;
        font-weight: 800;
        border-radius: 50%;
        width: 100px;
        height: 100px;
        line-height: 100px;
        margin: 12px auto;
        box-shadow: 0 8px 16px rgba(16, 185, 129, 0.3);
    }
    .mission-complete-modal .view-results-btn {
        background: #10b981;
        color: white;
        border: none;
        padding: 12px 32px;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 700;
        cursor: pointer;
        margin-top: 16px;
        transition: all 0.2s;
    }
    .mission-complete-modal .view-results-btn:hover {
        background: #059669;
        transform: translateY(-2px);
    }
    
    /* Tarjeta de Reporte Profesional */
    .report-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 24px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 18px 30px rgba(15, 23, 42, 0.05);
    }
    .report-card h3 {
        color: #1e293b;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .report-card .section-divider {
        height: 1px;
        background: #e2e8f0;
        margin: 20px 0;
    }
    .objectives-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
    }
    .objective-item {
        padding: 6px 0;
        font-size: 0.95rem;
    }
    .objective-item .check { color: #10b981; font-weight: 700; }
    .objective-item .cross { color: #ef4444; font-weight: 700; }
    
    .correction-item {
        padding: 12px;
        background: #fefce8;
        border-left: 4px solid #eab308;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .correction-item .label { font-weight: 600; color: #1e293b; }
    .correction-item .original-text { color: #b91c1c; text-decoration: line-through; }
    .correction-item .corrected-text { color: #15803d; font-weight: 600; }
    .correction-item .explanation { color: #64748b; font-size: 0.9rem; margin-top: 4px; }
    
    .alternative-item {
        padding: 12px;
        background: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .alternative-item .label { font-weight: 600; color: #1e293b; }
    .alternative-item .orig-text { color: #64748b; }
    .alternative-item .alt-text { color: #0369a1; font-weight: 600; }
    .alternative-item .reason { color: #64748b; font-size: 0.9rem; margin-top: 4px; }
    
    .strength-item {
        padding: 6px 0;
        color: #065f46;
    }
    .strength-item::before { content: "✨ "; }
    
    .suggestion-item {
        padding: 6px 0;
        color: #1e293b;
    }
    .suggestion-item::before { content: "💡 "; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. CARGA EFICIENTE Y CACHÉ DE MODELOS (@st.cache_resource)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo VAD (Silero)...")
def get_vad_resources():
    return load_vad_model()

@st.cache_resource(show_spinner="Cargando modelo Whisper (Faster-Whisper)...")
def get_whisper_resources():
    # Usamos modelo 'base' para balance de velocidad/precisión en CPU
    return load_transcription_model("base")

@st.cache_resource(show_spinner="Inicializando Azure Text-to-Speech...")
def get_tts_resources():
    return init_speech_synthesizer()

# Cargar recursos (se ejecutan una sola vez en el ciclo de vida del servidor Streamlit)
vad_model, vad_utils = get_vad_resources()
whisper_model = get_whisper_resources()
tts_synthesizer = get_tts_resources()

# ─────────────────────────────────────────────────────────────────────────────
# 3. CONTROL DEL ESTADO DE LA SESIÓN (SESSION STATE)
# ─────────────────────────────────────────────────────────────────────────────
# Base de datos de misiones organizadas
missions_db = get_available_missions()

# Inicializar grabadora asíncrona
if "recorder" not in st.session_state:
    st.session_state.recorder = StreamlitRecorder(fs=16000)

if "is_recording" not in st.session_state:
    st.session_state.is_recording = False

# Historial visual de mensajes renderizados en el chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Historial plano de chat para enviar al modelo de Ollama
if "ollama_history" not in st.session_state:
    st.session_state.ollama_history = []

if "active_mission_id" not in st.session_state:
    st.session_state.active_mission_id = None

if "used_vocabulary" not in st.session_state:
    st.session_state.used_vocabulary = set()

if "mission_completed" not in st.session_state:
    st.session_state.mission_completed = False

if "corrections_summary" not in st.session_state:
    st.session_state.corrections_summary = []
    
if "mission_objectives" not in st.session_state:
    st.session_state.mission_objectives = {}

if "mission_feedback" not in st.session_state:
    st.session_state.mission_feedback = None

# Control para mostrar/ocultar el reporte detallado
if "show_report" not in st.session_state:
    st.session_state.show_report = False

# ─────────────────────────────────────────────────────────────────────────────
# 4. BARRA LATERAL (SIDEBAR DE CONTROL)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://raw.githubusercontent.com/google/material-design-icons/master/png/action/extension/black/48dp/1x/baseline_extension_black_48dp.png", width=60)
    st.title("NEOBIT COACH")
    st.write("Plataforma de aprendizaje inmersivo basada en gamificación emocional.")
    
    st.markdown("---")
    st.subheader("⚙️ Configuración")
    
    # Restricción obligatoria: Sólo Principiante
    st.info("🎯 **Nivel Seleccionado:** Básico / Principiante (A1/A2)\n\n*Nota: Niveles intermedio y avanzado están desactivados temporalmente.*")
    
    st.markdown("---")
    st.subheader("🗺️ Misiones del MVP (Básicas)")
    
    # Construcción dinámica del selector agrupado por categorías
    category_names = {cat: data["name"] for cat, data in missions_db.items()}
    selected_cat_key = st.selectbox(
        "Selecciona la Categoría:",
        options=list(missions_db.keys()),
        format_func=lambda x: category_names[x]
    )
    
    cat_missions = missions_db[selected_cat_key]["missions"]
    mission_titles = {m["id"]: m["title"] for m in cat_missions}
    selected_mission_id = st.selectbox(
        "Selecciona la Misión:",
        options=list(mission_titles.keys()),
        format_func=lambda x: mission_titles[x]
    )
    
    # Obtener metadatos de la misión activa
    active_mission = get_mission_by_id(selected_mission_id)
    
    # Detectar si el usuario cambió de misión para limpiar historial
    if st.session_state.active_mission_id != selected_mission_id:
        st.session_state.active_mission_id = selected_mission_id
        st.session_state.used_vocabulary = set()
        st.session_state.mission_completed = False
        st.session_state.corrections_summary = []
        st.session_state.show_report = False
        if "balloons_triggered" in st.session_state:
            del st.session_state.balloons_triggered
        # Inyectar starter_line como primer turno fijo de Neobit
        starter_line = active_mission.get("starter_line", "Hello! I would like to order, please.") if active_mission else "Hello! I am ready to start."
        st.session_state.messages = [{"role": "assistant", "spoken": starter_line, "note": ""}]
        st.session_state.ollama_history = [{"role": "assistant", "content": starter_line}]
        
    st.markdown("---")
    # Botón para limpiar historial
    if st.button("🔄 Reiniciar Misión", use_container_width=True):
        starter_line = active_mission.get("starter_line", "Hello! I would like to order, please.") if active_mission else "Hello! I am ready to start."
        st.session_state.messages = [{"role": "assistant", "spoken": starter_line, "note": ""}]
        st.session_state.ollama_history = [{"role": "assistant", "content": starter_line}]
        st.session_state.used_vocabulary = set()
        st.session_state.mission_completed = False
        st.session_state.corrections_summary = []
        st.session_state.show_report = False
        if "balloons_triggered" in st.session_state:
            del st.session_state.balloons_triggered
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 5. CONTENIDO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="premium-title">🚀 Neobit Roleplay English Coach</div>', unsafe_allow_html=True)
st.markdown('<div class="premium-subtitle">Resuelve situaciones reales del día a día conversando en inglés con Neobit en nivel principiante.</div>', unsafe_allow_html=True)

# ── MODAL DE MISIÓN COMPLETADA ──
if st.session_state.mission_completed:
    feedback = st.session_state.get("mission_feedback") or {}
    score = feedback.get("score", 0)
    
    # Modal visual de "Misión Completada"
    st.markdown(f"""
    <div class="mission-complete-modal">
        <h1>🎉 Mission Completed!</h1>
        <div class="subtitle">You successfully completed the roleplay conversation.</div>
        <div class="score-badge">{score}%</div>
        <div style="color: #065f46; font-size: 0.95rem; margin-top: 8px;">Overall Score</div>
        <br/>
    </div>
    """, unsafe_allow_html=True)
    
    # Botón para ver reporte detallado
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if not st.session_state.show_report:
            if st.button("📊 View Detailed Results", type="primary", use_container_width=True):
                st.session_state.show_report = True
                st.rerun()
        else:
            if st.button("🔽 Hide Detailed Results", type="secondary", use_container_width=True):
                st.session_state.show_report = False
                st.rerun()
    
    # Reporte detallado
    if st.session_state.show_report and feedback:
        objectives = feedback.get("objectives", {})
        corrections = feedback.get("corrections", [])
        better_alternatives = feedback.get("better_alternatives", [])
        strengths = feedback.get("strengths", [])
        vocab_used = feedback.get("vocab_used", [])
        suggestions = feedback.get("suggestions", [])
        
        # ── Objetivos ──
        objectives_html = ""
        for k, v in objectives.items():
            display_name = k.replace("_", " ").capitalize()
            mark = '<span class="check">✓</span>' if v else '<span class="cross">✗</span>'
            objectives_html += f'<div class="objective-item">{mark} <strong>{display_name}</strong></div>'
        
        # ── Correcciones ──
        corrections_html = ""
        if corrections:
            for c in corrections:
                original = c.get("original", "")
                corrected = c.get("corrected", "")
                explanation = c.get("explanation", "")
                corrections_html += f"""
                <div class="correction-item">
                    <div><span class="label">Original:</span> <span class="original-text">{original}</span></div>
                    <div><span class="label">Corrected:</span> <span class="corrected-text">{corrected}</span></div>
                    <div class="explanation">{explanation}</div>
                </div>
                """
        else:
            corrections_html = '<div style="color: #64748b; padding: 12px;">No grammar errors detected. Great job!</div>'
        
        # ── Better Alternatives ──
        alternatives_html = ""
        if better_alternatives:
            for a in better_alternatives:
                original = a.get("original", "")
                alternative = a.get("alternative", "")
                reason = a.get("reason", "")
                alternatives_html += f"""
                <div class="alternative-item">
                    <div><span class="label">You said:</span> <span class="orig-text">{original}</span></div>
                    <div><span class="label">Better:</span> <span class="alt-text">{alternative}</span></div>
                    <div class="reason">{reason}</div>
                </div>
                """
        else:
            alternatives_html = '<div style="color: #64748b; padding: 12px;">Your phrasing was natural and appropriate.</div>'
        
        # ── Strengths ──
        strengths_html = "".join([f'<div class="strength-item">{s}</div>' for s in strengths])
        
        # ── Vocabulario ──
        vocab_html = ", ".join(vocab_used) if vocab_used else "<span style='color: #94a3b8;'>None of the target vocabulary was detected.</span>"
        
        # ── Sugerencias ──
        suggestions_html = "".join([f'<div class="suggestion-item">{s}</div>' for s in suggestions])
        
        # Renderizar reporte completo
        st.markdown(f"""
        <div class="report-card">
            <h3>📋 Overall Score</h3>
            <div style="font-size: 2rem; font-weight: 800; color: #065f46;">{score}%</div>
            
            <div class="section-divider"></div>
            
            <h3>✅ Objectives Completed</h3>
            <div class="objectives-grid">
                {objectives_html}
            </div>
            
            <div class="section-divider"></div>
            
            <h3>✏️ Grammar Corrections</h3>
            {corrections_html}
            
            <div class="section-divider"></div>
            
            <h3>💬 Better Alternatives</h3>
            {alternatives_html}
            
            <div class="section-divider"></div>
            
            <h3>✨ What You Did Well</h3>
            {strengths_html}
            
            <div class="section-divider"></div>
            
            <h3>📖 Vocabulary Used</h3>
            <div style="margin-bottom: 16px;">{vocab_html}</div>
            
            <div class="section-divider"></div>
            
            <h3>💡 Suggestions for Improvement</h3>
            {suggestions_html}
        </div>
        """, unsafe_allow_html=True)
    
    # Balloons solo una vez
    if "balloons_triggered" not in st.session_state:
        st.session_state.balloons_triggered = True
        st.balloons()

# ── MISSION BRIEFING CARD ─────────────────────────────────────────────────────
if active_mission and not st.session_state.mission_completed:
    show_pre_mission = not any(m.get("role") == "user" for m in st.session_state.messages)

    st.markdown("""
<div style="background:#ffffff; border:2px solid #e2e8f0; border-radius:20px;
            padding:22px 26px 18px 26px; margin-bottom:20px;
            box-shadow: 0 4px 20px rgba(15,23,42,0.06);">
""", unsafe_allow_html=True)

    # ── Mission title ──
    st.markdown(
        f"<div style='font-size:1.15rem; font-weight:800; color:#4f46e5; "
        f"margin-bottom:14px;'>📍 {active_mission.get('title', 'Mission')}</div>",
        unsafe_allow_html=True
    )

    if show_pre_mission:
        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            st.markdown("**🎭 Your Role**")
            st.markdown(f"> {active_mission.get('user_role', '')}")

            st.markdown("&nbsp;", unsafe_allow_html=True)

            st.markdown("**🎯 Your Mission**")
            st.markdown("> Take the customer's order and complete the service politely.")

            st.markdown("&nbsp;", unsafe_allow_html=True)

            st.markdown("**💡 Suggested Flow**")
            st.markdown(
                "1. Greet the customer  \n"
                "2. Ask what they want to eat  \n"
                "3. Ask what they want to drink  \n"
                "4. Confirm the order  \n"
                "5. Finish politely"
                
            )

        with col_right:
            st.markdown("**🗣 Useful Phrases**")
            st.markdown(
                "• *Welcome to our restaurant.*  \n"
                "• *What would you like to order?*  \n"
                "• *What would you like to drink?*  \n"
                "• *So your order is... Is that correct?*  \n"
                "• *Thank you for your order.*"
            )

            st.markdown("&nbsp;", unsafe_allow_html=True)

            vocab_goals = active_mission.get("vocabulary_goals", [])
            if vocab_goals:
                st.markdown("**🎯 Vocabulary Goals**")
                used = st.session_state.used_vocabulary
                badge_parts = []
                for g in vocab_goals:
                    if g.lower() in used:
                        badge_parts.append(f"✅ **{g}**")
                    else:
                        badge_parts.append(f"_{g}_")
                st.markdown("  ·  ".join(badge_parts))

        st.markdown("&nbsp;", unsafe_allow_html=True)

        with st.expander("📖 Mission Details", expanded=False):
            st.markdown("**📋 Full Scenario**")
            st.info(active_mission.get("narrative_context", ""))

            st.markdown("**🤖 Neobit's Role**")
            st.info(active_mission.get("neobit_role", ""))

            vocab_goals = active_mission.get("vocabulary_goals", [])
            if vocab_goals:
                st.markdown("**🎯 Full Vocabulary List**")
                st.write(", ".join(vocab_goals))

    else:
        # ── Compact in-mission banner ──
        used = st.session_state.used_vocabulary
        vocab_goals = active_mission.get("vocabulary_goals", [])
        badge_parts = []
        for g in vocab_goals:
            if g.lower() in used:
                badge_parts.append(f"✅ **{g}**")
            else:
                badge_parts.append(f"_{g}_")

        st.markdown(
            "<div style='font-size:0.9rem; color:#64748b; margin-bottom:4px;'>"
            "✨ <strong>Mission in progress</strong> — speak or type your reply below."
            "</div>",
            unsafe_allow_html=True
        )
        if badge_parts:
            st.markdown("**Vocabulary:** " + "  ·  ".join(badge_parts))

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. HISTORIAL DE CONVERSACIÓN DE CHAT (VISUAL)
# ─────────────────────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(msg["spoken"])
                # Las correcciones [NOTA] NO se muestran durante la conversación.
                # Nunca aparecen mensajes como "No correction needed", "Grammar OK", etc.

# ─────────────────────────────────────────────────────────────────────────────
# 7. CAPA DE INTERACCIÓN POR VOZ (CONTROLES DE AUDIO)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.write("<div style='text-align: center; font-weight: 600; margin-bottom: 10px; color: #94a3b8;'>CONTROLES DE VOZ</div>", unsafe_allow_html=True)
    
    # Si la misión se completó, deshabilitar grabación y mostrar botón de reinicio centrado
    if st.session_state.mission_completed:
        st.write("<div style='text-align: center; font-weight: 600; color: #10b981; margin-bottom: 15px;'>🏆 MISIÓN COMPLETADA CON ÉXITO</div>", unsafe_allow_html=True)
        if st.button("🔄 Jugar Otra Misión (Reiniciar)", type="primary", use_container_width=True):
            starter_line = active_mission.get("starter_line", "Hello! I would like to order, please.") if active_mission else "Hello! I am ready to start."
            st.session_state.messages = [{"role": "assistant", "spoken": starter_line, "note": ""}]
            st.session_state.ollama_history = [{"role": "assistant", "content": starter_line}]
            st.session_state.used_vocabulary = set()
            st.session_state.mission_completed = False
            st.session_state.corrections_summary = []
            st.session_state.show_report = False
            if "balloons_triggered" in st.session_state:
                del st.session_state.balloons_triggered
            st.rerun()
    # Botón único con cambio dinámico de estado
    elif not st.session_state.is_recording:
        if st.button("🎤 Empezar a Hablar (Grabar)", type="primary", use_container_width=True):
            st.session_state.is_recording = True
            st.session_state.recorder.start()
            st.rerun()
    else:
        st.write("🎙️ **Escuchando...** Habla en inglés al micrófono. Presiona el botón de abajo para terminar.")
        if st.button("⏹️ Terminar y Procesar Respuesta", type="secondary", use_container_width=True):
            st.session_state.is_recording = False
            
            with st.spinner("💾 Deteniendo grabación y extrayendo audio..."):
                audio_data = st.session_state.recorder.stop()
                
            if audio_data is not None and len(audio_data) > 0:
                # ── PASO 1: VAD ──
                with st.spinner("🔍 Analizando voz con Silero VAD..."):
                    timestamps = detect_speech(audio_data, vad_model, vad_utils, fs=16000)
                
                if not timestamps:
                    st.warning("⚠️ No se detectó voz clara. ¡Asegúrate de hablar cerca del micrófono e intenta de nuevo!")
                else:
                    # ── PASO 2: Transcripción ──
                    with st.spinner("✍️ Transcribiendo audio con Whisper..."):
                        resultados = transcribe_segments(audio_data, timestamps, whisper_model, fs=16000)
                    
                    texto_completo = " ".join([r["text"] for r in resultados]).strip()
                    
                    if texto_completo:
                        # Registrar palabras clave usadas
                        for goal in active_mission.get("vocabulary_goals", []):
                            if goal.lower() in texto_completo.lower():
                                st.session_state.used_vocabulary.add(goal.lower())

                        # Agregar el mensaje del usuario
                        st.session_state.messages.append({"role": "user", "content": texto_completo})
                        
                        # ── PASO 3: Ollama ──
                        with st.spinner("🤖 Pensando respuesta como Neobit (Principiante)..."):
                            respuesta, new_history = get_chat_response(
                                user_text=texto_completo,
                                chat_history=st.session_state.ollama_history,
                                difficulty="básico",
                                mission_data=active_mission
                            )
                            
                        st.session_state.ollama_history = new_history
                        
                        note_text = respuesta["note"]
                        spoken_text = respuesta.get("spoken", "")

                        # Evaluar objetivos para 'order_food'
                        if active_mission and active_mission.get("id") == "order_food":
                            all_msgs = st.session_state.messages + [{"role": "assistant", "spoken": spoken_text, "note": note_text}]
                            objectives = evaluate_order_food_objectives(all_msgs)
                            st.session_state.mission_objectives = objectives

                            # Detección de finalización:
                            # 1. Los 3 items elegidos via extract_order_state (fuente de verdad)
                            # 2. El usuario se despidió en este turno
                            user_said_farewell = any(
                                p in texto_completo.lower() for p in FAREWELL_PHRASES
                            )
                            completed = user_said_farewell

                            if completed:
                                st.session_state.mission_completed = True
                                # Análisis completo de TODA la conversación
                                analysis = analyze_conversation(
                                    st.session_state.messages + [{"role": "assistant", "spoken": spoken_text}],
                                    active_mission.get("vocabulary_goals")
                                )

                                # No hay penalización de objetivos ya que todos están completos
                                final_score = analysis.get("score", 100)

                                st.session_state.mission_feedback = {
                                    "score": final_score,
                                    "objectives": objectives,
                                    "corrections": analysis.get("corrections", []),
                                    "better_alternatives": analysis.get("better_alternatives", []),
                                    "strengths": analysis.get("strengths", []),
                                    "vocab_used": analysis.get("vocab_used", []),
                                    "suggestions": analysis.get("suggestions", []),
                                }

                        # Agregar la respuesta del asistente
                        st.session_state.messages.append({
                            "role": "assistant",
                            "spoken": respuesta["spoken"],
                            "note": note_text
                        })

                        st.rerun()
                    else:
                        st.warning("⚠️ No se pudo transcribir ningún texto. Por favor habla de nuevo.")
            else:
                st.error("❌ Ocurrió un problema capturando el audio.")
                
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 8. ENTRADA DE TEXTO ADICIONAL (CHAT INPUT)
# ─────────────────────────────────────────────────────────────────────────────
user_text_input = st.chat_input("Escribe tu respuesta aquí en inglés...", disabled=st.session_state.mission_completed)

if user_text_input and not st.session_state.mission_completed:
    texto_completo = user_text_input.strip()
    
    # Registrar palabras clave
    for goal in active_mission.get("vocabulary_goals", []):
        if goal.lower() in texto_completo.lower():
            st.session_state.used_vocabulary.add(goal.lower())

    # Agregar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": texto_completo})
    
    # ── Ollama ──
    with st.spinner("🤖 Pensando respuesta como Neobit (Principiante)..."):
        respuesta, new_history = get_chat_response(
            user_text=texto_completo,
            chat_history=st.session_state.ollama_history,
            difficulty="básico",
            mission_data=active_mission
        )
        
    st.session_state.ollama_history = new_history
    
    note_text = respuesta["note"]
    spoken_text = respuesta.get("spoken", "")

    # Evaluar objetivos para 'order_food'
    if active_mission and active_mission.get("id") == "order_food":
        all_msgs = st.session_state.messages + [{"role": "assistant", "spoken": spoken_text, "note": note_text}]
        objectives = evaluate_order_food_objectives(all_msgs)
        st.session_state.mission_objectives = objectives

        # Detección de finalización:
        # 1. Los 3 items elegidos via extract_order_state (fuente de verdad)
        # 2. El usuario se despidió en este turno
        user_said_farewell = any(
            p in texto_completo.lower() for p in FAREWELL_PHRASES
        )
        completed = user_said_farewell

        if completed:
            st.session_state.mission_completed = True
            analysis = analyze_conversation(
                st.session_state.messages + [{"role": "assistant", "spoken": spoken_text}],
                active_mission.get("vocabulary_goals")
            )
            final_score = analysis.get("score", 100)

            st.session_state.mission_feedback = {
                "score": final_score,
                "objectives": objectives,
                "corrections": analysis.get("corrections", []),
                "better_alternatives": analysis.get("better_alternatives", []),
                "strengths": analysis.get("strengths", []),
                "vocab_used": analysis.get("vocab_used", []),
                "suggestions": analysis.get("suggestions", []),
            }

    # Agregar respuesta del asistente
    st.session_state.messages.append({
        "role": "assistant",
        "spoken": respuesta["spoken"],
        "note": note_text
    })
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 9. SÍNTESIS DE VOZ DE AZURE EN HILO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
# Reproducir automáticamente la última respuesta del robot si es nueva
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    last_msg = st.session_state.messages[-1]
    if "last_played_msg" not in st.session_state or st.session_state.last_played_msg != last_msg["spoken"]:
        st.session_state.last_played_msg = last_msg["spoken"]
        if tts_synthesizer and last_msg["spoken"]:
            with st.spinner("🗣️ Reproduciendo respuesta por altavoz..."):
                speak_text(last_msg["spoken"], tts_synthesizer)