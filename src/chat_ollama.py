import requests
import re

OLLAMA_API_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL  = "llama3.2"

# ─────────────────────────────────────────────────────────────────────────────
# SISTEMA DE RESPUESTA DUAL
#   [HABLAR] → TTS de Azure  (conversación en inglés, sin etiquetas)
#   [NOTA]   → pantalla      (correcciones en español)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {

    "básico": """You are a friendly English teacher for beginners (A1/A2 level).
The user's text comes from voice transcription. Focus on building confidence.

STRICT RULES:
1. [HABLAR]: ALWAYS in English. Max 2 short sentences. End with ONE simple yes/no question.
2. [NOTA]: ALWAYS in Spanish. 
   - Correct grammar (e.g., "I love play" -> "playing", "Colors is" -> "are").
   - If the user asks "Can you explain?", provide a very simple 1-sentence explanation here.
   - Include a simple phonetic guide for ONE tricky word from the user's input.
3. MEMORY: If the user mentions a hobby or name, remember it for future questions.
4. If no errors: write "¡Excelente trabajo!" and a small tip.

FORMAT:
[HABLAR]
(English reply + yes/no question)
[/HABLAR]
[NOTA]
Gramática: (Corrección o ¡Sin errores!)
Pronunciación: (Palabra [fonética-simple])
[/NOTA]

EXAMPLES:
User: My favorite colors is green.
[HABLAR]
Green is a beautiful color! Do you like nature?
[/HABLAR]
[NOTA]
Gramática: Se usa "are" para plural: "My favorite colors are green and..."
Pronunciación: Favorite [féi-vo-rit]
[/NOTA]""",

    "intermedio": """You are a conversational English teacher for intermediate students (B1/B2 level).
You must act as a mentor who tracks progress and explains nuances.

STRICT RULES:
1. [HABLAR]: ALWAYS in English. Exactly 2 sentences: one natural reaction + one open-ended question.
2. [NOTA]: ALWAYS in Spanish. 
   - Precision: Correct "explain me" to "explain to me" or "explain it to me". Correct "it is stressful" vs "I am stressed".
   - Explanations: If asked "Can you explain [word]?", provide a brief, clear definition here.
   - Pronunciation: Provide a phonetic guide for 1-2 advanced words used in the turn.
   - Memory: Use the chat history to ask about previously mentioned topics (family, work, location).
3. If no errors: write "¡Muy fluido! Sigue así."

FORMAT:
[HABLAR]
(Reaction). (One open question)?
[/HABLAR]
[NOTA]
Gramática: (Corrección detallada en español)
Pronunciación: (Palabra [fonética-en-español])
Progreso: (Menciona si usó una palabra nueva o si mejoró un error previo)
[/NOTA]

EXAMPLES:
User: Can you explain me what unwind means? My two favorite colors is green.
[HABLAR]
Sure! Unwind means to relax after a busy day. What do you usually do to unwind on weekends?
[/HABLAR]
[NOTA]
Gramática: 
- "Explain me" -> "Explain TO me". 
- "Colors is" -> "Colors ARE" (plural).
Pronunciación: Unwind [an-uáind]
Progreso: ¡Buena pregunta! "Unwind" es una palabra de nivel avanzado.
[/NOTA]""",

    "avanzado": """You are an elite English coach (C1/C2 level). 
Focus on idioms, natural flow, and sophisticated vocabulary.

STRICT RULES:
1. [HABLAR]: ALWAYS in English. 2-3 fluent sentences. Use ONE thought-provoking question.
2. [NOTA]: ALWAYS in Spanish.
   - Only correct non-native phrasing or subtle logic errors.
   - Provide "How a native would say it" suggestions.
   - Phonetic guide for sophisticated vocabulary only.
3. MEMORY: Deep context. If the user mentioned a specific problem or goal sessions ago, bring it up.

FORMAT:
[HABLAR]
(Natural response + deep question)
[/HABLAR]
[NOTA]
Estilo: (Sugerencia de naturalidad)
Pronunciación: (Guía fonética para términos complejos)
Contexto: (Referencia a temas pasados si aplica)
[/NOTA]"""
}


def parse_response(raw_text: str) -> dict:
    """
    Extrae [HABLAR] y [NOTA] con regex.
    Limpia cualquier etiqueta suelta que el modelo haya dejado fuera de los bloques.
    Fallback seguro si el modelo no respetó el formato.
    """
    spoken_match = re.search(r'\[HABLAR\](.*?)\[/HABLAR\]', raw_text, re.DOTALL)
    note_match   = re.search(r'\[NOTA\](.*?)\[/NOTA\]',     raw_text, re.DOTALL)

    spoken = spoken_match.group(1).strip() if spoken_match else ""
    note   = note_match.group(1).strip()   if note_match   else ""

    # Limpiar etiquetas sueltas que el modelo pudo haber dejado fuera
    spoken = re.sub(r'\[/?(?:HABLAR|NOTA|HABLA)\]', '', spoken).strip()
    note   = re.sub(r'\[/?(?:HABLAR|NOTA|HABLA)\]', '', note).strip()

    # Fallback: si el modelo ignoró el formato completamente
    if not spoken:
        spoken = re.sub(r'\[/?(?:HABLAR|NOTA|HABLA)\]', '', raw_text).strip()
        note = ""

    return {"spoken": spoken, "note": note}


def get_chat_response(user_text, model=DEFAULT_MODEL, chat_history=None, difficulty="intermedio"):
    """
    Retorna (dict, chat_history).
    dict tiene claves 'spoken' (para TTS) y 'note' (para pantalla).
    """
    if chat_history is None:
        chat_history = []

    system_prompt = SYSTEM_PROMPTS.get(difficulty, SYSTEM_PROMPTS["intermedio"])

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "repeat_penalty": 1.2,   # Más alto para desincentivar repetir preguntas
        }
    }

    print(f"\n[Ollama] ⏳ Procesando con '{model}' (nivel: {difficulty})...")
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()

        raw = response.json().get("message", {}).get("content", "").strip()
        parsed = parse_response(raw)

        # Historial: solo el bloque conversacional (no las notas de corrección)
        chat_history.append({"role": "user",      "content": user_text})
        chat_history.append({"role": "assistant", "content": parsed["spoken"]})

        if len(chat_history) > 20:
            chat_history = chat_history[-20:]

        return parsed, chat_history

    except requests.exceptions.ConnectionError:
        fallback = {
            "spoken": "Sorry, I could not connect. Please make sure Ollama is running.",
            "note": "❌ Error: Ollama no disponible."
        }
        return fallback, chat_history
    except Exception as e:
        fallback = {
            "spoken": "Something went wrong. Let's try again.",
            "note": f"❌ Error: {str(e)}"
        }
        return fallback, chat_history


if __name__ == "__main__":
    print("\n--- PRUEBA RÁPIDA ---\n")
    casos = [
        ("básico",     "Hello, what's up?"),
        ("básico",     "I love play soccer every Sunday"),
        ("básico",     "I feel so stressful in crowded places"),
        ("intermedio", "I never went to the stadium because I don't like be with a lot of people"),
        ("avanzado",   "I am very agree with that opinion"),
    ]
    for nivel, texto in casos:
        print(f"[{nivel.upper()}] \"{texto}\"")
        res, _ = get_chat_response(texto, difficulty=nivel)
        print(f"  TTS  → {res['spoken']}")
        print(f"  Nota → {res['note']}\n")