import requests
import re
from mission_manager import build_beginner_prompt

OLLAMA_API_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL  = "qwen3:4b"

# ─────────────────────────────────────────────────────────────────────────────
# SISTEMA DE RESPUESTA DUAL
#   [HABLAR] → TTS de Azure  (conversación en inglés, sin etiquetas)
#   [NOTA]   → pantalla      (correcciones en español, solo al final)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {

    "básico": """You are a friendly English teacher for beginners (A1/A2 level).
The user's text comes from voice transcription. Focus on building confidence.

STRICT RULES:
1. [HABLAR]: ALWAYS in English. Max 2 short sentences. End with ONE simple yes/no question.
2. [NOTA]: ALWAYS in Spanish. ONLY include if there is a clear grammar or vocabulary error.
   - Correct grammar (e.g., "I love play" -> "playing", "Colors is" -> "are").
   - If the user asks "Can you explain?", provide a very simple 1-sentence explanation here.
   - Include a simple phonetic guide for ONE tricky word from the user's input.
3. MEMORY: If the user mentions a hobby or name, remember it for future questions.
4. CRITICAL RULE: If NO errors: leave [NOTA] completely EMPTY. Do NOT write any message like "No correction needed", "¡Excelente trabajo!", "Grammar OK", "Good sentence" or similar. Write absolutely nothing inside [NOTA].

FORMAT:
[HABLAR]
(English reply + yes/no question)
[/HABLAR]
[NOTA]
(ONLY if error found: Spanish grammar/vocabulary correction. Otherwise COMPLETELY EMPTY.)
[/NOTA]

EXAMPLES:
User: My favorite colors is green.
[HABLAR]
Green is a beautiful color! Do you like nature?
[/HABLAR]
[NOTA]
Gramática: Se usa "are" para plural: "My favorite colors are green and..."
Pronunciación: Favorite [féi-vo-rit]
[/NOTA]

User: Hello, how are you?
[HABLAR]
I am great, thank you! Are you ready to order?
[/HABLAR]
[NOTA]
[/NOTA]""",

    "intermedio": """You are a conversational English teacher for intermediate students (B1/B2 level).
You must act as a mentor who tracks progress and explains nuances.

STRICT RULES:
1. [HABLAR]: ALWAYS in English. Exactly 2 sentences: one natural reaction + one open-ended question.
2. [NOTA]: ALWAYS in Spanish. ONLY include if there is a clear error.
   - Precision: Correct "explain me" to "explain to me" or "explain it to me". Correct "it is stressful" vs "I am stressed".
   - Explanations: If asked "Can you explain [word]?", provide a brief, clear definition here.
   - Pronunciation: Provide a phonetic guide for 1-2 advanced words used in the turn.
   - Memory: Use the chat history to ask about previously mentioned topics (family, work, location).
3. CRITICAL RULE: If NO errors: leave [NOTA] completely EMPTY. Do NOT write any message.

FORMAT:
[HABLAR]
(Reaction). (One open question)?
[/HABLAR]
[NOTA]
(ONLY if error found: detailed Spanish correction. Otherwise COMPLETELY EMPTY.)
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
2. [NOTA]: ALWAYS in Spanish. ONLY include if there is a clear error.
   - Only correct non-native phrasing or subtle logic errors.
   - Provide "How a native would say it" suggestions.
   - Phonetic guide for sophisticated vocabulary only.
3. MEMORY: Deep context. If the user mentioned a specific problem or goal sessions ago, bring it up.
4. CRITICAL RULE: If NO errors: leave [NOTA] completely EMPTY.

FORMAT:
[HABLAR]
(Natural response + deep question)
[/HABLAR]
[NOTA]
(ONLY if error found: estilo suggestion. Otherwise COMPLETELY EMPTY.)
[/NOTA]"""
}


def parse_response(raw_text: str) -> dict:
    """
    Extrae [HABLAR] y [NOTA] con regex.
    Limpia bloques <think>...</think> de Qwen 3 antes de parsear.
    Fallback seguro si el modelo no respetó el formato.
    """
    # Eliminar bloques de razonamiento interno de Qwen 3
    clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()

    spoken_match = re.search(r'\[HABLAR\](.*?)\[/HABLAR\]', clean_text, re.DOTALL)
    note_match   = re.search(r'\[NOTA\](.*?)\[/NOTA\]',     clean_text, re.DOTALL)
    mission_ended = bool(re.search(r'\[FIN\]', clean_text))

    spoken = spoken_match.group(1).strip() if spoken_match else ""
    note   = note_match.group(1).strip()   if note_match   else ""

    # Limpiar etiquetas sueltas que el modelo pudo haber dejado fuera
    spoken = re.sub(r'\[/?(?:HABLAR|NOTA|HABLA|FIN)\]', '', spoken).strip()
    note   = re.sub(r'\[/?(?:HABLAR|NOTA|HABLA|FIN)\]', '', note).strip()

    # Fallback: si el modelo ignoró el formato completamente
    if not spoken:
        spoken = re.sub(r'\[/?(?:HABLAR|NOTA|HABLA|FIN)\]', '', clean_text).strip()
        note = ""

    return {"spoken": spoken, "note": note, "mission_ended": mission_ended}


def should_force_confirmation(user_text, chat_history):
    # Extract all user texts and assistant texts from the history + current user_text
    user_texts = [m.get("content", "").lower() for m in chat_history if m.get("role") == "user"] + [user_text.lower()]
    assistant_texts = [m.get("content", "").lower() for m in chat_history if m.get("role") == "assistant"]
    
    joined_user = " ".join(user_texts)
    joined_assistant = " ".join(assistant_texts)
    joined_all = joined_user + " " + joined_assistant

    # Check if food is selected (burger and fries mentioned)
    food_selected = ("burger" in joined_all or "hamburger" in joined_all) and ("fries" in joined_all)

    # Check if drink is selected
    drink_triggers = ["drink", "tea", "coffee", "water", "cola", "coke", "soda", "juice", "lemonade", "beverage"]
    drink_selected = any(d in joined_all for d in drink_triggers)

    # Check if user is confirming the order in the current turn
    confirm_triggers = ["your order is", "so that's", "so that is", "confirm", "that will be", "you ordered", "is that correct", "correct?", "is this correct", "so you want"]
    user_confirming_now = any(c in user_text.lower() for c in confirm_triggers)

    return food_selected and drink_selected and user_confirming_now


def get_chat_response(user_text, model=DEFAULT_MODEL, chat_history=None, difficulty="básico", mission_data=None):
    """
    Retorna (dict, chat_history).
    dict tiene claves 'spoken' (para TTS) y 'note' (para pantalla).
    Si mission_data está presente, construye el System Prompt dinámicamente de nivel principiante.
    """
    if chat_history is None:
        chat_history = []

    if mission_data:
        system_prompt = build_beginner_prompt(mission_data)
        if mission_data.get("id") == "order_food" and should_force_confirmation(user_text, chat_history):
            system_prompt += "\n\nCRITICAL CONTEXT: The user is confirming your order. Since food and drink are already selected, you must confirm that the order is correct. Reply with a short polite confirmation of the order (e.g., 'Yes, that's correct. Thank you.', 'That's right. Thank you very much.', or 'Perfect, thank you.'). Do NOT ask any questions, do NOT request any more information, and do NOT open any new topics. Keep it extremely brief and finish the roleplay."
    else:
        system_prompt = SYSTEM_PROMPTS.get(difficulty, SYSTEM_PROMPTS["básico"])

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "repeat_penalty": 1.1,
            "num_ctx": 1024,      # Contexto reducido → mucho más rápido en CPU
        }
    }

    print(f"\n[Ollama] Procesando con '{model}' (nivel: {difficulty})...")
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
        print(f"[Ollama] HTTP status: {response.status_code}")
        response.raise_for_status()

        raw_json = response.json()
        print(f"[Ollama] Respuesta JSON keys: {list(raw_json.keys())}")
        raw = raw_json.get("message", {}).get("content", "").strip()
        print(f"[Ollama] Raw content (primeros 200 chars): {raw[:200]}")
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
            "note": "Error de conexion: Ollama no esta disponible. Asegurate de que Ollama este corriendo con 'ollama serve'."
        }
        return fallback, chat_history
    except requests.exceptions.ReadTimeout:
        print(f"[Ollama] Timeout: el modelo tardo mas de 180 segundos.")
        fallback = {
            "spoken": "I am thinking... that took too long. Please try again.",
            "note": "El modelo tardo demasiado en responder (timeout 180s). El modelo puede estar sobrecargado. Intenta de nuevo o reinicia Ollama."
        }
        return fallback, chat_history
    except Exception as e:
        print(f"[Ollama] ERROR REAL: {type(e).__name__}: {str(e)}")
        fallback = {
            "spoken": "Something went wrong. Let's try again.",
            "note": f"Error: {type(e).__name__}: {str(e)}"
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