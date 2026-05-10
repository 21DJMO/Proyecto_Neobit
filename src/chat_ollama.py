import requests
import json

# Configuraciones
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# Recomendación de modelos ligeros y eficientes:
# 1. "phi3" (3.8B) - Excelente para seguir instrucciones, lógica y gramática. Muy ligero. Ideal para este caso.
# 2. "qwen2:1.5b" - Extremadamente rápido y ligero, requiere menos de 2GB de RAM.
# 3. "llama3" (8B) - El mejor y más inteligente, pero requiere un poco más de RAM (unos 8GB).
DEFAULT_MODEL = "phi3"

def get_chat_response(user_text, model=DEFAULT_MODEL, chat_history=None):
    """
    Envía el texto transcrito al modelo local de Ollama para que analice 
    la gramática, corrija y responda.
    """
    if chat_history is None:
        chat_history = []
        
    system_prompt = (
        "Eres un tutor de inglés. El usuario te enviará frases transcritas de su voz. "
        "Tu tarea es evaluar la gramática y continuar la conversación.\n\n"
        "REGLAS ESTRICTAS:\n"
        "1. No repitas el texto del usuario.\n"
        "2. Usa EXACTAMENTE este formato de dos líneas, sin agregar introducciones ni despedidas:\n"
        "💡 Correcciones: [Tus correcciones en español de forma breve. Si no hay errores, di '¡Perfecto!']\n"
        "🤖 Respuesta: [Tu respuesta conversacional en inglés, preferiblemente terminando con una pregunta para mantener la charla]\n\n"
        "EJEMPLO 1:\n"
        "Usuario: Yesterday I go to the store.\n"
        "💡 Correcciones: El pasado de 'go' es 'went'. Debería ser 'Yesterday I went to the store'.\n"
        "🤖 Respuesta: That's great! What did you buy at the store?\n\n"
        "EJEMPLO 2:\n"
        "Usuario: Hello, how are you?\n"
        "💡 Correcciones: ¡Perfecto!\n"
        "🤖 Respuesta: I'm doing well, thank you! What are you planning to do today?"
    )
    
    # Construir el arreglo de mensajes (formato estándar de chat de OpenAI/Ollama)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_text})
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False # False para recibir la respuesta completa y no palabra por palabra
    }
    
    print(f"\n[Ollama] ⏳ Analizando texto con el modelo '{model}'...")
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        
        result = response.json()
        ai_message = result.get("message", {}).get("content", "")
        
        # Guardamos en el historial para mantener el contexto en futuras iteraciones
        chat_history.append({"role": "user", "content": user_text})
        chat_history.append({"role": "assistant", "content": ai_message})
        
        return ai_message, chat_history
        
    except requests.exceptions.ConnectionError:
        return ("❌ Error: No se pudo conectar con Ollama. Asegúrate de tener Ollama "
                "abierto y ejecutándose en segundo plano."), chat_history
    except Exception as e:
        return f"❌ Error de Ollama: {str(e)}", chat_history

if __name__ == "__main__":
    print("\n--- PRUEBA DEL CHATBOT DE PRONUNCIACIÓN (OLLAMA) ---")
    print("Asegúrate de haber descargado el modelo en la terminal:")
    print("Ejemplo: 'ollama run phi3'")
    
    texto_prueba = "Hello my name is Pedro and I go to the supermarket at yesterday."
    print(f"\n🗣️ Usuario dijo: \"{texto_prueba}\"")
    
    respuesta, _ = get_chat_response(texto_prueba, model="phi3")
    print(f"\n{respuesta}\n")
