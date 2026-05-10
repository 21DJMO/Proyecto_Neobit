from audio_capture import record_continuous
from vad_silero import load_vad_model, detect_speech
from transcribe_whisper import load_transcription_model, transcribe_segments
from chat_ollama import get_chat_response

def main():
    print("\n" + "="*50)
    print("🚀 INICIANDO ASISTENTE DE IDIOMAS NEOBIT")
    print("="*50)
    
    # 1. Cargar modelos locales
    print("\n[1/3] Cargando VAD (Detección de voz)...")
    vad_model, vad_utils = load_vad_model()
    
    print("\n[2/3] Cargando Whisper (Transcripción)...")
    whisper_model = load_transcription_model("base")
    
    # Nota: Ollama ya está cargado localmente como servicio. 
    # Asegúrate de usar un modelo que hayas descargado, e.g. "phi3"
    llm_model = "phi3" 
    print(f"\n[3/3] Usando Ollama LLM (Modelo: {llm_model})...")
    
    # Historial de conversación
    chat_history = []
    
    print("\n" + "="*50)
    print("¡Todo listo! Habla al micrófono para empezar. Presiona Ctrl+C para salir.")
    print("="*50 + "\n")
    
    try:
        while True:
            # 2. Capturar audio
            audio = record_continuous(fs=16000)
            
            # 3. Detectar voz
            timestamps = detect_speech(audio, vad_model, vad_utils, fs=16000)
            
            # 4. Transcribir
            resultados = transcribe_segments(audio, timestamps, whisper_model, fs=16000)
            
            if not resultados:
                continue
                
            # Unir todo el texto detectado en esta iteración
            texto_completo = " ".join([r["text"] for r in resultados]).strip()
            
            if not texto_completo:
                continue
                
            print(f"\n🗣️ Tú dijiste: {texto_completo}")
            
            # 5. Enviar a Ollama para evaluar gramática y responder
            respuesta, chat_history = get_chat_response(
                user_text=texto_completo, 
                model=llm_model, 
                chat_history=chat_history
            )
            
            print(f"\n{respuesta}\n")
            print("-" * 50)
            
    except KeyboardInterrupt:
        print("\n\n👋 ¡Hasta luego! Cerrando asistente de idiomas...")

if __name__ == "__main__":
    main()
