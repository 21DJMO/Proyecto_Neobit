from audio_capture import record_continuous
from vad_silero import load_vad_model, detect_speech
from transcribe_whisper import load_transcription_model, transcribe_segments
from chat_ollama import get_chat_response
from tts_azure import init_speech_synthesizer, speak_text


def print_separator(char="─", width=52):
    print(char * width)


def main():
    print("\n" + "="*52)
    print("🚀  ASISTENTE DE IDIOMAS NEOBIT")
    print("="*52)

    # ── Cargar modelos ────────────────────────────────────────
    print("\n[1/4] Cargando VAD (detección de voz)...")
    vad_model, vad_utils = load_vad_model()

    print("\n[2/4] Cargando Whisper (transcripción)...")
    whisper_model = load_transcription_model("base")

    llm_model = "llama3.2"
    print(f"\n[3/4] Usando Ollama (modelo: {llm_model})...")

    print("\n[4/4] Inicializando Azure Text-to-Speech...")
    tts_synthesizer = init_speech_synthesizer()

    # ── Seleccionar nivel ─────────────────────────────────────
    print("\n" + "-"*52)
    print("Elige tu nivel de inglés:")
    print("  1. Básico      — vocabulario simple, oraciones cortas")
    print("  2. Intermedio  — conversación normal (B1/B2)")
    print("  3. Avanzado    — expresiones nativas (C1/C2)")
    opcion = input("\nNivel (1, 2 o 3) [por defecto: 2]: ").strip()

    difficulty = {"1": "básico", "3": "avanzado"}.get(opcion, "intermedio")
    print(f"✅ Nivel: {difficulty.capitalize()}")

    # ── Historial de conversación ─────────────────────────────
    chat_history = []

    print("\n" + "="*52)
    print("¡Todo listo! Habla al micrófono.")
    print("Presiona ENTER para parar la grabación.")
    print("Presiona Ctrl+C para salir.")
    print("="*52 + "\n")

    try:
        while True:
            # 1. Capturar audio
            audio = record_continuous(fs=16000)

            # 2. Detectar segmentos de voz
            timestamps = detect_speech(audio, vad_model, vad_utils, fs=16000)

            # 3. Transcribir
            resultados = transcribe_segments(audio, timestamps, whisper_model, fs=16000)

            if not resultados:
                print("⚠️  No detecté voz clara. Intenta de nuevo.\n")
                continue

            texto_completo = " ".join([r["text"] for r in resultados]).strip()

            if not texto_completo:
                continue

            print(f"\n🗣️  Tú dijiste: {texto_completo}")

            # 4. Obtener respuesta dual del modelo
            respuesta, chat_history = get_chat_response(
                user_text=texto_completo,
                model=llm_model,
                chat_history=chat_history,
                difficulty=difficulty
            )

            # 5. Mostrar en pantalla
            #    ├── La conversación (lo que el TTS dirá)
            #    └── Las correcciones (solo visual)
            print()
            print(f"🤖  Profesor: {respuesta['spoken']}")

            if respuesta["note"]:
                print_separator()
                print(f"📋  DETALLES DEL TURNO:")
                print(f"{respuesta['note']}")

            print_separator()

            # 6. TTS solo recibe el texto conversacional limpio
            if tts_synthesizer:
                speak_text(respuesta["spoken"], tts_synthesizer)

            print()

    except KeyboardInterrupt:
        print("\n\n👋 ¡Hasta luego! Cerrando asistente...")


if __name__ == "__main__":
    main()