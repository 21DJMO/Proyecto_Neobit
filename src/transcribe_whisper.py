from faster_whisper import WhisperModel
import numpy as np

def load_transcription_model(model_size="base"):
    print(f"⏳ Cargando modelo Whisper ({model_size})...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print("✅ Modelo Whisper cargado exitosamente.")
    return model

def transcribe_segments(audio_data, speech_timestamps, whisper_model, fs=16000):
    if not speech_timestamps:
        print("❌ No hay segmentos de voz para transcribir.")
        return []

    print("\n✍️ Transcribiendo tu conversación...")
    transcriptions = []

    for i, segment in enumerate(speech_timestamps):
        start_sample = segment['start']
        end_sample   = segment['end']

        audio_segment = audio_data[start_sample:end_sample].flatten().astype(np.float32)

        # IMPORTANTE: convertir el generador a lista INMEDIATAMENTE.
        # Si no se hace esto, el generador puede ser iterado varias veces
        # por distintas partes del código y Whisper duplica la transcripción.
        segments_generator, info = whisper_model.transcribe(
            audio_segment,
            beam_size=5,          # Más preciso que el default (1)
            vad_filter=True,      # Whisper filtra su propio silencio interno
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        segments_list = list(segments_generator)  # <- fuerza el generador una sola vez
        text = " ".join([seg.text for seg in segments_list]).strip()

        idioma = info.language
        print(f"   -> Segmento {i+1} [{idioma.upper()}]: \"{text}\"")

        transcriptions.append({
            "start_time": start_sample / fs,
            "end_time":   end_sample   / fs,
            "text":       text,
            "language":   idioma
        })

    return transcriptions