from faster_whisper import WhisperModel
import numpy as np

def load_transcription_model(model_size="base"):
    """
    Carga el modelo Faster-Whisper. 
    Para MVP en CPU y poder detectar Spanglish (inglés y español), 'base' es el ideal 
    ya que ofrece un gran balance entre velocidad y precisión.
    """
    print(f"⏳ Cargando modelo Whisper ({model_size})...")
    # Usamos CPU y compute_type="int8" para que no sature tu RAM ni se ponga lenta la PC
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print("✅ Modelo Whisper cargado exitosamente.")
    return model

def transcribe_segments(audio_data, speech_timestamps, whisper_model, fs=16000):
    """
    Toma el audio completo y los segmentos de voz detectados por Silero,
    recorta esos fragmentos exactos y los pasa por Whisper para obtener el texto.
    """
    if not speech_timestamps:
        print("❌ No hay segmentos de voz para transcribir.")
        return []
        
    print("\n✍️ Transcribiendo tu conversación...")
    transcriptions = []
    
    for i, segment in enumerate(speech_timestamps):
        # 1. Recortar el audio exacto usando los tiempos detectados por Silero
        start_sample = segment['start']
        end_sample = segment['end']
        
        # Whisper espera un arreglo 1D (flatten)
        audio_segment = audio_data[start_sample:end_sample].flatten().astype(np.float32)
        
        # 2. Transcribir el segmento recortado
        # Al no pasarle 'language', Whisper detectará automáticamente el idioma de cada segmento (Spanglish)
        segments_generator, info = whisper_model.transcribe(audio_segment)
        
        # 3. Extraer el texto del generador
        text = " ".join([seg.text for seg in segments_generator]).strip()
        
        # El modelo nos dice qué idioma detectó para este fragmento
        idioma = info.language
        print(f"   -> Segmento {i+1} [{idioma.upper()}]: \"{text}\"")
        
        transcriptions.append({
            "start_time": start_sample / fs,
            "end_time": end_sample / fs,
            "text": text,
            "language": idioma
        })
        
    return transcriptions

if __name__ == "__main__":
    from audio_capture import record_continuous
    from vad_silero import load_vad_model, detect_speech
    
    # 1. Cargar ambos modelos
    vad_model, vad_utils = load_vad_model()
    whisper = load_transcription_model("base")
    
    print("\n--- PRUEBA COMPLETA: VAD + TRANSCRIPCIÓN ---")
    
    # 2. Grabar conversación
    audio = record_continuous(fs=16000)
    
    # 3. Extraer silencios y detectar segmentos con Silero VAD
    timestamps = detect_speech(audio, vad_model, vad_utils, fs=16000)
    
    # 4. Transcribir el texto de cada uno de esos segmentos con Faster-Whisper
    resultados = transcribe_segments(audio, timestamps, whisper, fs=16000)
