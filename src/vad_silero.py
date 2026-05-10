import torch
import numpy as np

def load_vad_model():
    """
    Descarga (la primera vez) y carga el modelo pre-entrenado de Silero VAD
    directamente desde el hub oficial de PyTorch.
    """
    print("⏳ Cargando modelo Silero VAD...")
    # trust_repo=True es necesario en las nuevas versiones de PyTorch
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  trust_repo=True)
    print("✅ Modelo Silero VAD cargado exitosamente.")
    return model, utils

def detect_speech(audio_data, model, utils, fs=16000):
    """
    Toma un arreglo de numpy con audio y detecta los segmentos de voz exactos.
    Retorna una lista de diccionarios con el tiempo de inicio y fin (en milisegundos).
    """
    (get_speech_timestamps, _, read_audio, _, _) = utils

    # Silero VAD requiere que el audio sea un Tensor de PyTorch en formato 1D (Float32)
    # Primero aplanamos el arreglo de numpy por si tiene dimensiones extras
    audio_tensor = torch.from_numpy(audio_data.flatten()).float()

    print("🔍 Analizando el audio para detectar habla...")
    
    # get_speech_timestamps extrae las zonas donde hubo voz.
    # threshold: Sensibilidad (0.5 por defecto).
    # min_silence_duration_ms: Tiempo mínimo de silencio (en milisegundos) para considerar que la persona dejó de hablar. 
    #                          (Ej. 1000ms = 1 segundo de pausa permitida sin cortar el segmento)
    # speech_pad_ms: Milisegundos extras de margen al inicio y final del audio para no cortar palabras secas.
    speech_timestamps = get_speech_timestamps(
        audio_tensor, 
        model, 
        sampling_rate=fs, 
        threshold=0.5,
        min_silence_duration_ms=1000, 
        speech_pad_ms=200
    )    
    if not speech_timestamps:
        print("❌ No se detectó ninguna voz clara en el audio.")
    else:
        print(f"🗣️ Se detectaron {len(speech_timestamps)} segmento(s) de voz pura:")
        for i, segment in enumerate(speech_timestamps):
            # Convierte las muestras (samples) a segundos reales
            start_sec = segment['start'] / fs
            end_sec = segment['end'] / fs
            print(f"   -> Segmento {i+1}: inicio {start_sec:.2f}s | fin {end_sec:.2f}s")
            
    return speech_timestamps

if __name__ == "__main__":
    # Importamos nuestra nueva función de grabación continua
    from audio_capture import record_continuous
    
    # 1. Cargamos el modelo (esto puede tardar unos segundos la primera vez)
    vad_model, vad_utils = load_vad_model()
    
    print("\n--- PRUEBA DE VAD (CONVERSACIÓN SIN LÍMITE) ---")
    
    # 2. Grabamos audio sin límite de tiempo
    audio = record_continuous(fs=16000)
    
    # 3. Detectamos exactamente cuándo hablaste y cuándo hiciste silencio en toda la conversación
    timestamps = detect_speech(audio, vad_model, vad_utils, fs=16000)
