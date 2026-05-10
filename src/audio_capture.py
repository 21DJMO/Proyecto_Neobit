
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import queue

def record_audio(duration=5, fs=16000, filename="test_audio.wav"):
    """
    Graba audio desde el micrófono predeterminado por un tiempo fijo.
    """
    print(f"\n🎤 Grabando por {duration} segundos... ¡Habla ahora!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait() 
    print("✅ Grabación finalizada.")
    wav.write(filename, fs, recording)
    return recording

def record_continuous(fs=16000, filename="test_audio.wav"):
    """
    Graba audio sin límite de tiempo hasta que el usuario presiona Enter.
    Ideal para simular una conversación real.
    """
    q = queue.Queue()

    def callback(indata, frames, time, status):
        """Esta función se llama continuamente para ir guardando el audio entrante."""
        if status:
            print(status)
        q.put(indata.copy())

    print("\n🎤 Grabando SIN LÍMITE de tiempo... (Habla todo lo que quieras)")
    print("👉 PRESIONA 'ENTER' EN ESTA CONSOLA PARA DETENER LA GRABACIÓN 👈\n")
    
    # InputStream mantiene el micrófono abierto indefinidamente
    with sd.InputStream(samplerate=fs, channels=1, dtype='float32', callback=callback):
        input() # El programa se pausa aquí hasta que presiones Enter
        
    print("✅ Grabación finalizada. Procesando audio...")
    
    # Recolectamos todos los pedazos de audio que entraron
    audio_data = []
    while not q.empty():
        audio_data.append(q.get())
        
    # Unimos todo en un solo archivo
    recording = np.concatenate(audio_data, axis=0)
    wav.write(filename, fs, recording)
    print(f"📁 Audio guardado exitosamente como: {os.path.abspath(filename)}\n")
    
    return recording

if __name__ == "__main__":
    # Prueba grabando 5 segundos
    audio_data = record_audio(duration=5)
