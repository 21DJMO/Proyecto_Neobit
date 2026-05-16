
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
    Graba audio sin límite de tiempo hasta que el usuario presiona Enter en la consola.
    """
    q = queue.Queue()
    def callback(indata, frames, time, status):
        if status: print(status)
        q.put(indata.copy())
    with sd.InputStream(samplerate=fs, channels=1, dtype='float32', callback=callback):
        input() 
    audio_data = []
    while not q.empty():
        audio_data.append(q.get())
    recording = np.concatenate(audio_data, axis=0)
    wav.write(filename, fs, recording)
    return recording

    wav.write(filename, fs, recording)
    return recording

class StreamlitRecorder:
    """
    Clase para manejar grabación asíncrona en Streamlit.
    """
    def __init__(self, fs=16000):
        self.fs = fs
        self.q = queue.Queue()
        self.stream = None
        self.recording = False

    def callback(self, indata, frames, time, status):
        if status: print(status)
        if self.recording:
            self.q.put(indata.copy())

    def start(self):
        self.recording = True
        self.stream = sd.InputStream(samplerate=self.fs, channels=1, dtype='float32', callback=self.callback)
        self.stream.start()

    def stop(self, filename="streamlit_audio.wav"):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        audio_data = []
        while not self.q.empty():
            audio_data.append(self.q.get())
        
        if not audio_data:
            return None
            
        recording = np.concatenate(audio_data, axis=0)
        wav.write(filename, self.fs, recording)
        return recording

if __name__ == "__main__":
    # Prueba grabando 5 segundos
    audio_data = record_audio(duration=5)
