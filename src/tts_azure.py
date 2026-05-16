import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

def init_speech_synthesizer():
    speech_key = os.getenv('AZURE_SPEECH_KEY')
    service_region = os.getenv('AZURE_SPEECH_REGION')

    if not speech_key or not service_region:
        print("⚠️  Advertencia: No se encontraron las credenciales de Azure Speech en las variables de entorno.")
        print("⚠️  Asegúrate de configurar AZURE_SPEECH_KEY y AZURE_SPEECH_REGION en tu archivo .env.")
        return None

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    
    # Selecciona una voz. Una voz multilingüe como en-US-AvaMultilingualNeural puede hablar inglés y español.
    # Alternativas: "es-ES-ElviraNeural" para solo español, "en-US-AriaNeural" para solo inglés.
    speech_config.speech_synthesis_voice_name = "en-US-AvaMultilingualNeural" 
    
    # Crea un sintetizador de audio para usar el altavoz predeterminado
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    
    return synthesizer

def speak_text(text, synthesizer=None):
    if synthesizer is None:
        return
    
    # Llama a Azure para sintetizar y reproducir el texto
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"❌ La síntesis de voz fue cancelada: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"❌ Error details: {cancellation_details.error_details}")
