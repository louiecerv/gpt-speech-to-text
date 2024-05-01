import streamlit as st
import sounddevice as sd
from io import BytesIO
import openai
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("API_KEY"))
#client = OpenAI(api_key=st.secrets["API_key"])

def record_audio(duration=3):
    """Records audio from the microphone for a specified duration (seconds).
2
    Args:
        duration: The duration of recording in seconds (default: 3).

    Returns:
        A byte array containing the recorded audio data in WAV format.
    """
    fs = 44100  # Sampling rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    return recording.tobytes()

def transcribe_audio(audio_data):
    """Transcribes audio data using the OpenAI Whisper model.
    Args:
        audio_data: A byte array containing the recorded audio data in WAV format.
    Returns:
        The transcribed text from the audio, or an error message if transcription fails.
    """

    try:
        # Convert byte array to a file-like object for OpenAI API
        audio_fileb = BytesIO(audio_data)

        audio_file= open(audio_fileb, "rb")
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
        return transcription.text
    except openai.error.OpenAIError as e:
        return f"Error: {e}"

def app():
    st.title("Speech-to-Text with Streamlit and OpenAI")
    if st.button("Record Audio"):
        recorded_audio = record_audio()
        transcription = transcribe_audio(recorded_audio)
        st.write("Transcription:")
        st.write(transcription if transcription else "No transcription available.")

if __name__ == "__main__":
    app()   
