import streamlit as st
import sounddevice as sd
from io import BytesIO
import openai
from openai import OpenAI
import os
from pydub import AudioSegment

client = OpenAI(api_key=os.getenv("API_KEY"))
#client = OpenAI(api_key=st.secrets["API_key"])

def record_audio(duration=5):
    """Records audio from the microphone for a specified duration (seconds).
2
    Args:
        duration: The duration of recording in seconds (default: 3).

    Returns:
        A byte array containing the recorded audio data in WAV format.
    """
    fs = 44100  # Sampling rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    st.write("Recording started...")
    sd.wait()
    st.write("Recording complete")
    return recording.tobytes()

def transcribe_audio(audio_data):
    """Transcribes audio data using the OpenAI Whisper model.
    Args:
        audio_data: A byte array containing the recorded audio data in WAV format.
    Returns:
        The transcribed text from the audio, or an error message if transcription fails.
    """

    # Convert byte array to an AudioSegment object
    audio = AudioSegment.from_wav(BytesIO(audio_data))

    # Convert to one of the supported formats
    audio_format = "ogg"  # Choose a supported format
    audio_bytes = audio.export(format=audio_format).read()

    # Send the converted audio data to OpenAI API
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_bytes
    )

    return transcription.text



def app():
    st.title("Speech-to-Text with Streamlit and OpenAI")
    if st.button("Record Audio"):
        recorded_audio = record_audio()
        transcription = transcribe_audio(recorded_audio)
        st.write("Transcription:")
        st.write(transcription if transcription else "No transcription available.")

if __name__ == "__main__":
    app()   
