import streamlit as st
import sounddevice as sd
from io import BytesIO
import openai
from openai import OpenAI
import os
from pydub import AudioSegment
import wave  # For WAV file handling

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


def save_audio_to_file(audio_data, filename="recording.wav"):
    """Saves the recorded audio data (byte array) to a WAV file.

    Args:
        audio_data: A byte array containing the recorded audio data.
        filename: The desired filename for the WAV file (default: "recording.wav").
    """
    with wave.open(filename, "wb") as wav_file:
    wav_file.setnchannels(2)  # Set number of channels (stereo)
    wav_file.setsampwidth(2)  # Set sample width (16 bits)
    wav_file.setframerate(44100)  # Set frame rate (sampling rate)
    wav_file.writeframes(audio_data)  # Write audio data to the WAV file
    return filename

def transcribe_audio(filename):
    """Transcribes audio data using the OpenAI Whisper model.
    Args:
        audio_data: A byte array containing the recorded audio data in WAV format.
    Returns:
        The transcribed text from the audio, or an error message if transcription fails.
    """
    # Send the converted audio data to OpenAI API
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=filename
    )

    return transcription.text


def app():
    st.title("Speech-to-Text with Streamlit and OpenAI")
    if st.button("Record Audio"):
        recorded_audio = record_audio()
        filename = save_audio_to_file(recorded_audio)  # Save recording to file
        transcription = transcribe_audio(filename)
        st.write("Transcription:")
        st.write(transcription if transcription else "No transcription available.")

if __name__ == "__main__":
    app()   
