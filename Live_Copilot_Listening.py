import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
from io import BytesIO
import time
import os
from groq import Groq
import google.generativeai as genai

# Set up Groq API key
os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

os.environ["API_KEY"] = "AIzaSyCOhsh-JWBd6B006GA0UgdIW6wRcNon7lk"  # Replace with your actual API key
genai.configure(api_key=os.environ["API_KEY"])

# Initialize the Google Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Sample rate for audio recording
SAMPLE_RATE = 44100  # 44.1kHz for better quality

# Function to record audio for a fixed duration
def record_audio(duration=5, sample_rate=SAMPLE_RATE):
    st.write(f"Recording audio for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    return audio_data

# Function to save audio to an in-memory file and send to Whisper API via Groq
def transcribe_audio_with_whisper(audio_data, sample_rate):
    # Save audio data to an in-memory buffer
    audio_buffer = BytesIO()
    sf.write(audio_buffer, audio_data, sample_rate, format='wav')
    audio_buffer.seek(0)  # Rewind the buffer for reading

    # Create a transcription request to Groq API using Whisper model
    st.write("Sending audio to Whisper for transcription...")
    translation = client.audio.translations.create(
        file=("temp_recording.wav", audio_buffer.read()),  # Provide in-memory file
        model="whisper-large-v3",
        prompt="Please provide the transcript of this conversation.",
        response_format="text",
        temperature=0
    )
    
    # Return the transcription result
    transcription = translation
    return transcription

# Function to send the conversation to Google Gemini for summarization, key points, etc.
def summarize_and_analyze(conversation):
    prompt = f"Conversation: {conversation}\n\nPlease summarize, extract key points, suggest clarifications, and write 5 questions."
    response = model.generate_content(prompt)
    return response

# Streamlit App
st.title("Live Conversation Transcription and Analysis with OpenAI Whisper")

st.write("This app captures live audio, transcribes it using OpenAI Whisper, and periodically sends the conversation for summarization, key point extraction, and question generation.")

# Initialize session state for conversation and running status
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

if 'responses' not in st.session_state:
    st.session_state['responses'] = []

if 'running' not in st.session_state:
    st.session_state['running'] = False

# Add a field for the user to specify the frequency of transcription in seconds (default: 30 seconds)
transcription_frequency = st.number_input("Frequency of transcription and summarization (in seconds)", min_value=5, max_value=300, value=30, step=5)

# Function to handle the transcription and updating the UI asynchronously
def handle_transcription(audio_data, sample_rate):
    transcription = transcribe_audio_with_whisper(audio_data, sample_rate)
    if transcription:
        # Append the transcription to the conversation history
        st.session_state['conversation'].append(transcription)
        st.write("Conversation history so far:")
        st.write(" ".join(st.session_state['conversation']))

        # Periodically summarize and analyze the conversation every 5 transcriptions
        if len(st.session_state['conversation']) % 5 == 0:
            conversation = " ".join(st.session_state['conversation'])
            st.write("Sending to Google Gemini for analysis...")
            response = summarize_and_analyze(conversation)
            st.session_state['responses'].append(response.text)
            st.write("Analysis response:")
            st.write(response.text)

# Record and transcribe audio, then append to conversation history
if st.button("Start Transcription") and not st.session_state['running']:
    st.session_state['running'] = True
    st.write("Starting to capture audio...")

# Display the stop button, which is independent of the start button
if st.session_state['running']:
    if st.button("Stop Transcription"):
        st.session_state['running'] = False
        st.write("Transcription stopped.")

# Run the transcription loop while "running" is True
if st.session_state['running']:
    try:
        while st.session_state['running']:  # Run loop as long as running is True
            # Record audio for the duration specified by the user
            audio_data = record_audio(duration=transcription_frequency)

            # Immediately handle transcription after recording
            handle_transcription(audio_data, SAMPLE_RATE)

            # Small delay to prevent continuous looping
            time.sleep(1)

    except KeyboardInterrupt:
        st.session_state['running'] = False
        st.write("Transcription stopped.")

# Display the conversation history even after the stop button is pressed
st.write("### Conversation history:")
st.write(" ".join(st.session_state['conversation']))

# Display the responses from Gemini summarizations
st.write("### Gemini Analysis Responses:")
for i, response in enumerate(st.session_state['responses']):
    st.write(f"Response {i + 1}: {response}")
