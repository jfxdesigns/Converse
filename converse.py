import os
import torch
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize ChatGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

# Initialize speech recognizer and microphone
recognizer = sr.Recognizer()
microphone = sr.Microphone()

chat_history_path = "chat_history.txt"  # Path to the chat history file

def get_response(user_message):
    os.system("cls")

    # Load chat history from the file
    if os.path.exists(chat_history_path):
        with open(chat_history_path, "r") as chat_history_file:
            chat_history = chat_history_file.read()
    else:
        chat_history = ""

    # Combine user message and chat history
    combined_message = chat_history + user_message

    # Generate response using ChatGPT
    input_ids = tokenizer.encode(combined_message + tokenizer.eos_token, return_tensors="pt")
    response_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response_message = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Save chat history back to the file
    with open(chat_history_path, "w") as chat_history_file:
        chat_history_file.write(combined_message + "\n")

    tts = gTTS(text=response_message, lang="en")

    # Save response as MP3 file
    response_path = os.path.join(os.path.dirname(__file__), "response.mp3")
    tts.save(response_path)

    # Convert the MP3 file to an AudioSegment object
    response_audio = AudioSegment.from_mp3(response_path)

    # Play the response using pydub
    play(response_audio)

while True:
    with microphone as source:
        print("Listening for speech...")
        try:
            audio = recognizer.listen(source, timeout=999999999)
            recognized_text = recognizer.recognize_google(audio)
            print("Recognized: ", recognized_text)
            user_message = recognized_text
            get_response(user_message)

        except sr.WaitTimeoutError:
            print("No speech detected.")
        except sr.UnknownValueError:
            print("Speech could not be recognized.")
