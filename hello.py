import speech_recognition as sr
from gtts import gTTS
import threading
import logging
import time
from transformers import pipeline
from prometheus_client import start_http_server, Summary, Counter

# Initialize Prometheus metrics
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing requests')
ERROR_COUNTER = Counter('errors', 'Number of errors encountered', ['type'])

# Setup logging
def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='voice_ai_system.log',
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

setup_logging()

def recognize_speech_from_audio_file(recognizer, audio_file_path):
    logging.info(f"Starting processing of audio file: {audio_file_path}")
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)  # listen to the entire audio file
    try:
        text = recognizer.recognize_google(audio)
        logging.info(f"Recognized text: {text}")
        return text
    except sr.RequestError as e:
        ERROR_COUNTER.labels(type='speech_recognition').inc()
        logging.error("API unavailable or request failed")
        return None
    except sr.UnknownValueError:
        ERROR_COUNTER.labels(type='speech_recognition').inc()
        logging.error("Unable to recognize speech from audio")
        return None

def generate_response(input_text):
    logging.info(f"Sending text to LLM for response generation: {input_text}")
    pipe = pipeline("text-generation", model="gpt2")  # Adjust the model as necessary
    try:
        response = pipe(input_text, max_length=50)
        generated_text = response[0]['generated_text']
        logging.info(f"Generated response from LLM: {generated_text}")
        return generated_text
    except Exception as e:
        ERROR_COUNTER.labels(type='llm_generation').inc()
        logging.error(f"Error in generating response from LLM: {e}")
        return None

def text_to_speech(text, identifier):
    tts = gTTS(text=text, lang='en')
    mp3_path = f'response_{identifier}.mp3'
    tts.save(mp3_path)
    return mp3_path

@REQUEST_TIME.time()
def handle_conversation(audio_file_path, identifier):
    recognizer = sr.Recognizer()
    recognized_text = recognize_speech_from_audio_file(recognizer, audio_file_path)
    if recognized_text:
        response_text = generate_response(recognized_text)
        logging.info("AI Response: " + response_text)
        response_audio_path = text_to_speech(response_text, identifier)
        logging.info(f"Response saved to: {response_audio_path}")
    else:
        logging.info("No speech input detected or speech not recognized.")

def main(number_of_conversations):
    audio_files = ['/path/to/audio1.wav', '/path/to/audio2.wav', '/path/to/audio3.wav']
    threads = []
    for i in range(min(number_of_conversations, len(audio_files))):
        thread = threading.Thread(target=handle_conversation, args=(audio_files[i], i))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    start_http_server(8000)  # Start metrics server on port 8000
    main(3)  # Adjust the number of conversations as necessary
