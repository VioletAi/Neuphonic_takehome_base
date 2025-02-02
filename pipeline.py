import speech_recognition as sr
from gtts import gTTS
import threading

import tensorflow as tf
# forces tensorFlow to use only the cpu
tf.config.set_visible_devices([], 'GPU')

import os
os.environ["TRANSFORMERS_CACHE"] = "/home/wa285/rds/hpc-work/Neuphonic_takehome_base/cache"
from transformers import pipeline

import logging
from prometheus_client import start_http_server, Summary, Counter
import time

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing requests')
ERROR_COUNTER = Counter('errors', 'Number of errors encountered', ['type'])

def setup_logging():
    # configure logging to file and console
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='voice_ai_system.log',
                        filemode='a')
    # add console handler to also print the logs
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

setup_logging()

# setup logging
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



def recognize_speech_from_audio_file(recognizer, audio_file_path):
    logging.info(f"Starting processing of audio file: {audio_file_path}")
    with sr.AudioFile(audio_file_path) as source:
        # listen to the entire audio file
        audio = recognizer.record(source)
    try:
        # using google web speech API to recognize audio
        text = recognizer.recognize_google(audio)
        print("Recognized text:", text)
        return text
    except sr.RequestError as e:
        ERROR_COUNTER.labels(type='speech_recognition').inc()
        logging.error(f"Speech recognition API unavailable, Error: {e}")
        return None
    except sr.UnknownValueError:
        ERROR_COUNTER.labels(type='speech_recognition').inc()
        logging.error("Unable to recognize speech from audio")
        return None


# generate a response using a pre-trained large language model.
def generate_response(input_text):
    logging.info(f"Sending text to LLM for response generation: {input_text}")
    pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-Guard-2-8B")
    try:
        response = pipe(input_text)
        generated_text = response[0]['generated_text']
        logging.info(f"Generated response from LLM: {generated_text}")
        return generated_text
    except Exception as e:
        ERROR_COUNTER.labels(type='llm_generation').inc()
        logging.error(f"Error in generating response from LLM: {e}")
        return None


# convert text to speech and save as an MP3 file.
def text_to_speech(text, identifier):
    tts = gTTS(text=text, lang='en')
    mp3_path = f'output/response_{identifier}.mp3'
    tts.save(mp3_path)
    return mp3_path


@REQUEST_TIME.time()
def handle_conversation(audio_file_path, identifier):
    recognizer = sr.Recognizer()
    recognized_text = recognize_speech_from_audio_file(recognizer, audio_file_path)
    if recognized_text:
        response_text = generate_response(recognized_text)
        print("AI Response:", response_text)
        response_audio_path = text_to_speech(response_text, identifier)
        print("Response saved to:", response_audio_path)
    else:
        print("No speech input detected or speech not recognized.")


def main(number_of_conversations):
    # paths to different audio files for each conversation thread
    audio_files = ['dataset/LDC2004S13-segment1.wav', 'dataset/LDC2004S13-segment2.wav', 'dataset/LDC2004S13-segment3.wav']

    threads = []
    for i in range(min(number_of_conversations, len(audio_files))):
        thread = threading.Thread(target=handle_conversation, args=(audio_files[i], i))
        
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    start_http_server(8000)
    # the number of simultaneous conversations to handle
    number_of_conversations = 3  
    main(number_of_conversations)