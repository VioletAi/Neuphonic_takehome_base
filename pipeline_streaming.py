import pyaudio
import wave
import speech_recognition as sr
import logging
from prometheus_client import start_http_server, Summary, Counter
from gtts import gTTS
from transformers import pipeline
import os
import threading
from speech_recognition import AudioData

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing requests')
ERROR_COUNTER = Counter('errors', 'Number of errors encountered', ['type'])


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


def simulate_audio_stream(audio_file_path, recognizer, chunk_size=65000):

    wf = wave.open(audio_file_path, 'rb')

    logging.info("Starting simulated real-time speech recognition")

    data = wf.readframes(chunk_size)

    while data != b'':
        audio = AudioData(data, wf.getframerate(), wf.getsampwidth() * wf.getnchannels())
        try:
            text = recognizer.recognize_google(audio)
            logging.info(f"Recognized text: {text}")
            yield text
        except sr.UnknownValueError:
            logging.warning("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            logging.error(f"Could not request results from Google Speech Recognition service; {e}")

        # get the next chunk of data
        data = wf.readframes(chunk_size)

    wf.close()

def generate_response(input_text):
    logging.info(f"Sending text to LLM for response generation: {input_text}")
    model_name = "meta-llama/Meta-Llama-Guard-2-8B"
    os.environ["TRANSFORMERS_CACHE"] = "/home/your_user/.cache/huggingface/transformers"
    pipe = pipeline("text-generation", model=model_name)
    try:
        response = pipe(input_text, max_length=50, num_return_sequences=1)
        generated_text = response[0]['generated_text']
        logging.info(f"Generated response from LLM: {generated_text}")
        return generated_text
    except Exception as e:
        ERROR_COUNTER.labels(type='llm_generation').inc()
        logging.error(f"Error in generating response from LLM: {e}")
        return None

def text_to_speech(text, identifier):
    tts = gTTS(text=text, lang='en')
    mp3_path = f'output/response_{identifier}.mp3'
    tts.save(mp3_path)
    logging.info(f"Response saved to: {mp3_path}")
    return mp3_path

def handle_recognized_text(recognized_text, identifier):
    response_text = generate_response(recognized_text)
    if response_text:
        response_audio_path = text_to_speech(response_text, identifier)
    else:
        logging.info("No valid response generated.")

def start_simulated_streaming_recognition(audio_file_path, identifier):
    recognizer = sr.Recognizer()
    for recognized_text in simulate_audio_stream(audio_file_path, recognizer):
        handle_recognized_text(recognized_text, identifier)

if __name__ == "__main__":
    start_http_server(8000)
    audio_file_path = '/home/wa285/rds/hpc-work/Neuphonic_takehome_base/dataset/LDC2004S13-segment1.wav'

    thread = threading.Thread(target=start_simulated_streaming_recognition, args=(audio_file_path, "session1"))
    thread.start()
