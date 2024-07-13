import speech_recognition as sr
from gtts import gTTS
import threading

import tensorflow as tf
# forces tensorFlow to use only the cpu
tf.config.set_visible_devices([], 'GPU')

import os
os.environ["TRANSFORMERS_CACHE"] = "/home/wa285/rds/hpc-work/Neuphonic_takehome_base/cache"
from transformers import pipeline

# recognize speech from an audio file.
def recognize_speech_from_audio_file(recognizer, audio_file_path):
    with sr.AudioFile(audio_file_path) as source:
        print("Processing audio file...")
        # listen to the entire audio file
        audio = recognizer.record(source)  
    try:
        # using google web speech API to recognize audio
        text = recognizer.recognize_google(audio)
        print("Recognized text:", text)
        return text
    except sr.RequestError:
        print("API unavailable")
    except sr.UnknownValueError:
        print("Unable to recognize speech")

# generate a response using a pre-trained large language model.
def generate_response(input_text):
    pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-Guard-2-8B")
    ans=pipe(input_text)
    return ans[0]['generated_text']


# convert text to speech and save as an MP3 file.
def text_to_speech(text, identifier):
    tts = gTTS(text=text, lang='en')
    mp3_path = f'response_{identifier}.mp3'
    tts.save(mp3_path)
    return mp3_path

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
    # the number of simultaneous conversations you want to handle
    number_of_conversations = 3  
    main(number_of_conversations)