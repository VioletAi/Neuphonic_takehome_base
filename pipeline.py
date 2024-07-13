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
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save('response.mp3')
    return 'response.mp3'

def main():
    recognizer = sr.Recognizer()

    # specify path to audio file
    wav_file_path = '/home/wa285/rds/hpc-work/Neuphonic_takehome_base/dataset/LDC2004S13-segment1.wav'

    # recognize speech from the audio file
    recognized_text = recognize_speech_from_audio_file(recognizer, wav_file_path)
    if recognized_text:
        # generate a response using a language model
        response_text = generate_response(recognized_text)
        print("AI Response:", response_text)

        # convert text response to speech
        response_audio_path = text_to_speech(response_text)
    else:
        print("no speech input detected or speech not recognized.")

if __name__ == "__main__":
    main()
