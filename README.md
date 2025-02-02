# End-to-End Voice AI System

This repository contains the source code for an end-to-end Voice AI system designed to convert user speech input into a responsive speech output. The system incorporates speech recognition, a large language model (LLM) for processing and understanding the input, and text-to-speech (TTS) technology to deliver the final spoken response. This project aims to streamline interactions with AI by allowing for a natural spoken dialogue interface.

## System Overview

The system operates in several sequential stages:

1. **Speech Recognition**: Converts audio input into text using Google's Web Speech API via the `speech_recognition` library.
2. **LLM Processing**: Processes the recognized text through a pre-trained large language model (`meta-llama/Meta-Llama-Guard-2-8B`) to generate appropriate responses.
3. **Text-to-Speech (TTS)**: Converts the generated text responses back into speech using the `gTTS` library and saves it as an MP3 file.

### Running the System
Run the system using:

```bash
python pipline.py
```
## Usage

1. Ensure you have an audio file (WAV format) ready to be processed. Specify the path to this file in the `main()` function of the `pipeline.py` script.
2. Execute the script. The system will process the audio file, generate a response, and output the result as an MP3 file.

## Logging and Monitoring

The system includes detailed logging and monitoring to track the processing of user requests and the responses generated by the LLM. Logging is implemented using Python's `logging` library:

- **Logging**: System activities, such as the start and end of speech recognition, errors, and the generation of responses, are logged into `voice_ai_system.log`. This helps in debugging and maintaining the operational history.
- **Monitoring**: Using Prometheus, it monitors the system's performance and error rates. Metrics like request processing time and error counts are exposed on a web server running on port 8000. This allows for real-time monitoring and alerting based on these metrics.

## Concurrent Conversations

The system is capable of handling multiple conversations simultaneously on a single server. This is achieved through multithreading, where each conversation is processed in a separate thread:

- **Threading**: The system can initiate multiple threads to process several audio inputs concurrently, thus enabling real-time interaction for multiple users.
