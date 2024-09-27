import os
import sounddevice as sd
import queue
import vosk
import json

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import re
import string

nltk.download('punkt')
nltk.download('stopwords')


finaltext = []

# Path to the Vosk model directory (update this path on your system)
model_path = r"C:/Users/aleen/Downloads/vosk-model-small-en-us-0.15/vosk-model-small-en-us-0.15"

# Ensure the model path exists
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}. Download it from https://alphacephei.com/vosk/models")
    exit(1)

# Load the Vosk model
model = vosk.Model(model_path)

# Initialize a queue to store the audio data
audio_queue = queue.Queue()

# Callback function to read audio data from the microphone
def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    audio_queue.put(bytes(indata))

# Speech to text conversion function
def speech_to_text():
    # Sample rate for the audio input (adjust as needed for your system)
    sample_rate = 16000
    # Create a Vosk recognizer for the specified sample rate
    recognizer = vosk.KaldiRecognizer(model, sample_rate)

    # Start recording from the microphone
    with sd.RawInputStream(samplerate=sample_rate, blocksize=8000, dtype='int16', channels=1, callback=callback):
        print("Listening... Press Ctrl+C to stop.")
        while True:
            # Retrieve audio data from the queue
            audio_data = audio_queue.get()
            if recognizer.AcceptWaveform(audio_data):
                result = recognizer.Result()
                text = json.loads(result)["text"]
                print(f"Recognized: {text}")
                finaltext.append(text)
            else:
                partial_result = recognizer.PartialResult()
                partial_text = json.loads(partial_result)["partial"]

# Text processing functions
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    text = text.lower()  # Convert to lowercase
    return text

def tokenize_text(text):
    sentences = sent_tokenize(text)
    words = [word_tokenize(sent) for sent in sentences]
    return sentences, words

def create_frequency_distribution(words):
    stop_words = set(stopwords.words('english'))
    freq_dist = defaultdict(int)
    
    for sentence in words:
        for word in sentence:
            if word not in stop_words and word.isalnum():
                freq_dist[word] += 1
    return freq_dist

def score_sentences(sentences, words, freq_dist):
    sentence_scores = defaultdict(int)
    
    for i, sentence in enumerate(words):
        for word in sentence:
            if word in freq_dist:
                sentence_scores[i] += freq_dist[word]
                
    return sentence_scores

def generate_summary(sentences, sentence_scores, top_n=3):
    ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    summary_sentences = [sentences[i] for i, score in ranked_sentences[:top_n]]
    summary = ' '.join(summary_sentences)
    return summary

def extractive_summary(text, top_n=3):
    # Preprocess the text
    cleaned_text = preprocess_text(text)
    
    # Tokenize text
    sentences, words = tokenize_text(cleaned_text)
    
    # Create frequency distribution
    freq_dist = create_frequency_distribution(words)
    
    # Score sentences
    sentence_scores = score_sentences(sentences, words, freq_dist)
    
    # Generate summary
    summary = generate_summary(sentences, sentence_scores, top_n)
    
    return summary

def summary(text):
    summarized_text = extractive_summary(text, top_n=2)
    print("Summary:", summarized_text)

if __name__ == "__main__":
    try:
        # Start speech-to-text recognition
        speech_to_text()
    except KeyboardInterrupt:
        print("\nRecording terminated.")
        # Join the final recognized text into a single string
        text = ' '.join(finaltext)
        # Generate and print summary
        print(text)
        summary(text)
    except Exception as e:
        print(f"Error: {e}")
