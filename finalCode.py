import sounddevice as sd
import numpy as np
import torch
from transformers import BartForConditionalGeneration, BartTokenizer, BertTokenizer, BertForQuestionAnswering
import speech_recognition as sr

# Speech-to-text function
def speech_to_text(samplerate=16000, device_index=1):
    """
    Record audio until interrupted and transcribe the whole audio.
    """
    print("Recording. Press Ctrl+C to stop and transcribe.")
    audio_frames = []
    try:
        while True:
            chunk = sd.rec(
                int(5 * samplerate),
                samplerate=samplerate,
                channels=1,
                dtype="float32",
                device=device_index,
            )
            sd.wait()
            audio_frames.append(chunk)
    except KeyboardInterrupt:
        print("\nRecording stopped. Transcribing audio...")
        audio_data = np.concatenate(audio_frames, axis=0)
        audio_data = np.squeeze(audio_data)
        audio_data = (audio_data * 32767).astype(np.int16)

        try:
            recognizer = sr.Recognizer()
            audio = sr.AudioData(audio_data.tobytes(), samplerate, 2)
            text = recognizer.recognize_google(audio)
            print("Transcription complete.")
            return text
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand the audio."
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

# Summarization function
def summarize_large_text(text, model_name='facebook/bart-large-cnn', max_chunk_length=1024, max_summary_length=250, min_summary_length=30):
    """
    Summarize large text by splitting it into smaller chunks.
    """
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer.encode(text, return_tensors="pt", truncation=False)
    chunk_size = max_chunk_length - 2
    chunks = [inputs[0][i:i + chunk_size] for i in range(0, len(inputs[0]), chunk_size)]

    summaries = []
    for chunk in chunks:
        inputs_chunk = chunk.unsqueeze(0)
        summary_ids = model.generate(
            inputs_chunk,
            max_length=max_summary_length,
            min_length=min_summary_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return " ".join(summaries)

# BERT chatbot class
class BertChatbot:
    def __init__(self, model_name="bert-large-uncased-whole-word-masking-finetuned-squad"):
        """
        Initialize the chatbot with a pre-trained BERT model for question answering.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForQuestionAnswering.from_pretrained(model_name)

    def get_answer(self, question, context):
        """
        Answer a question based on the given context.
        """
        inputs = self.tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].tolist()[0]
        outputs = self.model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores) + 1

        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids[start_idx:end_idx])
        )
        return answer.strip()

# Main function
def main():
    chatbot = BertChatbot()
    print("BERT Chatbot: Hello! Would you like to record audio? (yes/no)")
    response = input("You: ").strip().lower()

    if response in ["yes", "y"]:
        context = speech_to_text()
        if "Google Speech Recognition could not understand" in context:
            print("BERT Chatbot: I couldn't understand the audio. Please try again.")
            return
        print("\nBERT Chatbot: Here's the transcribed text:")
        print(context)

        summary = summarize_large_text(context)
        print("\nBERT Chatbot: Here's the summarized text:")
        print(summary)
    else:
        print("BERT Chatbot: No audio was recorded. Goodbye!")
        return

    print("\nBERT Chatbot: You can now ask questions about the recorded context. Type 'exit' to quit.")
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "bye"]:
            print("BERT Chatbot: Goodbye!")
            break

        answer = chatbot.get_answer(question, context)
        print(f"BERT Chatbot: {answer}")

if __name__ == "__main__":
    main()
