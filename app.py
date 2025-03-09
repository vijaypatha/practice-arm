import pyaudio
import wave
import numpy as np
import whisper
import openai
import os

# Set up audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 30
WAVE_OUTPUT_FILENAME = "conversation.wav"

# Record audio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Recording...")
frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording finished.")

stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded audio to a WAV file
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

# Transcribe the audio using Whisper
print("Transcribing...")
model = whisper.load_model("base")
result = model.transcribe(WAVE_OUTPUT_FILENAME)
transcription = result["text"]
print("Transcription:", transcription)

# Summarize the transcription using OpenAI API
print("Summarizing...")
openai.api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",  # Changed from "gpt-4o" to "gpt-4"
    messages=[
        {
            "role": "system",
            "content": "You are a helpful customer assistant that summarizes conversations and generates follow-up SMS to delight the customer."
        },
        {
            "role": "user",
            "content": f"Based on the conversation text, please generate an SMS in 50 words or less:\n\n{transcription}"
        },
        {
            "role": "user",
            "content": f"""Based on the {transcription} conversation, you are an AI assistant analyzing customer call transcripts. Your goal is to estimate Customer Lifetime Value (CLV) and identify cross-selling opportunities based solely on the content of the conversations. Follow these steps:

Part 1: Estimate CLV
1. Analyze Purchase Intent: Extract phrases or themes that indicate interest in products or services.
2. Assess Sentiment: Perform sentiment analysis to classify the customer's tone as positive, neutral, or negative. Provide examples of phrases supporting your classification.
3. Gauge Engagement: Evaluate the level of engagement based on the length, detail, and frequency of customer inquiries.
4. Predict Churn Risk: Identify language suggesting dissatisfaction or intent to leave.
5. Estimate CLV: Assign a qualitative score (High/Medium/Low) based on purchase intent, sentiment, engagement, and churn risk. Justify your reasoning.

Part 2: Identify Cross-Selling Opportunities
1. Understand Customer Needs: Extract any explicit or implicit mentions of problems, goals, or needs that could be addressed with additional products/services.
2. Recommend Complementary Products: Suggest items that align with the customer's expressed interests or current purchases (e.g., if they mention buying a phone, suggest a case or warranty).
3. Personalize Offers: Tailor your recommendations based on the customer's tone and context to ensure relevance.
4. Timing and Framing: Highlight when and how to introduce the cross-sell during the conversation (e.g., after solving their primary issue).

Output Format:
Findings:
• Purchase Intent: Summary of key phrases/themes
• Sentiment Analysis: Positive/Neutral/Negative with examples
• Engagement Level: High/Medium/Low with justification
• Churn Risk: High/Medium/Low with justification
• Estimated CLV: High/Medium/Low with explanation

Cross-Selling Opportunities:
• Customer Needs Identified: List of needs/problems mentioned
• Recommended Products/Services: Specific complementary items
• Timing and Framing Suggestions: How and when to introduce cross-sell

Use this structure to analyze each transcript and provide actionable insights."""
        }
    ]
)

summary = response.choices[0].message.content
print("Summary:", summary)
