import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained model and tokenizer
model_name = "bert-base-uncased"  # Or "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


# Define the sentiment prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = torch.argmax(probs)
    sentiment = "positive" if label == 1 else "negative"
    return {"sentiment": sentiment, "confidence": probs[0][label].item()}


# Create the Streamlit app
def main():
  st.title("Sentiment Analysis")
  st.write("Enter some text to analyze its sentiment.")

  text_input = st.text_area("Enter text here:", "")

  if st.button("Analyze"):
      if text_input:
          result = predict_sentiment(text_input)
          st.write(f"Sentiment: {result['sentiment']}")
          st.write(f"Confidence: {result['confidence']:.2f}")
      else:
          st.write("Please enter some text.")

if __name__ == "__main__":
    main()
