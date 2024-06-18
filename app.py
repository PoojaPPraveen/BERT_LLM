from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

app = Flask(__name__)

# Load pre-trained model and tokenizer from Hugging Face
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Function to preprocess and predict sentiment
def predict_sentiment(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs[0]
    probabilities = torch.softmax(logits, dim=1).numpy()[0]
    
    # Adjust this depending on your model's labels
    sentiment_score = np.argmax(probabilities)
    
    if sentiment_score == 4:
        sentiment = "positive"
    elif sentiment_score == 3:
        sentiment = "slightly positive"
    elif sentiment_score == 2:
        sentiment = "neutral"
    elif sentiment_score == 1:
        sentiment = "slightly negative"
    else:
        sentiment = "negative"
    
    return sentiment, probabilities

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    
    sentiment, probabilities = predict_sentiment(text)
    
    response = {
        'text': text,
        'sentiment': sentiment,
        'probabilities': probabilities.tolist()
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
