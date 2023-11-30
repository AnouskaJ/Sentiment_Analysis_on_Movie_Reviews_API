import pickle
from flask import Flask, request, jsonify, render_template
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

model = pickle.load(open('best_model.sav', 'rb'))


app = Flask(__name__)

@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    text = data['text']
    text = text.lower()
    text = re.sub("<br />", "", text)
    text = re.sub(r'https\S+|www\S+|http\S+', "", text, flags=re.MULTILINE)
    text = re.sub(r"\@w+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text_tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_text = [w for w in text_tokens if not w in stop_words]
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in filtered_text]
    text = " ".join(text)

    vect = TfidfVectorizer()
    X = vect.fit_transform([text])

    y_pred = model.predict(X)
    sentiment = 'positive' if y_pred[0] == 1 else 'negative'

    return jsonify({'sentiment': sentiment})

@app.route('/')
def home():
    return render_template('index.html')

from textblob import TextBlob

@app.route('/predict', methods=['POST'])
def get_sentiment():
    text = request.form['text']
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        sentiment = "Positive"
    elif sentiment < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return render_template('result.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)