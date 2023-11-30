from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import Flask, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle
from textblob import TextBlob
nltk.download('stopwords')
nltk.download('punkt')
app = Flask(__name__)
CORS(app)
model = pickle.load(open('best_model.sav', 'rb'))

def preprocess_text(text):
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
    return text

@app.route('/api', methods=['POST'])
def predict_sentiment():
    data = request.get_json(force=True)

    preprocessed_text = preprocess_text(data['text'])

    blob = TextBlob(preprocessed_text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        sentiment_label = 'positive'
    elif sentiment < 0:
        sentiment_label = 'negative'
    else:
        sentiment_label = 'neutral'

    return jsonify({'sentiment': sentiment_label})

if __name__ == '__main__':
    app.run(debug=True)
