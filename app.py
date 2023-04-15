# Import the necessary libraries
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

# Load the saved model
model = pickle.load(open('best_model.sav', 'rb'))

# Create a new Flask app
app = Flask(__name__)

# Define an endpoint for the API
@app.route('/api', methods=['POST'])
def predict():
    # Get the text data from the request
    data = request.get_json(force=True)
    
    # Preprocess the text data
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
    
    # Vectorize the preprocessed text data
    vect = TfidfVectorizer()
    X = vect.fit_transform([text])
    
    # Predict the sentiment of the text data using the trained model
    y_pred = model.predict(X)
    sentiment = 'positive' if y_pred[0] == 1 else 'negative'
    
    # Return the prediction result
    return jsonify({'sentiment': sentiment})

# Create a simple HTML form to accept user input
@app.route('/')
def home():
    return render_template('index.html')

# Define a function to handle the form submission and display the prediction result on the webpage
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


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
