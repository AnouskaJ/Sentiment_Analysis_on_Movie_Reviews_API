# Movie Sentiment Analysis

This sentiment analysis model processes movie reviews from the IMDB dataset. It utilizes various algorithms, including Logistic Regression, Naive Bayes, and Support Vector Classifier (SVC) to predict sentiments as positive or negative. The model integrates data cleaning and visualization techniques such as removing stopwords, stemming, generating a word cloud, and displaying a bar chart highlighting the most common words in positive reviews.

## Algorithms Used
- Logistic Regression
- Naive Bayes
- Support Vector Classifier (SVC)

### Backend: Simple Flask API for Sentiment Analysis
The backend of this project involves a Flask API that performs Sentiment Analysis on textual data. It utilizes a pre-trained Support Vector Machine (SVM) model. The API accepts text inputs, processes them through various natural language processing techniques, and predicts the sentiment (positive or negative) of the text.

### Frontend: React Application
The frontend of this project is built using React. It provides a user-friendly interface for users to input movie reviews. These reviews are sent to the Flask API via HTTP requests, and the predicted sentiment is displayed back to the user in real-time.

## Project Structure

### Backend (Flask API)
- **app.py:** Contains the Flask API code for sentiment analysis.
- **best_model.sav:** Pre-trained Support Vector Machine (SVM) model used for sentiment prediction.

### Frontend (React Application)
- **App.js:** Main React component handling the UI and interaction logic.
- **App.css:** Styling for the React components.

## Screenshots

![Sample Screenshot](https://github.com/AnouskaJ/Sentiment_Analysis_on_Movie_Reviews_API/assets/82711261/b076deb3-1519-4cc4-a027-ffc032bd3edf)

## Getting Started
To run this project locally:
1. **Backend Setup:** Navigate to the Flask API directory and install the necessary dependencies using `pip install -r requirements.txt`. Then, start the Flask server.
2. **Frontend Setup:** Enter the React application directory, install the required dependencies with `npm install`, and launch the React app using `npm start`.
