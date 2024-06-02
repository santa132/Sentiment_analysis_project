import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Define the path to your model and other components
MODEL_PATH = "Models/model_xgb.pkl"
SCALER_PATH = "Models/scaler.pkl"
VECTORIZER_PATH = "Models/countVectorizer.pkl"

# Load and customize stopwords
stop_words = set(stopwords.words("english"))
negations = {'not', 'no', 'nor', "didn't", "wasn't", "isn't", "aren't", "doesn't"}
custom_stopwords = stop_words - negations

# Function to load models and other resources
def load_resources():
    predictor = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    cv = pickle.load(open(VECTORIZER_PATH, "rb"))
    return predictor, scaler, cv

# Load resources on server start
predictor, scaler, cv = load_resources()

# Preprocess text using lemmatization
def preprocess_text(text_input):
    lemmatizer = WordNetLemmatizer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in custom_stopwords]
    return " ".join(review)

# Predict sentiment based on input text
def single_prediction(text_input):
    text_processed = preprocess_text(text_input)
    X_prediction = cv.transform([text_processed]).toarray()
    X_prediction_scaled = scaler.transform(X_prediction)
    y_prediction = predictor.predict_proba(X_prediction_scaled)
    threshold = 0.90
    positive_probability = y_prediction[0][1]  # assuming the second column is "Positive"
    if positive_probability > threshold:
        return "Positive"
    else:
        return "Negative"

# Streamlit UI
st.title('Alexa Reviews Sentiment Prediction project')
st.write('This application predicts the sentiment of the input text using XGBoost model.')

# Text input
user_input = st.text_area("Enter text here:", value="", height=None, max_chars=None, key=None)

if st.button('Predict Sentiment'):
    if user_input:
        # Call the prediction function
        
        prediction = single_prediction(user_input)
        st.write(f'Predicted Sentiment: {prediction}')
    else:
        st.write('Please enter some text to analyze.')

