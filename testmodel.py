import re
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download
from io import BytesIO

# Download necessary NLTK resources
download('wordnet')
download('omw-1.4')

# Load and customize stopwords
stop_words = set(stopwords.words("english"))
# Removing negations such as 'not', 'no', 'nor'
negations = {'not', 'no', 'nor', "didn't", "wasn't", "isn't", "aren't", "doesn't"}
custom_stopwords = stop_words - negations

# Function to load models and other resources
def load_resources():
    predictor = pickle.load(open("Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open("Models/scaler.pkl", "rb"))
    cv = pickle.load(open("Models/countVectorizer.pkl", "rb"))
    return predictor, scaler, cv

# Preprocess text using lemmatization
def preprocess_text(text_input):
    lemmatizer = WordNetLemmatizer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in custom_stopwords]
    return " ".join(review)

def single_prediction(predictor, scaler, cv, text_input):
    text_processed = preprocess_text(text_input)
    X_prediction = cv.transform([text_processed]).toarray()
    X_prediction_scaled = scaler.transform(X_prediction)
    y_prediction = predictor.predict_proba(X_prediction_scaled)

    print("Processed text:", text_processed)
    print("Scaled feature vector:", X_prediction_scaled)
    print("Prediction probabilities:", y_prediction)

    threshold = 0.88
    positive_probability = y_prediction[0][1]  # assuming the second column is "Positive"

    if positive_probability > threshold:
        return "Positive"
    else:
        return "Negative"

# Main function to handle user input
def main():
    predictor, scaler, cv = load_resources()

    while True:
        text_input = input("Enter the text for prediction (type 'exit' to quit): ")
        if text_input.lower() == 'exit':
            break
        prediction = single_prediction(predictor, scaler, cv, text_input)
        print("Predicted Sentiment:", prediction)

if __name__ == "__main__":
    main()
