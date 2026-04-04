import pandas as pd
import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Correct features and labels
X = ["I love this movie", "The film is bad"]  # replace with actual data if you have
y = [1, 0]  # 1 = positive, 0 = negative

# Vectorize text
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vect, y)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("NLP Sentiment Analysis")

text = st.text_area("Enter your Review")

if st.button("Predict"):
    review = [text]  # get user input
    result = model.predict(vectorizer.transform(review))[0]  # fix undefined variable
    st.write("Prediction:", "positive" if result else "negative")
