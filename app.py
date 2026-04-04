import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Example training data
X = ["I love this movie", "This movie is great", "Amazing film",
     "The film is bad", "Horrible movie", "Not good", "I hated this film", "Fantastic movie", "Could be better"]
y = [1, 1, 1, 0, 0, 0, 0, 1, 0]  # 1 = positive, 0 = negative

# Vectorize text
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vect, y)

# Streamlit app
st.title("NLP Sentiment Analysis")

review = st.text_area("Enter your Review")

if st.button("Predict"):
    if review.strip() != "":
        prediction = "positive" if model.predict(vectorizer.transform([review]))[0] else "negative"
        st.write("Prediction:", prediction)
    else:
        st.write("Please enter a review to predict.")
