import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    return text

# Load trained model
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)

# Streamlit app
def main():
    st.title("BBC News Classifier")
    st.subheader("Enter your news article below:")

    # Text input for news article
    article_text = st.text_area("Input Text", "")

    if st.button("Classify"):
        # Preprocess text
        processed_text = preprocess_text(article_text)
        # Vectorize text
        text_vectorized = cv.transform([processed_text])
        # Predict category
        prediction = classifier.predict(text_vectorized)
        # Map prediction to category
        categories = {0: "Business News", 1: "Tech News", 2: "Politics News", 3: "Sports News", 4: "Entertainment News"}
        predicted_category = categories.get(prediction[0], "Unknown")
        # Display result
        st.write("Predicted Category:", predicted_category)

if __name__ == "__main__":
    main()
