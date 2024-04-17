import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Download resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
@st.cache
def load_data():
    dataset = pd.read_csv("")https://gist.githubusercontent.com/ArbilShofiyurrahman/6d5902192d1134bd178b0434b43a7204/raw/8a166bb36b3fd8fb578a1ae3188ccd4133b92258/BBC%2520News%2520Train.csv
    return dataset

# Preprocessing functions
def remove_tags(text):
    remove = re.compile(r'<.*?>')
    return re.sub(remove, '', text)

def special_char(text):
    reviews = ''
    for x in text:
        if x.isalnum():
            reviews = reviews + x
        else:
            reviews = reviews + ' '
    return reviews

def convert_lower(text):
    return text.lower()

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    return ' '.join([x for x in words if x not in stop_words])

def lemmatize_word(text):
    wordnet = WordNetLemmatizer()
    return " ".join([wordnet.lemmatize(word) for word in text.split()])

def preprocess_text(text):
    text = remove_tags(text)
    text = special_char(text)
    text = convert_lower(text)
    text = remove_stopwords(text)
    text = lemmatize_word(text)
    return text

# Model training and prediction
@st.cache
def train_model():
    dataset = load_data()
    dataset['Text'] = dataset['Text'].apply(preprocess_text)

    x = dataset['Text']
    y = dataset['CategoryId']

    cv = CountVectorizer(max_features=5000)
    x = cv.fit_transform(x).toarray()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    return classifier, cv, accuracy

# Streamlit app
def main():
    st.title("BBC News Classification App")
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Home", "Prediction"])

    if page == "Home":
        st.write("## Welcome to BBC News Classification App")
        st.write("This app classifies BBC news articles into different categories such as Business, Tech, Politics, Sports, and Entertainment.")

    elif page == "Prediction":
        st.subheader("Prediction")
        text_input = st.text_area("Enter a news article:", "Hour ago, I contemplated retirement for a lot of reasons. I felt like people were not sensitive enough to my injuries. I felt like a lot of people were backed, why not me? I have done no less. I have won a lot of games for the team, and I am not feeling backed, said Ashwin")
        if st.button("Predict"):
            classifier, cv, accuracy = train_model()
            text_input = preprocess_text(text_input)
            text_input = cv.transform([text_input])
            prediction = classifier.predict(text_input)
            categories = {0: "Business", 1: "Tech", 2: "Politics", 3: "Sports", 4: "Entertainment"}
            result = categories[prediction[0]]
            st.write("Predicted Category:", result)

if __name__ == "__main__":
    main()
