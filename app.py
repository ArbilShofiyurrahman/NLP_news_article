import streamlit as st
import pandas as pd
from sklearn.externals import joblib

# Load trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("random_forest_model.joblib")

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
        text_input = st.text_area("Enter a news article (leave blank for default text):", "Hour ago, I contemplated retirement for a lot of reasons. I felt like people were not sensitive enough to my injuries. I felt like a lot of people were backed, why not me? I have done no less. I have won a lot of games for the team, and I am not feeling backed, said Ashwin")
        if st.button("Predict"):
            if text_input.strip():  # Cek apakah input tidak kosong
                model = load_model()
                text_input = preprocess_text(text_input)  # Lakukan pra-pemrosesan di sini jika perlu
                prediction = model.predict([text_input])
                categories = {0: "Business", 1: "Tech", 2: "Politics", 3: "Sports", 4: "Entertainment"}
                result = categories[prediction[0]]
                st.write("Predicted Category:", result)
            else:
                st.warning("Please enter a news article.")
            
if __name__ == "__main__":
    main()
