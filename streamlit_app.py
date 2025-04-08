import streamlit as st
import pickle
import sklearn

# Load the trained model and vectorizer
with open("fake_news_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("count_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# App Title
st.title("📰 Fake News Classifier")
st.markdown("This app uses a machine learning model to detect whether a news article is **Fake** or **Real**.")

# User Input
news_text = st.text_area("Enter the news article text:", height=200)

# Predict Button
if st.button("Classify"):
    if news_text.strip() == "":
        st.warning("Please enter some news content to classify.")
    else:
        # Vectorize the input and make prediction
        input_data = vectorizer.transform([news_text])
        prediction = model.predict(input_data)[0]

        if prediction == 0:
            st.success("✅ The news article is **Real**.")
        else:
            st.error("🚨 The news article is **Fake**.")

# Optional footer
st.markdown("---")

