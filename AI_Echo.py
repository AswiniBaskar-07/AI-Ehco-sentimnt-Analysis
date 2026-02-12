import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# PAGE SETTINGS

st.set_page_config(page_title="AI Echo Sentiment Analysis", layout="centered")


# LOAD MODEL

model = joblib.load("logistic_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# TEXT CLEANING FUNCTION

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)


# SIDEBAR MENU

menu = st.sidebar.selectbox(
    "Navigation",
    ["Prediction", "EDA Dashboard"]
)


# 🔮 PREDICTION PAGE

if menu == "Prediction":

    st.title("Sentiment Prediction")
    st.write("Enter a review below:")

    user_input = st.text_area("Review Text")

    if st.button("Predict"):

        if user_input.strip() == "":
            st.warning("Please enter review text")
        else:
            cleaned = clean_text(user_input)
            vector = tfidf.transform([cleaned])
            prediction = model.predict(vector)[0]

            # Color Based Output
            if prediction == "Positive":
                st.markdown(
                    f"<h2 style='color:green;'>🟢 {prediction}</h2>",
                    unsafe_allow_html=True
                )

            elif prediction == "Negative":
                st.markdown(
                    f"<h2 style='color:red;'>🔴 {prediction}</h2>",
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    f"<h2 style='color:orange;'>🟡 {prediction}</h2>",
                    unsafe_allow_html=True
                )

# 📊 EDA DASHBOARD

elif menu == "EDA Dashboard":

    st.title("Exploratory Data Analysis")

    try:
        df = pd.read_csv(r"C:\Users\BaBuReDdI\Downloads\chatgpt_style_reviews_dataset.xlsx - Sheet1 (3).csv")
    except:
        st.error("Dataset not found in project folder.")
        st.stop()

    # Create Sentiment
    def label_sentiment(rating):
        if rating >= 4:
            return "Positive"
        elif rating == 3:
            return "Neutral"
        else:
            return "Negative"

    df["sentiment"] = df["rating"].apply(label_sentiment)

    # Sentiment Distribution
    st.subheader("Sentiment Distribution")

    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="sentiment", ax=ax1)
    st.pyplot(fig1)

    # Rating Distribution
    st.subheader("Rating Distribution")

    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x="rating", ax=ax2)
    st.pyplot(fig2)

    # Platform Analysis
    st.subheader("Average Rating by Platform")

    fig3, ax3 = plt.subplots()
    df.groupby("platform")["rating"].mean().plot(kind="bar", ax=ax3)
    st.pyplot(fig3)

    # Verified Purchase Analysis
    st.subheader("Verified vs Non-Verified Average Rating")

    fig4, ax4 = plt.subplots()
    df.groupby("verified_purchase")["rating"].mean().plot(kind="bar", ax=ax4)
    st.pyplot(fig4)
