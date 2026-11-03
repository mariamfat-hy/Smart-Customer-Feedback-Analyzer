 
import streamlit as st
import pandas as pd
import joblib
import re

st.set_page_config(
    page_title="Smart Customer Feedback Analyzer",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def load_models():
    model = joblib.load("sentiment_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer


@st.cache_data
def load_data():
    sentiment = pd.read_csv("sentiment_summary.csv")
    department = pd.read_csv("department_summary.csv")
    return sentiment, department


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


model, vectorizer = load_models()
sentiment_data, department_data = load_data()

st.title("Smart Customer Feedback Analyzer")
st.write("AI-powered customer feedback analysis for small businesses.")

st.divider()

st.header("Analyze Customer Review")

review = st.text_area(
    "Enter a customer review:",
    placeholder="Write the customer's feedback here..."
)

if st.button("Analyze Review"):
    if not review.strip():
        st.warning("Please enter a customer review.")
    else:
        cleaned_review = clean_text(review)

        if not cleaned_review:
            st.warning("Please enter a valid English review.")
        else:
            review_vector = vectorizer.transform([cleaned_review])
            probabilities = model.predict_proba(review_vector)[0]

            best_index = probabilities.argmax()
            prediction = model.classes_[best_index]
            confidence = probabilities[best_index]

            if confidence < 0.55:
                st.warning("Uncertain: This review needs human review.")
                st.write("Most likely sentiment:", prediction)
                st.write(f"Confidence: {confidence:.2%}")

            elif prediction == "Positive":
                st.success("Sentiment: Positive 😊")
                st.write(f"Confidence: {confidence:.2%}")

            elif prediction == "Negative":
                st.error("Sentiment: Negative ☹️")
                st.write(f"Confidence: {confidence:.2%}")

            else:
                st.info("Sentiment: Neutral 😐")
                st.write(f"Confidence: {confidence:.2%}")

st.divider()

st.header("Business Insights")

counts = sentiment_data.set_index("Sentiment")["Count"].to_dict()
total_reviews = sum(counts.values())

positive_count = counts.get("Positive", 0)
negative_count = counts.get("Negative", 0)
neutral_count = counts.get("Neutral", 0)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Reviews", f"{total_reviews:,}")
col2.metric("Positive Reviews", f"{positive_count:,}")
col3.metric("Negative Reviews", f"{negative_count:,}")
col4.metric("Neutral Reviews", f"{neutral_count:,}")

st.subheader("Customer Sentiment Distribution")

st.bar_chart(
    sentiment_data.set_index("Sentiment")
)

st.subheader("Customer Feedback by Department")

department_pivot = department_data.pivot_table(
    index="Department Name",
    columns="Sentiment",
    values="Count",
    fill_value=0
)

department_percentage = department_pivot.div(
    department_pivot.sum(axis=1),
    axis=0
) * 100

st.bar_chart(department_percentage)

st.caption(
    "Insights are based on the sample clothing reviews dataset. "
    "Predictions are estimates and may not always be correct."
)
