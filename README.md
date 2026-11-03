 # Smart Customer Feedback Analyzer

An AI-powered customer feedback analysis tool designed to help small businesses understand customer opinions and improve their products and services.

The project uses Natural Language Processing (NLP) and Machine Learning to classify customer reviews into Positive, Negative, and Neutral sentiments, providing useful business insights through an interactive Streamlit dashboard.

## Problem Statement

Small businesses often receive multiple customer reviews but may not have enough time to manually analyze them and identify recurring customer opinions and concerns.

This project aims to simplify customer feedback analysis by automatically classifying reviews and presenting sentiment statistics to support better business decisions.

## Key Features

- **Sentiment Analysis:** Classifies customer reviews into Positive, Negative, and Neutral categories.
- **Text Preprocessing:** Cleans and prepares customer reviews for Machine Learning.
- **Machine Learning Models:** Implements and compares Logistic Regression and Linear SVC.
- **Uncertainty Detection:** Flags predictions with confidence below 55% for human review.
- **Business Insights:** Displays customer sentiment statistics.
- **Department Analysis:** Analyzes customer feedback across different product departments.
- **Interactive Dashboard:** Provides a user-friendly interface built with Streamlit.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLP (Natural Language Processing)
- TF-IDF Vectorization
- Logistic Regression
- Linear SVC
- Streamlit
- Joblib

## Dataset

The project uses the Women's Clothing E-Commerce Reviews dataset, containing 23,486 customer reviews.

After removing reviews with missing text, 22,641 reviews were used for analysis and model development.

Customer ratings were converted into sentiment labels:

| Rating | Sentiment |
|---|---|
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

Dataset: [Women's Clothing E-Commerce Reviews](https://huggingface.co/datasets/Censius-AI/ECommerce-Women-Clothing-Reviews)

## Machine Learning Workflow

1. **Data Loading:** Imported customer reviews using Pandas.
2. **Data Cleaning:** Removed missing review texts and unnecessary columns.
3. **Sentiment Labeling:** Created sentiment categories based on customer ratings.
4. **Exploratory Data Analysis:** Examined sentiment distribution and customer feedback.
5. **Text Preprocessing:** Converted text to lowercase and removed punctuation and unnecessary characters.
6. **Data Splitting:** Divided the dataset into 80% training and 20% testing data.
7. **Feature Extraction:** Applied TF-IDF with unigrams and bigrams.
8. **Model Training:** Trained Logistic Regression and Linear SVC classifiers.
9. **Model Evaluation:** Compared model performance using Accuracy, Precision, Recall, and F1-score.
10. **Deployment:** Built an interactive dashboard using Streamlit.

## Model Performance

Two Machine Learning algorithms were evaluated:

| Metric | Logistic Regression | Linear SVC |
|---|---:|---:|
| Accuracy | 77.15% | 79.00% |
| Macro F1-score | 0.60 | 0.58 |
| Negative F1-score | 0.53 | 0.51 |
| Neutral F1-score | 0.38 | 0.34 |
| Positive F1-score | 0.89 | 0.90 |

Logistic Regression was selected for the application because it achieved a higher Macro F1-score and performed slightly better on the Negative and Neutral classes.

## Dashboard

The Streamlit application allows users to:

- Enter a customer review.
- View its predicted sentiment.
- Check the model's prediction confidence.
- Identify uncertain predictions that require human review.
- Explore overall customer sentiment statistics.
- Compare sentiment distribution across product departments.

## Project Structure

```text
Smart-Customer-Feedback-Analyzer/
│
├── app.py
├── Smart_Customer_Feedback_Analyzer.ipynb
├── sentiment_model.joblib
├── tfidf_vectorizer.joblib
├── sentiment_summary.csv
├── department_summary.csv
├── requirements.txt
└── README.md
```

## Installation and Setup

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Navigate to the project directory and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit dashboard:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

## Future Improvements

- Support additional business categories and industries.
- Improve sentiment classification, especially for Neutral reviews.
- Support Arabic customer reviews.
- Analyze recurring customer complaints using topic analysis.
- Allow businesses to upload their own customer review datasets.
- Improve dashboard visualizations and reporting.

## Project Goal

This project demonstrates how Natural Language Processing and Machine Learning can transform customer feedback into meaningful business insights and help small businesses better understand their customers.

**Developed as a practical Machine Learning and NLP project using Python.**
