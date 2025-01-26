import streamlit as st
import joblib
import os
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix  # Import csr_matrix

import nltk
nltk.data.path.append(r'C:\nltk_data')  # Or a folder where you want to store NLTK resources
nltk.download('vader_lexicon')

# Check if model files exist
print("Model file exists:", os.path.exists("logistic_regression_model.pkl"))
print("Vectorizer file exists:", os.path.exists("tfidf_vectorizer.pkl"))

# Load the model and vectorizer
model_path = "logistic_regression_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# Load resources
model, vectorizer = load_model_and_vectorizer()

# Instantiate SentimentIntensityAnalyzer at the start
analyzer = SentimentIntensityAnalyzer()

# App title
st.title("Sentiment Analysis App")

sentiment = "Not Analyzed Yet"  

# User input
user_input = st.text_input("Enter a review or text for sentiment analysis:", "")

# Time input with "No Time" option
no_time = st.checkbox("No Time")
if no_time:
    hour = None
    minute = None
    am_pm = None
else:
    # Separate inputs for hours, minutes, and AM/PM
    col1, col2, col3 = st.columns(3)
    with col1:
        hour = st.number_input("Hour", min_value=1, max_value=12, value=12)
    with col2:
        minute = st.number_input("Minute", min_value=0, max_value=59, value=0)
    with col3:
        am_pm = st.selectbox("AM/PM", ["AM", "PM"])

    # Convert to 24-hour format
    if am_pm == "PM" and hour != 12:
        hour += 12
    elif am_pm == "AM" and hour == 12:
        hour = 0

# Rating slider with "No Rating" option
no_rating = st.checkbox("No Rating")
if no_rating:
    rating_input = None
else:
    rating_input = st.slider("Rate the review from 0 to 5:", 0, 5, 3)

# Predict sentiment
if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Preprocess input
        input_vectorized = vectorizer.transform([user_input])

        # Create a sparse matrix for the numeric features
        if rating_input is not None and not no_time:
            numeric_features = np.array([[rating_input, hour, minute]])
        elif rating_input is not None:
            numeric_features = np.array([[rating_input]])
        elif not no_time:
            numeric_features = np.array([[hour, minute]])
        else:
            numeric_features = np.array([[]])
        numeric_features_sparse = csr_matrix(numeric_features)

        # Combine TF-IDF features and numeric features
        input_combined = hstack([input_vectorized, numeric_features_sparse])

        # Adjust the input features to match the model's expected number of features
        expected_features = model.n_features_in_
        if input_combined.shape[1] > expected_features:
            input_combined = input_combined[:, :expected_features]
        elif input_combined.shape[1] < expected_features:
            padding = csr_matrix((input_combined.shape[0], expected_features - input_combined.shape[1]))
            input_combined = hstack([input_combined, padding])

        # Predict sentiment
        prediction = model.predict(input_combined)[0]
        sentiment_map = {2: "Negative", 0: "Neutral", 1: "Positive"}
        sentiment = sentiment_map.get(prediction, "Unknown")
        
        # Display result
        st.write(f"*Predicted Sentiment:* {sentiment}")
        if not no_time:
            st.write(f"*Time Selected:* {hour % 12 or 12}:{minute:02d} {am_pm}")
        else:
            st.write("*Time Selected:* No Time")
        if rating_input is not None:
            st.write(f"*Rating Provided:* {rating_input}/5")
        else:
            st.write("*Rating Provided:* No Rating")
    else:
        st.write("Please enter some text for analysis.")

with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        scores = analyzer.polarity_scores(text)
        st.write('Polarity: ', round(scores['compound'], 2))
        st.write('Positive: ', round(scores['pos'], 2))
        st.write('Neutral: ', round(scores['neu'], 2))
        st.write('Negative: ', round(scores['neg'], 2))
        
         # Text cleaning
        pre = st.text_input('Clean Text: ')
        if pre:
            cleaned_text = re.sub(r'\s+', ' ', pre)  # Remove extra spaces
            cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)  # Remove punctuation
            cleaned_text = re.sub(r'\d+', '', cleaned_text)  # Remove numbers
            cleaned_text = cleaned_text.lower()  # Convert to lowercase
            st.write(cleaned_text)
            
with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file', type=['csv', 'xlsx'])

    def score(x):
        if isinstance(x, str):
            scores = analyzer.polarity_scores(x)
            return scores['compound']
        return 0  # Return neutral score for non-string entries

    def analyze(x):
        if x >= 0.5:
            return 'Positive'
        elif x <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    if upl:
        try:
            if upl.name.endswith('.csv'):
                df = pd.read_csv(upl)
            elif upl.name.endswith('.xlsx'):
                df = pd.read_excel(upl, engine='openpyxl')
            else:
                st.error("Unsupported file format")
                df = None

            if df is not None:
                if 'Unnamed: 0' in df.columns:
                    del df['Unnamed: 0']
                if 'content' in df.columns:  # Change 'text' to 'content'
                    df['score'] = df['content'].apply(score) 
                    df['analysis'] = df['score'].apply(analyze)
                    st.write(df.head(10))

                    @st.cache
                    def convert_df(df):
                        # Cache the conversion to prevent computation on every rerun
                        return df.to_csv().encode('utf-8')

                    csv = convert_df(df)

                    st.download_button(
                        label="Download data as CSV",
                        data=csv,
                        file_name='sentiment.csv',
                        mime='text/csv',
                    )
                else:
                    st.error("The uploaded file does not contain a 'content' column.") 
        except Exception as e:
            st.error(f"An error occurred: {e}")

        

    
