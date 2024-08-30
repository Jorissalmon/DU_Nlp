import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from joblib import load
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    stop_words = set(stopwords.words('english'))  # Stop words
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    cleaned_text = ' '.join(tokens)  # Reassemble text
    return cleaned_text

# Load the model and TF-IDF vectorizer
model_rf = load('modeles/Random Forest_model.joblib')
tfidf = load('modeles/vectorizer.joblib')

# Sentiment prediction function
def predict_sentiment(text, model):
    processed_text = preprocess_text(text)
    text_tfidf = tfidf.transform([processed_text])
    prediction = model.predict(text_tfidf)[0]
    return prediction

# Function to clean and extract hashtags
def extract_hashtags(text):
    hashtags = re.findall(r'#\w+', str(text))
    return [tag.lower() for tag in hashtags]  # Clean hashtags

# Streamlit interface configuration
st.title("Sentiment and Hashtag Analysis by Platform")

# Upload Excel file
st.header("Upload Your Excel File")
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    # Read the Excel file
    data = pd.read_excel(uploaded_file)

    # Convert 'Timestamp' column to datetime format
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')

    # Extract all unique hashtags
    all_hashtags = set()

    for hashtags in data['Hashtags'].dropna():
        cleaned_hashtags = extract_hashtags(hashtags)  # Clean and extract hashtags
        all_hashtags.update(cleaned_hashtags)

    # Select hashtags and platforms
    st.subheader("Select Hashtags and Platforms")
    selected_hashtag = st.selectbox("Choose a hashtag to analyze", sorted(list(all_hashtags)))
    platforms = st.multiselect("Choose platforms", data['Platform'].unique())

    # Filter data based on selected hashtags and platforms
    if selected_hashtag and platforms:
        filtered_data = data[data['Platform'].isin(platforms)]

        # Filter rows containing the selected hashtag
        filtered_data = filtered_data[filtered_data['Hashtags'].apply(
            lambda x: selected_hashtag in [hashtag.lower() for hashtag in extract_hashtags(x)] if pd.notna(x) else False
        )]

        # Extract Year and Month from 'Timestamp'
        filtered_data['Year'] = filtered_data['Timestamp'].dt.year
        filtered_data['Month'] = filtered_data['Timestamp'].dt.month

        # Group by month, year, and platform to create an evolving graph
        filtered_data_grouped = filtered_data.groupby(['Platform', 'Year', 'Month']).size().reset_index(name='Counts')

        # Create a combined column for graph display
        filtered_data_grouped['Month_Year'] = filtered_data_grouped['Year'].astype(str) + '-' + filtered_data_grouped['Month'].astype(str).str.zfill(2)

        # Display a sample of the data
        st.subheader("Filtered Data (5 rows)")
        st.write(filtered_data.head())

        # Add a selector for choosing the type of graph
        chart_type = st.selectbox(
            "Choose the type of graph",
            ["Single Curve", "Curves by Platform"]
        )

        # Create a new figure
        plt.subplots(figsize=(12, 6))

        if chart_type == "Single Curve":
            # Group data by month and year, combining counts for all platforms
            single_curve_data = filtered_data_grouped.groupby(['Month_Year']).agg({'Counts': 'sum'}).reset_index()

            sns.lineplot(data=single_curve_data, x='Month_Year', y='Counts', marker='o')
            plt.title("Evolution of the Selected Hashtag Mentions (Single Curve)")
            plt.xlabel("Month-Year")
            plt.ylabel("Number of Mentions")

        elif chart_type == "Curves by Platform":
            sns.lineplot(data=filtered_data_grouped, x='Month_Year', y='Counts', hue='Platform', marker='o')
            plt.title("Evolution of the Selected Hashtag Mentions by Platform")
            plt.xlabel("Month-Year")
            plt.ylabel("Number of Mentions")

        # Format x-axis ticks to be readable
        plt.xticks(rotation=75)

        # Display the plot with Streamlit
        st.pyplot(plt.gcf())

        # Export results with predictions
        st.subheader("Download Results")
        result_file = filtered_data.to_excel("result_with_predictions.xlsx", index=False)
        st.download_button(
            label="Download Excel file with predictions",
            data=open("result_with_predictions.xlsx", "rb").read(),
            file_name="result_with_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Text area for individual analysis
st.header("Sentiment Analysis for Custom Text")
user_input = st.text_area("Enter a message here", "")

if st.button("Analyze"):
    if user_input:
        sentiment = predict_sentiment(user_input, model_rf)
        st.write(f"The predicted sentiment is: **{sentiment}**")
    else:
        st.write("Please enter a message.")
