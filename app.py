import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Télécharger les ressources NLTK nécessaires
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialiser le lemmatiseur
lemmatizer = WordNetLemmatizer()

# Fonction de prétraitement du texte
def preprocess_text(text):
    text = text.lower()  # Conversion en minuscule
    text = re.sub(r'[^\w\s]', '', text)  # Suppression des signes de ponctuation
    tokens = word_tokenize(text)  # Tokenisation
    stop_words = set(stopwords.words('english'))  # Mots vides
    tokens = [word for word in tokens if word not in stop_words]  # Suppression des mots vides
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatisation
    cleaned_text = ' '.join(tokens)  # Reconstitution du texte
    return cleaned_text

# Charger le modèle et le vecteur TF-IDF
model = load('modeles/Random Forest_model.joblib')
tfidf = load('modeles/vectorizer.joblib')



# Prédiction sur un texte d'entrée
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    text_tfidf = tfidf.transform([processed_text])
    prediction = model.predict(text_tfidf)[0]
    return prediction

# Interface Streamlit
st.title("Analyse de Sentiments")
st.write("Entrez votre message et nous vous dirons s'il est positif, neutre ou négatif")

# Saisie de l'utilisateur
user_input = st.text_area("Message à analyser")

if st.button("Analyser"):
    if user_input:
        # Prédire le sentiment
        sentiment = predict_sentiment(user_input)
        
        # Afficher le résultat
        st.write(f"Le sentiment du message est : **{sentiment}**")
        
        # Suggestions de messages similaires
        st.write("Messages similaires :")
        data = pd.read_csv("sentimentdata_corrige.csv")
        data_tfidf = tfidf.transform(data['Text'])
        user_input_tfidf = tfidf.transform([preprocess_text(user_input)])
        similarities = cosine_similarity(user_input_tfidf, data_tfidf).flatten()
        similar_indices = similarities.argsort()[-5:][::-1]  # Top 5 messages similaires
        similar_messages = data.iloc[similar_indices]
        for i, row in similar_messages.iterrows():
            st.write(f"- {row['Text']} (Sentiment: {row['Sentiment_Category']})")

        # Insights par plateforme
        st.write("Statistiques par plateforme :")
        platform_stats = data.groupby('Platform')['Sentiment_Category'].value_counts(normalize=True).unstack()
        st.bar_chart(platform_stats)
        
    else:
        st.write("Veuillez entrer un message.")
