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
from wordcloud import WordCloud

# Télécharger les ressources nécessaires NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialiser le lemmatiseur
lemmatizer = WordNetLemmatizer()

# Fonction de prétraitement du texte
def preprocess_text(text):
    text = text.lower()  # Convertir en minuscules
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation
    tokens = word_tokenize(text)  # Tokenisation
    stop_words = set(stopwords.words('english'))  # Mots vides
    tokens = [word for word in tokens if word not in stop_words]  # Supprimer les mots vides
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatisation
    cleaned_text = ' '.join(tokens)  # Réassembler le texte
    return cleaned_text

# Charger le modèle et le vectoriseur TF-IDF
model_rf = load('modeles/Random Forest_model.joblib')
tfidf = load('modeles/vectorizer.joblib')

# Fonction de prédiction du sentiment
def predict_sentiment(text, model):
    processed_text = preprocess_text(text)
    text_tfidf = tfidf.transform([processed_text])
    prediction = model.predict(text_tfidf)[0]
    return prediction

# Fonction pour extraire les hashtags
def extract_hashtags(text):
    hashtags = re.findall(r'#\w+', str(text))
    return [tag.lower() for tag in hashtags]  # Nettoyer les hashtags

# Configuration de la charte graphique
st.set_page_config(page_title="Sentiment and Hashtag Analysis", layout="wide")
st.markdown(
    """
    <style>
    .main {background-color: #f5f5f5;}
    h1, h2, h3 {color: #333;}
    .stButton>button {background-color: #4CAF50; color: white;}
    </style>
    """,
    unsafe_allow_html=True
)

# Palette de couleurs uniforme
color_palette = ['#FF9999', '#66B2FF', '#99FF99']

# Titre principal
st.title("Sentiment and Hashtag Analysis by Platform")

# Charger le fichier Excel
st.header("Upload Your Excel File")
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

# Boîte de texte pour tester le modèle et bouton de téléchargement dans la barre latérale
st.sidebar.header("Test the Model")
input_text = st.sidebar.text_area("Enter a text to predict sentiment")

if st.sidebar.button("Analyze"):
    if input_text:
        prediction = predict_sentiment(input_text, model_rf)
        st.sidebar.write(f"Predicted Sentiment: **{prediction}**")
    else:
        st.sidebar.write("Please enter a text.")


if uploaded_file is not None:
    # Lire le fichier Excel
    data = pd.read_excel(uploaded_file)

    # Ajouter les prédictions au DataFrame
    data['Sentiment_Prediction'] = data['Text'].apply(lambda x: predict_sentiment(x, model_rf))

    # Ne conserver que les tweets avec des prédictions
    filtered_data_with_predictions = data.dropna(subset=['Sentiment_Prediction'])

    # Bouton pour télécharger les résultats
    st.sidebar.header("Download Results")
    st.sidebar.write("Download the Excel file containing the data with sentiment predictions.")
    output_file = "result_with_predictions.xlsx"
    if st.sidebar.button("Download Results"):
        filtered_data_with_predictions.to_excel(output_file, index=False)
        st.sidebar.download_button(
            label="Download Excel file with predictions",
            data=open(output_file, "rb").read(),
            file_name=output_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # Convertir la colonne 'Timestamp' au format datetime
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')

    # Extraire tous les hashtags uniques
    all_hashtags = set()

    for hashtags in data['Hashtags'].dropna():
        cleaned_hashtags = extract_hashtags(hashtags)  # Nettoyer et extraire les hashtags
        all_hashtags.update(cleaned_hashtags)

    # Ajouter l'option "Tous les hashtags"
    all_hashtags = ["Tous les hashtags"] + sorted(list(all_hashtags))

    # Sélection des plateformes
    st.header("Select Platforms")
    platforms = st.multiselect("Choose platforms", data['Platform'].unique())

    if platforms:
        # Filtrer les données en fonction des plateformes sélectionnées
        filtered_data = data[data['Platform'].isin(platforms)]

        # Extraire l'année et le mois de 'Timestamp'
        filtered_data['Year'] = filtered_data['Timestamp'].dt.year
        filtered_data['Month'] = filtered_data['Timestamp'].dt.month

        # Grouper par mois, année et plateforme pour créer un graphique évolutif
        filtered_data_grouped = filtered_data.groupby(['Platform', 'Year', 'Month']).size().reset_index(name='Counts')

        # Créer une colonne combinée pour l'affichage du graphique
        filtered_data_grouped['Month_Year'] = filtered_data_grouped['Year'].astype(str) + '-' + filtered_data_grouped['Month'].astype(str).str.zfill(2)

        # Afficher le graphique d'évolution
        st.header("Evolution of Mentions Over Time")
        chart_type = st.selectbox("Choose the type of graph", ["Single Curve", "Curves by Platform"])

        # Sélection des hashtags juste en dessous du sélecteur de courbe
        selected_hashtag = st.selectbox("Choose a hashtag to analyze", all_hashtags)

        if selected_hashtag != "Tous les hashtags":
            # Filtrer les lignes contenant le hashtag sélectionné
            filtered_data = filtered_data[filtered_data['Hashtags'].apply(
                lambda x: selected_hashtag in [hashtag.lower() for hashtag in extract_hashtags(x)] if pd.notna(x) else False
            )]

            # Recalculer les groupes pour le graphique d'évolution après la sélection du hashtag
            filtered_data_grouped = filtered_data.groupby(['Platform', 'Year', 'Month']).size().reset_index(name='Counts')
            filtered_data_grouped['Month_Year'] = filtered_data_grouped['Year'].astype(str) + '-' + filtered_data_grouped['Month'].astype(str).str.zfill(2)

        # Créer une nouvelle figure
        plt.subplots(figsize=(14, 4))

        if chart_type == "Single Curve":
            # Grouper les données par mois et année, en combinant les comptes pour toutes les plateformes
            single_curve_data = filtered_data_grouped.groupby(['Month_Year']).agg({'Counts': 'sum'}).reset_index()

            sns.lineplot(data=single_curve_data, x='Month_Year', y='Counts', marker='o', color=color_palette[0])
            plt.title("Evolution of the Selected Hashtag Mentions (Single Curve)")
            plt.xlabel("Month-Year")
            plt.ylabel("Number of Mentions")

        elif chart_type == "Curves by Platform":
            sns.lineplot(data=filtered_data_grouped, x='Month_Year', y='Counts', hue='Platform', palette=color_palette, marker='o')
            plt.title("Evolution of the Selected Hashtag Mentions by Platform")
            plt.xlabel("Month-Year")
            plt.ylabel("Number of Mentions")

        # Formater les ticks de l'axe x pour être lisibles
        plt.xticks(rotation=75)

        # Afficher le graphique avec Streamlit
        st.pyplot(plt.gcf())


        # Diagramme circulaire des sentiments et Nuages de mots
        st.sidebar.subheader("Sentiment Distribution")
        sentiment_counts = filtered_data_with_predictions['Sentiment_Prediction'].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=color_palette)
        plt.title("Sentiment Distribution")
        st.sidebar.pyplot(plt.gcf())

        # Diagramme circulaire des sentiments et Nuages de mots
        st.header("Sentiment Distribution and Word Clouds")
        # Word clouds sur une seule ligne
        st.subheader("Word Clouds")
        col1, col2, col3 = st.columns(3)

        positive_tweets = filtered_data_with_predictions[filtered_data_with_predictions['Sentiment_Prediction'] == 'positif']['Text'].values
        negative_tweets = filtered_data_with_predictions[filtered_data_with_predictions['Sentiment_Prediction'] == 'négatif']['Text'].values
        neutral_tweets = filtered_data_with_predictions[filtered_data_with_predictions['Sentiment_Prediction'] == 'neutre']['Text'].values

        with col1:
            if len(positive_tweets) > 0:
                positive_text = " ".join(positive_tweets)
                positive_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(preprocess_text(positive_text))
                st.subheader("Positive")
                plt.figure(figsize=(10, 5))
                plt.imshow(positive_wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt.gcf())

        with col2:
            if len(negative_tweets) > 0:
                negative_text = " ".join(negative_tweets)
                negative_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(preprocess_text(negative_text))
                st.subheader("Negative")
                plt.figure(figsize=(10, 5))
                plt.imshow(negative_wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt.gcf())

        with col3:
            if len(neutral_tweets) > 0:
                neutral_text = " ".join(neutral_tweets)
                neutral_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Greys').generate(preprocess_text(neutral_text))
                st.subheader("Neutral")
                plt.figure(figsize=(10, 5))
                plt.imshow(neutral_wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt.gcf())