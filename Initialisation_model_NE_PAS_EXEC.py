import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


data = pd.read_csv("sentimentdata_corrige.csv")  # Charger les données de référence

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Télécharger les ressources NLTK nécessaires
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Initialiser le lemmatiseur
lemmatizer = WordNetLemmatizer()

# Fonction de prétraitement du texte
def preprocess_text(text):
    # Conversion en minuscule
    text = text.lower()
    
    # Suppression des signes de ponctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenisation
    tokens = word_tokenize(text)
    
    # Suppression des mots vides (stopwords)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatisation
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Reconstitution du texte
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
from joblib import dump

# Étape 1: Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['Sentiment_Category'], 
                                                    test_size=0.2, random_state=42, stratify=data['Sentiment_Category'])

# Étape 2: Convertir le texte en vecteurs TF-IDF
tfidf = TfidfVectorizer(
    ngram_range=(1,1),
    min_df=1,
    max_df= 0.95,
    max_features=5000
)

X_train_tfidf = tfidf.fit_transform(X_train)

#Enregistrement du Vectorizer
joblib.dump(tfidf, 'modeles/vectorizer.joblib')

X_test_tfidf = tfidf.transform(X_test)

# Étape 3: Rééchantillonnage avec SMOTE pour équilibrer les classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Étape 4: Entraîner un modèle SVM
model = LinearSVC(class_weight='balanced')
model.fit(X_train_resampled, y_train_resampled)

# Étape 5: Prédictions et rapport de classification
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred, zero_division=0))

cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores)}\n")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from joblib import dump
import numpy as np


# Liste des modèles à tester
models = {
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=21),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200, class_weight='balanced', random_state=42),
    "Naive Bayes": MultinomialNB(),
    "KNN": KNeighborsClassifier(),
    "Neural Network (MLP)": MLPClassifier(max_iter=300, random_state=42)
}

# Étape 4: Tester les modèles
for model_name, model in models.items():
    print(f"--- {model_name} ---")
    
    # Entraîner le modèle
    model.fit(X_train_resampled, y_train_resampled)

    # Enregistrer le modèle dans un fichier .joblib
    model_filename ='modeles/' + model_name + '_model.joblib'
    dump(model, model_filename)
    
    # Prédictions
    y_pred = model.predict(X_test_tfidf)
    
    # Afficher le rapport de classification
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Validation croisée pour obtenir une idée générale de la performanceF
    cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5)
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean CV Score: {np.mean(cv_scores)}\n")


# Charger le modèle pré-entraîné
model = joblib.load("modeles/Random Forest_model.joblib")
tfidf = joblib.load("modeles/vectorizer.joblib")

# Interface utilisateur
st.title("Analyse de Sentiments")
st.write("Entrez votre message et nous vous dirons s'il est positif, neutre ou négatif")

# Saisie de l'utilisateur
user_input = st.text_area("Message à analyser")

# Prédire le sentiment
if st.button("Analyser"):
    if user_input:

        # Prétraitement du texte utilisateur
        processed_input = preprocess_text(user_input)

        user_input_tfidf = tfidf.transform([processed_input])

        # Prédiction
        prediction = model.predict(user_input_tfidf)[0]

        # Afficher le résultat
        sentiment_map = {0: "négatif", 1: "neutre", 2: "positif"}
        st.write(f"Le sentiment du message est : **{sentiment_map[prediction]}**")

        # Suggestions de messages similaires
        st.write("Messages similaires :")
        # Calculer les similarités (exemple basique)
        data_tfidf = tfidf.transform(data['Text'])
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(user_input_tfidf, data_tfidf).flatten()
        similar_indices = similarities.argsort()[-5:][::-1]  # Top 5 messages similaires
        similar_messages = data.iloc[similar_indices]
        for i, row in similar_messages.iterrows():
            st.write(f"- {row['Text']} (Sentiment: {sentiment_map[row['Sentiment_encode']]})")

        # Insights par plateforme
        st.write("Statistiques par plateforme :")
        platform_stats = data.groupby('Platform')['Sentiment_encode'].value_counts(normalize=True).unstack()
        st.bar_chart(platform_stats)

    else:
        st.write("Veuillez entrer un message.")
