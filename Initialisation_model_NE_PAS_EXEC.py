import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from joblib import dump
from sklearn.metrics import classification_report

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
    stop_words = set(stopwords.words('english'))  # Suppression des mots vides (stopwords)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatisation
    cleaned_text = ' '.join(tokens)  # Reconstitution du texte
    return cleaned_text



# Charger les données
data = pd.read_csv("sentimentdata_corrige.csv")

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    data['Text'], 
    data['Sentiment_Category'], 
    test_size=0.2, 
    random_state=42, 
    stratify=data['Sentiment_Category']
)

# Convertir le texte en vecteurs TF-IDF
tfidf = TfidfVectorizer(
    ngram_range=(1, 1),
    min_df=1,
    max_df=0.95,
    max_features=5000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Enregistrement du Vectorizer
dump(tfidf, 'modeles/vectorizer.joblib')

# Liste des modèles pour les sentiments
models = {
    "SVM": LinearSVC(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=21),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200, class_weight='balanced', random_state=42),
    "Naive Bayes": MultinomialNB(),
    "KNN": KNeighborsClassifier(),
    "Neural Network (MLP)": MLPClassifier(max_iter=300, random_state=42)
}

# Entraîner et évaluer les modèles de sentiment
for model_name, model in models.items():
    print(f"--- {model_name} ---")
    
    # Entraîner le modèle
    model.fit(X_train_tfidf, y_train)
    
    # Enregistrer le modèle dans un fichier .joblib
    model_filename = 'modeles/' + model_name + '_model.joblib'
    dump(model, model_filename)
    
    # Prédictions
    y_pred = model.predict(X_test_tfidf)
    
    # Afficher le rapport de classification
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Validation croisée
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean CV Score: {np.mean(cv_scores)}\n")


