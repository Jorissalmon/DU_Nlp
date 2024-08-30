---
title: "Projet DU Sorbonne Analytics"
author: "Nolwenn, Zakaria & [Votre Nom]"
output: github_document
---

# Projet DU Sorbonne Analytics

Ce projet fait partie du cursus du **DU Sorbonne Analytics** et a été réalisé en collaboration avec **Nolwenn** et **Zakaria**.

## Description du projet

L'objectif de ce projet est de développer un modèle d'analyse de sentiment à partir de données textuelles, permettant de prédire le sentiment exprimé dans un texte (positif, négatif ou neutre). Le modèle utilise diverses techniques de traitement du langage naturel (NLP) telles que la vectorisation TF-IDF, le suréchantillonnage SMOTE pour l'équilibrage des classes, et plusieurs algorithmes de machine learning, notamment le Random Forest, SVM, et le Réseau de Neurones (MLP).

## Structure du projet

Le projet est organisé en plusieurs fichiers et dossiers, chacun ayant un rôle spécifique dans la construction et l'évaluation du modèle de machine learning.

- **data/** : Ce dossier contient les jeux de données utilisés pour l'entraînement et les tests du modèle.
- **notebooks/** : Contient les notebooks Jupyter utilisés pour l'exploration des données et le développement initial des modèles.
- **modeles/** : Ce dossier contient les modèles entraînés et les objets sérialisés (e.g., vectoriseur TF-IDF, modèles joblib).
- **Initialisation_model_NE_PAS_EXEC.py** : Script Python principal pour l'entraînement des modèles et l'évaluation des performances.
- **app.py** : Application Streamlit permettant d'effectuer une analyse de sentiment sur des données texte et de visualiser les résultats.
- **requirements.txt** : Fichier listant les dépendances Python nécessaires pour exécuter le projet.
  
## Installation

Pour exécuter ce projet localement, suivez les étapes suivantes :  
git clone https://github.com/votre-utilisateur/votre-projet.git  
cd votre-projet  

### Créer et activer un environnement virtuel :  
python -m venv myenv  
source myenv/bin/activate  # Pour Linux/Mac  
myenv\Scripts\activate  # Pour Windows  

### Installer les dépendances :  
pip install -r requirements.txt  

### Télécharger les ressources NLTK nécessaires :  
nltk.download('stopwords')  
nltk.download('punkt')  
nltk.download('wordnet')  

## Exécution  
### 1. Entraînement des modèles  
Pour entraîner les modèles de machine learning sur vos données textuelles, exécutez le script suivant :  
python Initialisation_model_NE_PAS_EXEC.py  
Ce script entraîne plusieurs modèles, sauvegarde les meilleurs modèles sous forme de fichiers .joblib et affiche les performances de chaque modèle.  

### 2. Lancer l'application Streamlit  
Vous pouvez utiliser l'application Streamlit pour effectuer des analyses de sentiment sur des fichiers de données ou des textes personnalisés. Pour lancer l'application :  
streamlit run app.py  

### 3. Utilisation de l'application  
Analyse de fichier Excel : Vous pouvez charger un fichier Excel contenant des données textuelles à analyser. L'application vous permet de filtrer les données par hashtag et par plateforme, puis de visualiser l'évolution des mentions.  
Analyse de texte personnalisé : Vous pouvez entrer un texte personnalisé pour analyser son sentiment à l'aide du modèle entraîné.  

Les résultats de l'analyse sont sauvegardés dans un fichier Excel téléchargeable via l'application Streamlit. Le rapport de classification et les scores de validation croisée sont affichés pour chaque modèle testé.  

-Collaborateurs :-  
Nolwenn  
Zakaria  
Joris  
