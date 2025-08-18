# Projet : Prédiction des fluctuations du marché des cryptomonnaies

## Identité
**Auteur :** Léo Palich  
**Contexte :** Projet personnel d’expérimentation en data science et machine learning appliqué à la finance.  

## Objectif
L’objectif de ce projet était de concevoir un modèle de machine learning capable de **prédire les fluctuations du marché des cryptomonnaies**.  
Plus précisément :  
- Utiliser des données historiques de prix pour anticiper les hausses et baisses.  
- Évaluer si des modèles classiques pouvaient capturer la dynamique très volatile des marchés crypto.  
- Explorer la faisabilité d’un outil d’aide à la décision basé sur l’IA pour le trading.  

# Description technique

## Technologies utilisées
- **Python** (Jupyter Notebooks)  
- **Bibliothèques principales :**
  - [Pandas](https://pandas.pydata.org/) pour la manipulation des séries temporelles  
  - [NumPy](https://numpy.org/) pour le traitement numérique  
  - [Scikit-learn](https://scikit-learn.org/) pour la modélisation machine learning  
  - Visualisation via [Matplotlib](https://matplotlib.org/)  

## Fonctionnement général
1. **Collecte et préparation des données** (`datas.ipynb`)  
   - Intégration de données de prix (open, close, high, low, volume).  
   - Nettoyage et mise en forme pour constituer un dataset exploitable.  

2. **Conception et entraînement du modèle** (`training.ipynb`)  
   - Création d’un pipeline de prédiction basé sur des modèles de régression/classification.  
   - Entraînement sur une partie des données, test sur la période restante.  

3. **Évaluation des performances**  
   - Utilisation de métriques classiques (erreur moyenne, précision sur tendance).  
   - Visualisation des prédictions comparées aux données réelles.  

# Résultats
- Le modèle n’a pas réussi à fournir des prédictions fiables sur la direction du marché.  
- Les fluctuations des cryptomonnaies se sont révélées trop volatiles pour les approches utilisées.  
- Le projet a cependant permis de :  
  - Approfondir mes compétences en **séries temporelles** et en **prétraitement de données financières**.  
  - Comprendre les limites des modèles classiques sur un marché aussi instable.  
  - Poser les bases pour tester des approches plus avancées (LSTM, Transformers, modèles probabilistes).  

# Perspectives
- Explorer des modèles spécialisés dans les séries temporelles non linéaires (RNN, LSTM, Transformers).  
- Intégrer des indicateurs techniques et des données contextuelles (tweets, actualités, volume social).  
- Comparer la robustesse de différentes stratégies de prédiction face à la forte volatilité des cryptomonnaies.  