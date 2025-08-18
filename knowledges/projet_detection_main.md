# Projet : Détection de gestes de la main pour contrôle robotique

## Identité
**Auteur :** Léo Palich (en collaboration avec un groupe de projet académique)  
**Contexte :** Projet universitaire de robotique sociale, utilisant mes acquis du projet personnel de détection de silhouettes pour les appliquer à la détection de gestes manuels précis et au contrôle d’un robot sous Webots.  

## Objectif
À l’origine, le projet devait porter sur le **contrôle du bras du robot**, mais notre groupe a choisi de se focaliser sur la **détection de gestes de la main**.  
Cette approche originale nous a permis de sortir du lot, et a eu un impact concret : un enseignant en situation de handicap a pu contrôler le robot uniquement grâce à ses mouvements de main.  

L’objectif était donc :  
- Construire un système capable de reconnaître en temps réel différents gestes de la main.  
- Associer ces gestes à des commandes robotisées pour interagir directement avec le robot dans Webots.  
- **Concevoir les gestes de manière intuitive**, car il fallait intégrer **10 gestes différents** et garantir leur utilisabilité immédiate par l’utilisateur.  


# Description technique

## Technologies utilisées
- **Langage :** Python  
- **Bibliothèques principales :**
  - [OpenCV](https://opencv.org/) pour la capture vidéo  
  - [Mediapipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) pour l’extraction des 21 points clés de la main  
  - [Scikit-learn](https://scikit-learn.org/) pour l’entraînement du modèle (Random Forest)  
  - [Joblib](https://joblib.readthedocs.io/) pour la sauvegarde des modèles  
  - [Pandas](https://pandas.pydata.org/) pour la gestion des datasets  
  - [Webots](https://cyberbotics.com/) pour la simulation robotique  

## Fonctionnement général
1. **Enregistrement des gestes** (`record_gestures.py`)  
   - Capture des coordonnées 3D (x, y, z) des 21 points de la main avec Mediapipe.  
   - Sauvegarde des données dans des fichiers CSV labellisés par geste.  

2. **Construction du dataset** (`datas.py`)  
   - Fusion automatique de tous les fichiers `gestures_*.csv` en un dataset unique `gestures_dataset.csv`.  

3. **Entraînement du modèle** (`train_model.py`)  
   - Utilisation d’un **Random Forest Classifier**.  
   - Évaluation via rapport de classification (précision, rappel, f1-score).  
   - Sauvegarde du modèle entraîné au format `.pkl`.  

4. **Reconnaissance en temps réel** (`model_test.py`)  
   - Détection de la main via webcam.  
   - Extraction des coordonnées et classification en temps réel.  
   - Affichage du geste reconnu + probabilité directement sur la vidéo.  

5. **Intégration robotique** (`main.py`)  
   - Interface en ligne de commande pour enregistrer, entraîner et tester les gestes.  
   - Couplage avec un script de contrôle robot sous Webots pour exécuter des actions en fonction des gestes reconnus (ex. avancer, reculer, tourner, saluer, etc.).  

# Résultats
- Reconnaissance fiable de **10 gestes différents** en temps réel.  
- Gestes pensés et optimisés pour être **intuitifs** et utilisables immédiatement.  
- Démonstration avec un robot Webots contrôlé par gestes.  
- **Notre groupe a réussi à faire en sorte que le robot atteigne la ligne d’arrivée rapidement et sans encombre, démontrant la robustesse de notre approche.**  
- Cette performance technique a pris encore plus de sens lorsqu’elle a permis à un enseignant en situation de handicap de contrôler le robot uniquement grâce à ses mains, illustrant concrètement l’impact humain et inclusif du projet.  

# Perspectives
- Déployer la solution sur un robot physique réel pour des interactions directes hors simulation.  