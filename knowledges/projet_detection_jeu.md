# Projet : Détection d'entités avec Overlay en temps réel (début 2025)

## Identité
**Auteur :** Léo Palich  
**Contexte :** Projet personnel de développement mêlant vision par ordinateur, apprentissage profond et interfaces graphiques.  

## Objectif
L’objectif de ce projet est de mettre en place un système capable de **détecter en temps réel des entités visuelles** (par exemple des silhouettes) à l’écran, puis de **superposer dynamiquement des informations** via un overlay transparent.  
Le but est simple :  
- Vérifier s’il est possible de détecter des silhouettes extrêmement variables en temps réel grâce à un modèle YOLO intégré dans un overlay interactif.  


# Parcours académique
Ce projet s’inscrit dans le cadre de mon parcours en **sciences cognitives** et en **IA appliquée**, où j’explore la vision artificielle.  

# Description technique

## Technologies utilisées
- **Langage :** Python  
- **Bibliothèques principales :**
  - [PyQt5](https://doc.qt.io/qtforpython/) pour l’interface graphique et l’overlay  
  - [MSS](https://python-mss.readthedocs.io/) pour la capture d’écran rapide  
  - [OpenCV](https://opencv.org/) et [NumPy](https://numpy.org/) pour le traitement d’image  
  - [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) pour la détection d’objets  
  - [Torch](https://pytorch.org/) pour l’accélération GPU des modèles  

## Fonctionnement général
1. **Capture d’écran en continu** grâce à `mss` (monitor principal).  
2. **Passage de l’image au modèle YOLO** (`best.pt` entraîné sur la classe *entity*).  
3. **Traitement des résultats** :
   - Extraction des boîtes englobantes (`x1, y1, x2, y2`).  
   - Récupération des classes et des scores de confiance.  
4. **Affichage via Overlay PyQt5** :
   - Fenêtre transparente, non bloquante et toujours au-dessus.  
   - Dessin des boîtes en temps réel.  
   - Étiquettes affichant nom de classe + score (dans `et.py`).  
   - Système de **lissage des boîtes** pour éviter les sauts (dans `entity.py`).  

## Variantes implémentées
- `et.py` : version complète avec affichage des **labels et scores de confiance**.  
- `entity.py` : version optimisée avec **downscaling + lissage** des boîtes pour plus de stabilité visuelle.  
- `record.ipynb` : notebook permettant l’enregistrement ou l’expérimentation.  
- `entity.ipynb` : notebook d’exploration pour tester et analyser le modèle.  

# Résultats attendus
- Détection en temps réel avec overlay fluide et non intrusif.  
- Robustesse sur des environnements complexes avec beaucoup de mouvement.  
- Base modulable pour :  
  - surveillance en temps réel,  
  - détection de triche/exploits en jeux vidéo,  
  - analyse comportementale.  

# Perspectives
- Optimisation du dataset dans ce contexte précis
- Bonne base architecturale pour détecter d'autre entités en temps réel (avec un dataset adapté)