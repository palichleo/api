# Projet : Création d'un système RAG pour un LLM [été 2025]

## Identité
Auteur : Léo Palich  
Contexte : projet personnel pour doter mon site/portfolio d’un assistant qui connaît mes projets, mon CV et mes notes, et répond de façon courte, factuelle et datée.

## Objectif
- Construire un système Retrieval-Augmented Generation (RAG) capable de répondre sur mon parcours, mes réalisations et mes compétences.
- Priorité à la qualité des réponses et au respect de la temporalité (passé/présent), tout en restant compatible avec un serveur CPU peu puissant.
- Éviter les hallucinations en ne parlant que depuis mes propres documents.

# Description technique

## Pile technologique
- Node.js (Express) pour l’API, streaming de réponses.
- ChromaDB comme vecteur-store local.
- Embeddings locaux via @xenova/transformers + onnxruntime-web (modèle : paraphrase-multilingual-MiniLM-L12-v2, quantized).
- Génération avec Groq (LLM chat completions), modèle par défaut : llama-3.1-70b-versatile pour la qualité.
- Supervision avec PM2, variables d’environnement via .env.

## Architecture du pipeline
1. Ingestion
   - Lecture de fichiers Markdown, textes et notes personnelles. Extraction de texte simple.
   - Nettoyage basique (suppression des blocs de code, liens transformés en texte, guillemets normalisés).
   - Option sentence explode : chaque puce ou phrase devient un document court pour améliorer le rappel.

2. Indexation
   - Découpage “intelligent” des Markdown : conservation d’un chemin de titres (H1 > H2 > H3) et détection d’une éventuelle année dans le texte.
   - Enrichissement de métadonnées : source, hpath (chemin de titres), year (si détectée), liste de mots-clés simples.
   - Embeddings locaux (MiniLM quantized) puis ajout par lots dans Chroma.
   - Réindexation possible à la demande via un script Node.

3. Récupération (retrieval)
   - Expansion locale de la requête : normalisation, suppression des stopwords, variantes synonymes ciblées (parcours/expérience, IA/ML, data/statistiques, freelance/indépendant).
   - Recherche dans Chroma pour chaque variante, union des candidats.
   - Scoring local combiné : similarité cosinus (embedding), recouvrement lexical (Jaccard sur tokens), récence (bonus si l’année est proche d’aujourd’hui).
   - MMR pour diversifier les passages retenus.
   - Déduplication légère par similarité d’embeddings.

4. Génération
   - Construction de contexte court (jusqu’à 6 extraits) avec nettoyage Markdown pour éviter tout style.
   - Prompt système minimal : identité “Tu es Léo Palich”, date courante injectée (NOW=YYYY-MM-DD), consigne stricte “texte brut, 1–2 phrases, pas de Markdown”.
   - Filtre de sortie sur le stream (suppression des astérisques, backticks et autres marqueurs) pour garantir un rendu naturel.

## Fichiers clés
- index.js : API Express, appel Groq en streaming, nettoyage du contexte et filtrage de sortie.
- rag/indexer.js : découpage Markdown, extraction métadonnées, embeddings et insertion dans Chroma.
- rag/embedder.js : pipeline d’embedding local (onnxruntime-web, sans worker).
- rag/retriever.js : expansions locales, scoring, MMR, déduplication, retour des meilleurs extraits.

# Résultats
- Réponses courtes et naturelles sur mon parcours, sans style Markdown intempestif.
- Diminution nette des hallucinations grâce à la contrainte “répondre uniquement depuis les extraits”.
- Respect de la temporalité : le système privilégie les formulations au passé pour les projets datés et évite de présenter d’anciens travaux au présent.
- Fonctionne en local sur CPU (embeddings) avec un coût externe minimal côté génération.

# Limites
- Sur un serveur tourné CPU, le LLM était très lent.
- La qualité dépend de la clarté et de la couverture des documents sources.

# Perspectives
- Activer un reranker cross-encoder local (bge-reranker-base quantized) pour un tri encore plus précis.
- Tester des embeddings plus puissants (multilingual-e5-base, bge-m3) si la RAM le permet, avec réindexation complète.
- Ajouter une évaluation automatique (jeu de questions/réponses attendu) pour suivre la qualité à chaque mise à jour.
- Mettre en place une tâche d’ingestion continue (watcher) pour réindexer dès qu’un fichier du portfolio change.
- Générer des citations cliquables vers la source dans l’UI front pour la traçabilité.