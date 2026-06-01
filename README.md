
# 🎫 Automatic Customer Support Ticket Routing

Notre projet porte sur la classification automatique de tickets de support client en utilisant des méthodes de Machine Learning classique et de Deep Learning.

## 👥 Équipe "Ti Pip" et Rôles

* **Arthur Le Coroller** (S4) : Acquisition du jeu de données, nettoyage, Analyse Exploratoire (EDA) et séparation stratifiée (train/val/test).
* **Antoine Legall** (S7) : Implémentation de bout en bout du modèle de base (Baseline) TF-IDF + Régression Logistique.
* **Arslan Tetu** (S11) : Fine-tuning du modèle DistilBERT, validation et optimisation des hyperparamètres.
* **Evan Tangatchy** (S14) : Évaluation finale, analyse qualitative des erreurs et création des supports de présentation.
* **Enzo Dziewulski** (S15) : Préparation de la soutenance finale et structuration/finalisation de ce dépôt GitHub.

## 📝 Description du Problème

Dans les grandes entreprises, les équipes de support perdent un temps précieux à trier manuellement les tickets entrants pour les rediriger vers les bons services (facturation, technique, etc.). Le tri humain ne passe pas à l'échelle et manque de cohérence, tandis que les systèmes à base de mots-clés échouent face à la complexité et l'ambiguïté du langage naturel.
Ce projet a pour but de construire un système de Machine Learning robuste pour classifier automatiquement ces tickets à partir de texte brut, optimisant ainsi la charge de travail et le temps de réponse.

## 📊 Jeu de Données (Dataset)

* **Source :** [Kaggle - Customer Support Ticket Tagger](https://www.kaggle.com/code/warcoder/customer-support-ticket-tagger) (Licence Apache 2.0).
* **Taille :** 339 échantillons de messages de clients.
* **Répartition des classes :** Fort déséquilibre avec 38% pour le Support Technique, 17% pour le Support Produit, et 45% pour la catégorie "Autre".
* **Séparation :** Split stratifié de 80% (Entraînement), 10% (Validation), et 10% (Test) pour garantir la représentativité des classes minoritaires.

## 🧠 Modèles et Approches

Afin d'évaluer l'apport du Deep Learning, nous comparons deux approches :

1. **Baseline (Machine Learning Classique) :** * **Modèle :** Vectorisation TF-IDF combinée à une Régression Logistique.
* **Objectif attendu :** Macro F1-score entre 0.65 et 0.75.


2. **Modèle Profond (Deep Learning) :** * **Modèle :** Fine-tuning de l'architecture Transformer `distilbert-base-uncased`.
* **Motivation :** Le dataset étant très petit, l'utilisation de poids pré-entraînés (Transfer Learning) permet d'exploiter une solide connaissance linguistique de base. L'agrégation via le token `[CLS]` alimentera une couche linéaire de classification.
* **Objectif attendu :** Macro F1-score entre 0.80 et 0.88.



## 📈 Évaluation et Métriques

* **Métrique Principale :** **Macro F1-score** (avec une cible $\ge 0.82$). Le choix de la métrique "Macro" est essentiel ici pour traiter toutes les classes avec une importance égale malgré le déséquilibre.
* **Métriques Secondaires :** Exactitude (Accuracy), F1-score par classe et Matrice de confusion.
* **Analyse d'Erreur :** Inspection manuelle ciblée sur les erreurs récurrentes (ex: messages trop courts, ambiguïté entre "Technique" et "Autre") pour comprendre les failles du modèle.

## ⚠️ Risques et Mitigations

* **Surapprentissage (lié aux 339 échantillons) :** Utilisation d'un modèle compact (DistilBERT), d'une forte régularisation (Dropout) et d'un arrêt précoce basé sur la validation (Early Stopping).
* **Déséquilibre des classes :** Utilisation d'une fonction de perte pondérée (Class-Weighted Cross-Entropy Loss) et d'un échantillonnage stratifié pour corriger le biais vers la classe majoritaire "Autre".


