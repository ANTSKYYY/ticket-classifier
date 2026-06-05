# 🎫 Automatic Customer Support Ticket Routing

🚀 Testez notre modèle en direct : [Application Web Interactive (XAI)](https://antskyyy.github.io/deeplearning/)

Notre projet porte sur la classification automatique de tickets de support client en utilisant des méthodes de Machine Learning classique et de Deep Learning. Face aux limites d'un jeu de données initial restreint, notre équipe a adopté une approche **Data-Centric** couplée à des techniques d'entraînement avancées (**Curriculum Learning & Adversarial Training**) pour atteindre des performances de niveau production.

## 👥 Équipe "Ti Pip" et Rôles

* **Arthur Le Coroller (S4) :** Approche Data-Centric, acquisition, augmentation synthétique ciblée, purification des labels par IA et séparation stratifiée.
* **Antoine Legall (S7) :** Implémentation de la Baseline (TF-IDF + Régression Logistique) et mise en évidence des limites d'apprentissage sans contexte.
* **Arslan Tetu (S11) :** Architecture Deep Learning, fine-tuning de DistilBERT via un pipeline robuste (Curriculum Learning en 2 phases, Adversarial Training et `WeightedTrainer`).
* **Evan Tangatchy (S14) :** Évaluation finale du pipeline, extraction des métriques et analyse qualitative des erreurs résiduelles (chevauchement sémantique).
* **Enzo Dziewulski (S15) :** Préparation de la soutenance finale, défense des choix architecturaux et structuration globale du dépôt GitHub.

🔗 [Voir les journaux de bord et missions détaillées de chacun](https://github.com/ANTSKYYY/deeplearning/blob/main/docs/missions.md)

## 📝 Description du Problème

Dans les grandes entreprises, les équipes de support perdent un temps précieux à trier manuellement les tickets entrants pour les rediriger vers les bons services (Facturation, Technique, Demandes d'évolution...). Le tri humain ne passe pas à l'échelle et manque de cohérence. Les systèmes classiques à base de mots-clés, quant à eux, échouent face à la complexité et l'ambiguïté du langage naturel.

Ce projet a pour but de construire un classifieur NLP robuste, capable de comprendre l'intention sémantique d'un ticket en texte brut pour le router automatiquement avec une haute précision.

## 📊 Jeu de Données (Approche Data-Centric)

Le dataset initial (339 échantillons issus de Kaggle) s'étant révélé trop bruité et insuffisant, nous avons opéré un pivot méthodologique majeur vers l'ingénierie des données :

* **Volume :** Extension à environ 4 000 échantillons.
* **Classes cibles :** `Support` (Technique), `Feature Request` (Évolution), et `Billing and Payments` (Facturation).
* **Purification :** Application de filtres par expressions régulières (Regex) strictes et auto-correction des erreurs de labellisation humaine par un modèle IA de validation croisée.
* **Équilibrage :** Génération synthétique (via Gemini) exclusivement réservée à la classe minoritaire (*Feature Request*) pour combler son déficit, et *undersampling* strict de la classe majoritaire (*Support*) plafonnée à ~2000 tickets.
* **Séparation :** Split stratifié de 80% (Train), 10% (Val), et 10% (Test).

## 🧠 Architecture et Modèles

Afin de valider l'apport du Deep Learning, nous avons comparé deux approches :

1. **Baseline (Machine Learning Classique) :** * **Modèle :** Vectorisation TF-IDF + Régression Logistique.
* **Bilan :** Ce modèle "sac-de-mots" échoue à capter le contexte, entraînant des scores proches de zéro sur les classes minoritaires.


2. **Modèle Profond (DistilBERT + Curriculum Learning) :** * **Modèle :** Fine-tuning de l'architecture Transformer `distilbert-base-uncased`.
* **Innovation :** Pour rendre le modèle robuste face au vocabulaire trompeur, l'entraînement se déroule en deux phases (Curriculum Learning). La **Phase 1** apprend les concepts sur des données faciles. La **Phase 2** (Adversarial Training) injecte des données ambiguës avec un *learning rate* réduit pour affiner les frontières de décision sans provoquer d'oubli catastrophique.
* **Gestion du déséquilibre :** Utilisation d'une fonction de perte pondérée sur-mesure (`WeightedTrainer`).



## 🏆 Résultats Finaux et Évaluation

L'objectif initial de notre proposition (Proposal) était d'atteindre un Macro F1-score $\ge 0.82$. Le modèle final a largement dépassé ces attentes :

* **Macro F1-score : 0.9103**
* **Accuracy Globale : 90.27%**

**Analyse Qualitative :** L'audit des 9% d'erreurs restantes démontre que le modèle ne souffre plus de sous-apprentissage. La détection des *Feature Requests* est parfaite (F1 = 1.00). Les confusions résiduelles se situent exclusivement à la frontière entre le "Support" et la "Facturation", reflétant l'ambiguïté naturelle du langage humain (ex: un client signalant un "bug technique" l'empêchant de "payer"), où même un opérateur humain hésiterait.

## ⚠️ Gestion des Risques

* **Risque de données insuffisantes :** Mitigé par l'augmentation synthétique et l'approche Data-Centric.
* **Risque de biais majoritaire :** Mitigé par l'undersampling (Arthur) et la modification mathématique de la *Cross-Entropy Loss* via injection de poids de classes (Arslan).
* **Risque de fragilité aux mots-clés ("Faux positifs") :** Mitigé par la Phase 2 de l'entraînement (Adversarial) forçant le modèle à se concentrer sur l'attention contextuelle du token `[CLS]`.

---

## 📄 Licence et Utilisation des Données (Dataset)

Le jeu de données nettoyé et augmenté produit pour ce projet (notamment `train_final_3_classes.csv` et les données adversariales) est mis à disposition sous la licence **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/deed.fr)**.

**Vous êtes libre de :**

* **Partager** - copier, distribuer et communiquer le dataset par tous moyens et sous tous formats.
* **Adapter** - remixer, transformer et créer à partir du dataset pour toute utilisation, y compris commerciale.

**Sous les conditions suivantes :**

* **Attribution (Crédit) :** Vous devez créditer notre équipe, intégrer un lien vers la licence et indiquer si des modifications ont été effectuées.

**Origine des données et modifications :**
Ce dataset est une version lourdement modifiée (approche *Data-Centric*) d'un corpus originalement issu de [Kaggle -Customer Support Ticket Tagger](https://www.kaggle.com/code/warcoder/customer-support-ticket-tagger). Les modifications apportées par notre équipe incluent :

1. Le nettoyage strict par expressions régulières (Regex).
2. La correction automatisée des labels par IA.
3. L'injection de tickets synthétiques générés par l'IA (Gemini) spécifiquement pour équilibrer la classe *Feature Request*.

Si vous utilisez ce dataset dans vos travaux, merci de citer ce dépôt GitHub :

> *Ti Pip Team* -"Ticket Classification Data-Centric Dataset", 2026.
