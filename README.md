# 🎫 Automatic Customer Support Ticket Routing

> Un classifieur NLP (DistilBERT) robuste atteignant un **Macro F1-score de 0.9178** grâce à une approche Data-Centric, de l'Adversarial Training et du Curriculum Learning.

🚀 **Testez notre modèle en direct :** [Application Web Interactive (XAI)](https://antskyyy.github.io/deeplearning/)

## ⚙️ Setup & Reproduction (en 5 étapes)

```bash
# 1. Cloner le dépôt
git clone https://github.com/ANTSKYYY/deeplearning.git
cd deeplearning

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Entraîner le modèle complet (Phase 1 & Phase 2)
make train

# 4. Évaluer le modèle sur le jeu de test
make eval

# 5. Analyser les résultats :** Consultez le dossier `results/` pour retrouver les métriques officielles (`metrics.json`), la matrice de confusion (`figures/final_confusion_matrix.png`) et l'audit des confusions (`qualitative_analysis_errors.csv`).

```
---

## 📝 Description du Problème

Dans les grandes entreprises, les équipes de support perdent un temps précieux à trier manuellement les tickets entrants pour les rediriger vers les bons services (Facturation, Technique, Demandes d'évolution). Les systèmes classiques à base de mots-clés échouent face à l'ambiguïté du langage naturel. Ce projet implémente un classifieur NLP de niveau production pour router automatiquement ces tickets en texte brut avec une haute précision.

## 👥 Équipe "Ti Pip" et Rôles

* **Arthur Le Coroller (S4) :** Approche Data-Centric, acquisition, nettoyage par regex, et séparation stratifiée.
* **Antoine Legall (S7) :** Implémentation de la Baseline (TF-IDF + Régression Logistique) et structuration du code de production.
* **Arslan Tetu (S11) :** Architecture Deep Learning, fine-tuning de DistilBERT via un pipeline robuste (Curriculum Learning en 2 phases, Adversarial Training et `WeightedTrainer`).
* **Evan Tangatchy (S14) :** Évaluation finale du pipeline, extraction des métriques et analyse qualitative des erreurs résiduelles (chevauchement sémantique).
* **Enzo Dziewulski (S15) :** Préparation de la soutenance finale, défense des choix architecturaux et gestion des risques.

🔗 [Voir les journaux de bord et missions détaillées de chacun](https://github.com/ANTSKYYY/deeplearning/blob/main/docs/missions.md)

## 📊 Jeu de Données (Approche Data-Centric)

Le dataset initial (339 échantillons) étant trop bruité, nous avons opéré un pivot méthodologique majeur vers l'ingénierie des données :

* **Volume :** Extension à environ 4 000 échantillons.
* **Classes cibles :** `Support` (Technique), `Feature Request` (Évolution), et `Billing and Payments` (Facturation).
* **Purification :** Application de filtres par expressions régulières strictes et auto-correction des erreurs de labellisation.
* **Équilibrage :** Génération synthétique pour la classe minoritaire (*Feature Request*) et *undersampling* strict de la classe majoritaire (*Support*).

## 🧠 Architecture et Modèles

Afin de valider l'apport du Deep Learning, nous avons comparé deux approches :

1. **Baseline (Machine Learning Classique) :** Vectorisation TF-IDF + Régression Logistique. Ce modèle échoue à capter le contexte, entraînant des scores très bas sur les classes minoritaires.
2. **Modèle Profond (DistilBERT + Curriculum Learning) :** Fine-tuning de l'architecture Transformer `distilbert-base-uncased`.
* **Innovation :** L'entraînement se déroule en deux phases. La **Phase 1** apprend les concepts sur des données faciles. La **Phase 2** (Adversarial Training) injecte 25% de données ambiguës avec un learning rate réduit pour affiner les frontières de décision sans provoquer d'oubli catastrophique.
* **Gestion du déséquilibre :** Utilisation d'une fonction de perte pondérée sur-mesure (`WeightedTrainer`).



## 🏆 Résultats Finaux et Évaluation

L'objectif initial de notre proposition était d'atteindre un Macro F1-score $\ge$ 0.82. Le modèle final a largement dépassé ces attentes :

* **Macro F1-score : 0.9178**
* **Accuracy Globale : 91.16%**

**Analyse Qualitative :** La détection des *Feature Requests* est parfaite (F1 = 1.00). Les 9% d'erreurs restantes se situent exclusivement à la frontière entre le "Support" et la "Facturation", reflétant l'ambiguïté naturelle du langage humain (ex: un client signalant un "bug technique" l'empêchant de "payer"), où même un opérateur humain hésiterait.

## ⚠️ Gestion des Risques

* **Données insuffisantes :** Mitigé par l'augmentation synthétique et l'approche Data-Centric.
* **Biais majoritaire :** Mitigé par l'undersampling et la modification mathématique de la *Cross-Entropy Loss* via l'injection de poids de classes.
* **Fragilité aux mots-clés (Faux positifs) :** Mitigé par la Phase 2 de l'entraînement (Adversarial) forçant le modèle à se concentrer sur l'attention contextuelle globale.

---

## Acknowledgements & AI Disclosure
As per syllabus requirements, we disclose the use of the following AI tools during this project:

* Google Gemini: Used to brainstorm and synthetically generate minority class tickets (Feature Requests) during the data augmentation phase.
* ChatGPT / Claude: Used to debug specific PyTorch/Hugging Face cache permission errors, brainstorm dataset purification strategies, and refactor our exploratory Jupyter Notebooks into production-grade.

## 📄 Licence

* **Code source :** Ce projet est mis à disposition sous la licence **[MIT License](https://github.com/ANTSKYYY/deeplearning/blob/main/LICENSE)**.
* **Jeu de données :** Le dataset nettoyé et augmenté produit pour ce projet est mis à disposition sous la licence **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/deed.fr)**. Si vous utilisez ce dataset, merci de citer ce dépôt : *> Ti Pip Team - "Ticket Classification Data-Centric Dataset", 2026.*
