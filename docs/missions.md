

## Phase 1 : Préparation des données (Milestone S4 - Arthur)

[Travail détaillé](https://github.com/ANTSKYYY/deeplearning/blob/main/personnalwork/Arthur.md)

Arthur est en charge de l'acquisition, du nettoyage et de la préparation des données (S4). L'ancien dataset (339 échantillons) a été remplacé par une base de données beaucoup plus robuste pour permettre un apprentissage profond efficace.

* **Le jeu de données :** Il contient désormais environ 4 000 échantillons de tickets clients [Dataset](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets).
* **Classes et distribution :** Le dataset se divise en trois catégories : *Support*, *Feature Request*, et *Billing and Payments*.
* **Équilibrage (Undersampling) :** Pour éviter un biais massif, la classe majoritaire (*Support*) est plafonnée à 2000 échantillons maximum avant d'être mélangée aux autres classes.
* **Nettoyage avancé :** Application d'un pipeline Regex (`clean_text_bert`) pour standardiser les salutations, signatures, versions et noms de logiciels spécifiques (ex: remplacement par des tokens génériques comme `SOFTWARE_PRODUCT`).

## Phase 2 : Modèle de Base / Baseline (Milestone S7 - Antoine)

[Travail détaillé](https://github.com/ANTSKYYY/deeplearning/blob/main/personnalwork/Antoine.md)

Antoine est responsable de l'implémentation de la baseline (S7) sur ce nouveau dataset élargi.

* **Architecture :** Le modèle de base utilise une vectorisation TF-IDF combinée à une régression logistique.
* **Objectif :** Établir un score de référence (Macro F1-score) sur le nouveau corpus de 4K données avant de passer au Deep Learning.

## Phase 3 : Modèle Deep Learning Avancé (Milestone S11 - Arslan)

[Travail détaillé](https://github.com/ANTSKYYY/deeplearning/blob/main/personnalwork/Arslan.md)

Arslan a entièrement repensé l'approche Deep Learning avec un pipeline DistilBERT intégrant de l'**Adversarial Training** et du **Curriculum Learning** (S11).

* **Architecture & Curriculum Learning :** Le fine-tuning de `distilbert-base-uncased` se fait désormais en deux étapes progressives :
* **Phase 1 (Données normales / faciles) :** Le modèle apprend à construire des représentations sémantiques solides sur le dataset propre (LR = 1.5e-5, 2 epochs).
* **Phase 2 (Données difficiles / Adversarial) :** Injection d'exemples adversariaux (oversamplés pour atteindre un ratio de 1:3, soit 25% du dataset augmenté). Le taux d'apprentissage est drastiquement réduit (LR = 5e-6) pour ajuster les frontières de décision ambiguës sans provoquer d'oubli catastrophique (*catastrophic forgetting*) des acquis de la Phase 1.


* **Gestion du déséquilibre (WeightedTrainer) :** Création d'une classe d'entraînement personnalisée injectant des poids de classes (`class_weights`) dynamiques directement dans la fonction de perte Cross-Entropy.


## Phase 4 : Évaluation et Analyse des erreurs (Milestone S14 - Evan)

[Travail détaillé](https://github.com/ANTSKYYY/deeplearning/blob/main/personnalwork/Evan.md)

Evan est en charge de l'évaluation finale et de l'analyse des erreurs générées par le nouveau pipeline (S14).

* **Métriques :** La métrique principale reste le Macro F1-score, qui valide l'efficacité du `WeightedTrainer` sur les classes minoritaires. Les métriques secondaires incluent l'accuracy globale.
* **Analyse qualitative :** Il faudra générer la matrice de confusion finale. L'enjeu sera de vérifier si le modèle de la Phase 2 (Adversarial) résiste mieux aux faux positifs et aux intentions ambiguës par rapport au modèle de la Phase 1 ou à la baseline TF-IDF.

## Phase 5 : Gestion des risques et Défense finale (Milestone S15 - Enzo)

[Travail détaillé](https://github.com/ANTSKYYY/deeplearning/blob/main/personnalwork/Enzo.md)

Enzo est responsable de la préparation de la soutenance et de la finalisation du dépôt (S15), avec de nouveaux arguments liés aux modifications d'Arslan et Arthur.

* **Mitigation des risques de données :** Le problème initial (dataset trop petit avec 339 échantillons) a été résolu par l'acquisition du nouveau dataset de 4000 lignes.
* **Mitigation de la fragilité du modèle :** La vulnérabilité habituelle des modèles NLP aux attaques ou au bruit est directement contrée par la **Phase 2 (Adversarial Training)** du pipeline d'Arslan.
* **Mitigation du déséquilibre et de l'oubli :** Le risque de prédire uniquement la classe majoritaire est géré mathématiquement par l'injection de poids (`WeightedTrainer`). Le risque de *catastrophic forgetting* lors de la phase 2 est géré par la réduction du *learning rate*.
