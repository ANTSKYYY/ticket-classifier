

[## Phase 1 : Préparation des données (Milestone S4 - Arthur)](https://github.com/ANTSKYYY/deeplearning/blob/main/personnalwork/Arthur.md)

Arthur est en charge de l'acquisition, du nettoyage et de la séparation des données (S4).

* **Le jeu de données :** Il provient de Kaggle et contient 339 échantillons de messages de support client.
* **Déséquilibre des classes :** Le dataset est composé de 38% de support technique, 17% de support produit, et 45% pour la catégorie "Autre".
* **Séparation :** Il faut créer une séparation stratifiée de 80/10/10 pour l'entraînement, la validation et les tests. La stratification est cruciale ici pour s'assurer que la petite classe (17%) soit bien représentée dans chaque sous-ensemble.

## Phase 2 : Modèle de Base / Baseline (Milestone S7 - Antoine)

Antoine est responsable de l'implémentation de la baseline (S7).

* **Architecture :** Le modèle de base utilise une vectorisation TF-IDF combinée à une régression logistique.
* **Lien avec le cours :** Pour bien argumenter cette partie dans le rapport final, vous pouvez vous appuyer sur les fichiers `reading-1-logistic-regression-as-a-neural-network-companion-to-d.md` et `reading-2-deep-learning-vs-classical-machine-learning.md` présents dans votre GitHub. Cela vous aidera à expliquer théoriquement la différence entre les représentations lexicales éparses (TF-IDF) et la classification linéaire face aux réseaux de neurones profonds.
* **Objectif :** Atteindre un Macro F1-score d'environ 0.65 à 0.75.

## Phase 3 : Modèle Deep Learning (Milestone S11 - Arslan)

Arslan doit réaliser le fine-tuning du modèle DistilBERT (S11).

* **Architecture :** Vous allez utiliser `distilbert-base-uncased` via Hugging Face. Le modèle prendra les tickets tokenisés (WordPiece), et l'embedding contextuel du token `[CLS]` sera passé dans une couche de classification linéaire suivie d'une fonction softmax.
* **Lien avec le cours :** Référez-vous au fichier `reading-1-companion-notes-for-goodfellow-ch-10-sequence-modeling.md` pour maîtriser la théorie de la modélisation de séquences. De plus, le fichier `reading-2-sigmoid-softmax-and-log-loss-full-derivations.md` sera essentiel pour bien comprendre et justifier mathématiquement l'utilisation du softmax et de la perte logistique (log-loss/cross-entropy).
* **Hyperparamètres :** Utilisez l'optimiseur AdamW avec un taux d'apprentissage (learning rate) compris entre 2e-5 et 5e-5, un scheduler linéaire et des étapes de warmup. La fonction de perte sera la cross-entropy.
* **Objectif :** Surpasser la baseline et atteindre un Macro F1-score entre 0.80 et 0.88.

## Phase 4 : Évaluation et Analyse des erreurs (Milestone S14 - Evan)

Evan est en charge de l'évaluation finale et de l'analyse des erreurs (S14).

* **Métriques :** La métrique principale est le Macro F1-score, avec une cible de $\ge0.82$, ce qui permet de donner une importance égale à toutes les classes malgré le déséquilibre. Les métriques secondaires incluent l'accuracy globale et les F1-scores par classe.
* **Analyse qualitative :** Il faudra générer une matrice de confusion et inspecter manuellement un échantillon de tickets mal classifiés. L'objectif est d'identifier les schémas d'erreur (ex: intentions ambiguës, messages trop courts, chevauchement entre "Support technique" et "Autre") entre la baseline et DistilBERT.

## Phase 5 : Gestion des risques et Défense finale (Milestone S15 - Enzo)

Enzo est responsable de la préparation de la soutenance et de la finalisation du dépôt (S15).

* **Mitigation des risques liés aux données :** La petite taille du dataset (339 échantillons) risque de provoquer du surapprentissage (overfitting). Pour contrer cela, appliquez une forte régularisation (dropout) et un arrêt précoce (early stopping) basé sur la validation.
* **Mitigation du déséquilibre :** Pour éviter que le modèle ne prédise majoritairement la classe "Autre", assurez-vous d'utiliser une perte pondérée par classe (class-weighted loss) lors de l'entraînement.
* **Lien avec le cours pour la présentation :** Pour structurer la présentation finale et les diapositives, vous pouvez vous inspirer des ressources du GitHub, notamment le fichier `reading-3-skim-guide-three-illustrative-prior-year-project-poste.md` qui montre des exemples de projets des années précédentes, ainsi que `reading-3-how-to-draft-a-feasible-project-proposal.md` pour vous assurer que les conclusions répondent bien aux critères d'un projet réalisable.
