# 🚀 Milestone S7 : Modèle de Base (Baseline) - TF-IDF & Régression Logistique

**Responsable :** Antoine Legall

**Objectif :** Implémenter le modèle de base (*baseline*) de bout en bout en utilisant une vectorisation TF-IDF combinée à une Régression Logistique.
**Prérequis :** Les données ont été nettoyées et divisées (80% Train, 10% Validation, 10% Test) lors de la phase précédente (S4) par Arthur.

## 1. Pipeline de la Baseline (Code Python)

Le script ci-dessous charge les données préparées, extrait les caractéristiques textuelles (TF-IDF) et entraîne un modèle de Régression Logistique. L'évaluation se concentre sur le **Macro F1-score**, métrique choisie pour gérer le déséquilibre des classes.

```python
# src/run_baseline.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. CHARGEMENT DES DONNÉES (Split d'Arthur)
# ==========================================
print("Chargement des données...")
train_df = pd.read_csv('data/train.csv')
val_df = pd.read_csv('data/val.csv')
test_df = pd.read_csv('data/test.csv')

# Séparation des features (X) et labels (y)
X_train, y_train = train_df['text'].fillna(""), train_df['label']
X_val, y_val = val_df['text'].fillna(""), val_df['label']
X_test, y_test = test_df['text'].fillna(""), test_df['label']

# ==========================================
# 2. VECTORISATION TF-IDF
# ==========================================
print("Extraction des features avec TF-IDF...")
# Extraction des unigrammes et bigrammes, en ignorant les mots vides (stopwords)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# ==========================================
# 3. ENTRAÎNEMENT DE LA RÉGRESSION LOGISTIQUE
# ==========================================
print("Entraînement de la Régression Logistique...")
# class_weight='balanced' permet de gérer le déséquilibre des classes observé lors de l'EDA
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train_tfidf, y_train)

# ==========================================
# 4. ÉVALUATION (Validation & Test)
# ==========================================
print("\n--- ÉVALUATION SUR LE SET DE VALIDATION ---")
y_val_pred = model.predict(X_val_tfidf)
val_macro_f1 = f1_score(y_val, y_val_pred, average='macro')
print(f"Validation Macro F1-score: {val_macro_f1:.4f}")

print("\n--- ÉVALUATION SUR LE SET DE TEST ---")
y_test_pred = model.predict(X_test_tfidf)
test_macro_f1 = f1_score(y_test, y_test_pred, average='macro')
print(f"Test Macro F1-score: {test_macro_f1:.4f}")

print("\nRapport de classification détaillé (Test) :")
print(classification_report(y_test, y_test_pred))

# ==========================================
# 5. MATRICE DE CONFUSION
# ==========================================
cm = confusion_matrix(y_test, y_test_pred, labels=model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title(f'Matrice de Confusion - Baseline TF-IDF + LR\nMacro F1: {test_macro_f1:.2f}')
plt.ylabel('Vraie Classe')
plt.xlabel('Classe Prédite')
plt.tight_layout()
plt.savefig('docs/images/baseline_confusion_matrix.png')
print("Matrice de confusion sauvegardée dans 'docs/images/baseline_confusion_matrix.png'.")

```

## 2. Argumentation Théorique (Lien avec le cours DL2026)

Dans le cadre du cours, il est essentiel de justifier la présence de cette Baseline avant de passer aux architectures de Deep Learning.

### A. Pourquoi commencer par le Machine Learning Classique ?

Comme l'explique le cours `reading-2-deep-learning-vs-classical-machine-learning.md`, le Deep Learning n'est pas toujours la solution par défaut.

* **Taille des données :** Le Machine Learning classique est particulièrement performant sur des jeux de données de petite taille. Notre dataset ne contient que 339 échantillons, ce qui est largement en dessous des dizaines de milliers d'échantillons typiquement requis pour entraîner un modèle profond performant depuis zéro.
* **Feature Engineering manuel vs appris :** Contrairement au Deep Learning qui apprend ses propres représentations hiérarchiques à partir de données brutes, l'approche classique nécessite une ingénierie des caractéristiques. Ici, nous utilisons l'algorithme TF-IDF qui crée une **représentation lexicale éparse (sparse)**, attribuant un poids aux mots en fonction de leur fréquence dans le document et de leur rareté dans le corpus global.
* **Le rôle de la baseline :** Le cours stipule expressément qu'il faut *toujours entraîner une baseline classique* afin de vérifier que le modèle profond (DistilBERT) apporte une réelle valeur ajoutée. Si le modèle profond n'est pas meilleur que la régression logistique, c'est que le modèle profond est cassé, pas meilleur.

### B. La Régression Logistique comme réseau de neurones primitif

D'après `reading-1-logistic-regression-as-a-neural-network-companion-to-d.md`, la Régression Logistique peut être vue comme le plus petit réseau de neurones possible (un réseau à un seul neurone sans couche cachée).

* **Structure mathématique :** Le modèle effectue une transformation linéaire ($z = w \cdot x + b$) suivie d'une fonction d'activation non-linéaire (Softmax, car nous avons une classification multi-classes).
* **Perte et interprétabilité :** Le modèle minimise une fonction de perte **Cross-Entropy** (log-loss), ce qui lui confère de très bonnes propriétés de convergence (le problème est convexe). De plus, contrairement aux réseaux profonds opaques, cette baseline est hautement interprétable (les poids $w$ nous disent exactement quels mots influencent telle ou telle catégorie).

## 3. Objectifs et Résultats Attendus

* **Cible :** Comme défini dans le Proposal, l'objectif pour cette baseline est d'obtenir un **Macro F1-score situé entre 0.65 et 0.75**.
* **Prochaine étape :** Une fois ce score obtenu, la valeur de la Baseline sera figée. Le jalon suivant (S11 - Arslan) consistera à appliquer un modèle de langage pré-entraîné (DistilBERT) pour extraire des embeddings contextuels (plutôt que les embeddings sparses du TF-IDF) afin d'essayer de battre cette limite et d'atteindre 0.80 - 0.88 de Macro F1-score.

## 4. Analyse des Résultats de la Baseline

Les résultats obtenus sur le set de test (Macro F1-score de 0.42) sont inférieurs à notre cible théorique (0.65 - 0.75). L'analyse du rapport de classification met en évidence une défaillance majeure :
* **Incapacité à détecter la classe minoritaire :** Le modèle obtient un F1-score de 0.00 sur la classe `Product Support`. 
* **Surapprentissage (Overfitting) :** L'écart important entre la validation (0.64) et le test (0.42) montre que l'approche TF-IDF peine à généraliser sur un si petit jeu de données (339 échantillons).

**Conclusion S7 :** Cette baseline classique démontre que la simple occurrence de mots (TF-IDF) ne permet pas de capturer les nuances sémantiques nécessaires pour router des tickets ambigus, surtout pour des classes sous-représentées. Cela valide pleinement la nécessité de passer à la phase suivante (S11) : l'utilisation d'un modèle de langage contextuel pré-entraîné (DistilBERT) via Transfer Learning.

  
![grahe](../img/baseline_confusion_matrix.png "Le titre de mon image")
