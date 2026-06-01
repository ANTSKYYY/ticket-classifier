# 📊 Milestone S4 : Préparation des Données et Analyse Exploratoire (EDA)

**Responsable :** Arthur Le Coroller

**Objectif :** Acquérir, nettoyer et préparer le jeu de données pour les modèles de Machine Learning (Baseline) et de Deep Learning (DistilBERT), en respectant les principes d'ingénierie des données vus dans le cours DL2026.

## 1. Description du Jeu de Données

Le jeu de données `customer_tickets.csv` provient de Kaggle. Il contient des messages de support client en langage naturel (colonne `body`) et la catégorie de routage associée (colonne `queue`).

**Problématique soulevée par les données brutes :**

* Le dataset original contient potentiellement plusieurs sous-catégories (IT Support, Customer Service, etc.).
* Il comporte des balises HTML (ex: `<br>`) et des balises d'anonymisation (ex: `<name>`, `<tel_num>`).
* **Déséquilibre des classes :** Conformément à notre proposition, nous visons une répartition ciblée : Support Technique (38%), Support Produit (17%), et Autre (45%).

## 2. Pipeline de Nettoyage et de Séparation (Code Python)

Voici le script utilisé pour préparer les données. Ce code utilise `pandas` pour la manipulation des données et `scikit-learn` pour la séparation stratifiée.

```python
import pandas as pd
import re
from sklearn.model_selection import train_test_split

# ==========================================
# 1. ACQUISITION DES DONNÉES
# ==========================================
file_path = 'customer_tickets.csv'
df = pd.read_csv(file_path)

print(f"Nombre total d'échantillons originaux : {len(df)}") # Attendu : ~339 échantillons

# ==========================================
# 2. NETTOYAGE DES DONNÉES (CLEANING)
# ==========================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Suppression des balises HTML (ex: <br>)
    text = re.sub(r'<.*?>', ' ', text)
    # Remplacement des sauts de ligne par des espaces
    text = text.replace('\n', ' ').replace('\r', '')
    # Suppression des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['body_clean'] = df['body'].apply(clean_text)

# Regroupement des classes pour correspondre à la proposition du projet (3 classes cibles)
# Technical Support -> Technical Support
# Product Support -> Product Support
# Tout le reste (IT Support, Customer Service, etc.) -> Other
def map_classes(label):
    if label == "Technical Support":
        return "Technical Support"
    elif label == "Product Support":
        return "Product Support"
    else:
        return "Other"

df['queue_mapped'] = df['queue'].apply(map_classes)

# ==========================================
# 3. ANALYSE EXPLORATOIRE (EDA)
# ==========================================
print("\n--- Distribution des classes cibles ---")
class_dist = df['queue_mapped'].value_counts(normalize=True) * 100
print(class_dist)
# Attendu: Other (~45%), Technical Support (~38%), Product Support (~17%)

# ==========================================
# 4. SÉPARATION STRATIFIÉE 80/10/10
# ==========================================
# Étape 1 : Séparer en Entraînement (80%) et Reste (20%)
X_train, X_temp, y_train, y_temp = train_test_split(
    df['body_clean'], 
    df['queue_mapped'], 
    test_size=0.20, 
    stratify=df['queue_mapped'], # Stratification cruciale ici
    random_state=42
)

# Étape 2 : Séparer le Reste (20%) en Validation (10%) et Test (10%)
# Puisque test_size=0.5 sur les 20% restants, cela donne 10% du dataset total chacun.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, 
    y_temp, 
    test_size=0.50, 
    stratify=y_temp, 
    random_state=42
)

print("\n--- Tailles des sous-ensembles ---")
print(f"Train: {len(X_train)} échantillons ({len(X_train)/len(df)*100:.0f}%)")
print(f"Validation: {len(X_val)} échantillons ({len(X_val)/len(df)*100:.0f}%)")
print(f"Test: {len(X_test)} échantillons ({len(X_test)/len(df)*100:.0f}%)")

# Sauvegarde des datasets propres
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
val_df = pd.DataFrame({'text': X_val, 'label': y_val})
test_df = pd.DataFrame({'text': X_test, 'label': y_test})

train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

```

## 3. Lien avec le Cours DL2026

La réalisation de cette phase respecte rigoureusement les directives du cours (Module 2, Reading 3 - *How to Draft a Feasible Project Proposal*) pour dérisquer le projet le plus tôt possible :

1. **Focus sur la Qualité des Données ("Data work is given equal weight to modelling") :** La proposition montre que les réseaux neuronaux nécessitent des données bien comprises. Le traitement des balises HTML et des symboles de variables internes (`<name>`, `<tel_num>`) évite que le modèle Baseline (TF-IDF) n'apprenne sur du bruit (Reading 2 - Classical ML).
2. **Gestion du Déséquilibre (Class Imbalance) :** La classe *Product Support* ne représente que 17% des 339 échantillons (soit environ 57 tickets au total). Une séparation aléatoire aurait pu la faire disparaître des ensembles de validation ou de test. **L'utilisation de la stratification (`stratify=y`) garantit que ces 17% se retrouvent uniformément** dans les splits d'entraînement, de validation et de test.
3. **Préparation pour la Modélisation :** * Pour Antoine (Baseline - S7) : Le nettoyage poussé (espaces, retours chariots) va optimiser la matrice creuse (sparse matrix) générée par le TF-IDF.
* Pour Arslan (Deep Learning - S11) : Les modèles de type *Transformer* (DistilBERT) utilisent un tokeniseur par sous-mots (WordPiece) qui gère bien la casse, mais qui est perturbé par du code HTML non nettoyé. L'approche est donc prête pour le Fine-Tuning.
