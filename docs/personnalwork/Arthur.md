# 📊 Milestone S4 : Préparation des Données et Approche Data-Centric

**Responsable :** Arthur Le Coroller
**Objectif :** Acquérir, nettoyer, augmenter et purifier le jeu de données pour garantir le succès des modèles de Machine Learning (Baseline) et de Deep Learning (DistilBERT), en appliquant les principes d'ingénierie des données vus dans le cours DL2026.

## 1. Contexte et Problématique Initiale

Lors de la phase d'exploration initiale, notre jeu de données original (Kaggle) ne contenait qu'environ 339 échantillons. Les premiers tests d'entraînement montraient que les modèles plafonnaient à une précision d'environ 50%, même avec des architectures avancées comme DistilBERT.

Des audits approfondis des données ont révélé que le problème ne venait pas des modèles, mais de la **vérité terrain** :

* Volume de données largement insuffisant pour du Deep Learning.
* Bruit important dans les textes bruts.
* Déséquilibre critique des classes.
* Erreurs humaines fréquentes dans l'étiquetage d'origine.

J'ai donc décidé de pivoter vers une approche **Data-Centric**, en considérant que l'optimisation de la donnée était prioritaire sur l'optimisation de l'algorithme.

## 2. Pipeline Data-Centric (Méthodologie et Code)

Pour résoudre ces problèmes, j'ai mis en place un pipeline de traitement en quatre grandes étapes.

### Étape 1 : Augmentation des Données et Fusion

Pour pallier le manque de volume, le corpus a été étendu à plus de 4 000 échantillons. **Il est important de préciser que la très grande majorité de ces données provient directement du dataset Kaggle et est composée de tickets réels (non synthétiques).** Cependant, la classe *Feature Request* était en sous-effectif critique. Pour combler ce vide spécifique, des tickets ont été générés par IA (Gemini) en se basant strictement sur des modèles de tickets déjà existants. **Seule la catégorie des demandes de fonctionnalités a fait l'objet d'une augmentation synthétique.** L'ensemble a ensuite été fusionné et brassé (*shuffle*) pour éviter tout biais d'ordre d'apprentissage.

```python
import pandas as pd

# Chargement du dataset réel (Kaggle) et du dataset synthétique ciblé
df_ancien = pd.read_csv('data/new/data4.csv') # Données 100% réelles
df_nouveau = pd.read_csv('data/new/synthetic_feature_requests_gemini.csv') # Uniquement Feature Requests

# Uniformisation des colonnes et nettoyage des valeurs nulles
df_ancien = df_ancien[['queue', 'body']].rename(columns={'text': 'body', 'label': 'queue'}, errors='ignore')
df_nouveau = df_nouveau[['queue', 'body']]

# Fusion et Mélange (Shuffle)
df_final = pd.concat([df_ancien, df_nouveau], ignore_index=True)
df_final = df_final.dropna(subset=['queue', 'body']).drop_duplicates(subset=['body'])
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

df_final.to_csv('train_final_3_classes.csv', index=False)

```

### Étape 2 : Extraction Chirurgicale (Regex N-Grams)

Pour garantir l'intégrité de la classe *Feature Request* (qu'elle soit réelle ou synthétique), j'ai conçu un extracteur basé sur des expressions régulières strictes validées humainement. Cela permet d'isoler mathématiquement les vraies demandes d'évolution du reste du bruit.

```python
import re

strict_keywords = [
    r'request the implementation', r'suggest redesigning', 
    r'requesting a new feature', r'would be a great addition',
    r'it would be nice if', r'suggest adding', r'feature request'
]
pattern = re.compile('|'.join(strict_keywords), flags=re.IGNORECASE)

def find_strict_triggers(text):
    if not isinstance(text, str): return ""
    matches = pattern.findall(text)
    return ", ".join(set([m.lower() for m in matches]))

df_final['matched_keywords'] = df_final['body'].apply(find_strict_triggers)
feature_requests_df = df_final[df_final['matched_keywords'] != ""].copy()

```

### Étape 3 : Auto-Correction des Labels par l'IA

De nombreux tickets originaux issus de Kaggle étaient mal catégorisés par les opérateurs humains. J'ai déployé un script d'inférence de masse pour qu'un modèle IA pré-entraîné relise l'intégralité du dataset. Si l'IA contredisait le label humain avec une confiance extrême ($\ge 90\%$), le label était corrigé automatiquement.

```python
from transformers import pipeline

# Inférence de masse sur le corpus nettoyé
classifier = pipeline("text-classification", model="./distilbert-ticket-classifier-final", device=-1)
results = classifier(train_df['clean_text'].to_list(), truncation=True, max_length=128)

train_df['Prediction_IA'] = [res['label'] for res in results]
train_df['Confidence'] = [res['score'] for res in results]

# Écrasement des erreurs humaines évidentes
SEUIL_CONFIANCE = 0.90
mask_to_correct = (train_df['queue'] != train_df['Prediction_IA']) & (train_df['Confidence'] >= SEUIL_CONFIANCE)

train_df.loc[mask_to_correct, 'queue'] = train_df.loc[mask_to_correct, 'Prediction_IA']
train_df.drop(columns=['clean_text', 'Prediction_IA', 'Confidence']).to_csv('data/train_autocorrected.csv', index=False)

```

### Étape 4 : Équilibrage (Undersampling)

Même après augmentation, la classe *Support* (100% issue de Kaggle) restait sur-représentée. Pour éviter que le modèle ne développe un biais de prédiction majoritaire, j'ai appliqué un undersampling strict sur cette classe avant la séparation finale stratifiée.

```python
# Plafonnement de la classe majoritaire
support = df_final[df_final["queue"] == "Support"].sample(n=2094, random_state=42)
others = df_final[df_final["queue"] != "Support"]

# Reconstitution du dataset équilibré
df_balanced = pd.concat([support, others], ignore_index=True)
print(df_balanced["queue"].value_counts())

```

## 3. Lien avec le Cours DL2026

La réalisation de ce milestone valide rigoureusement les concepts abordés dans le cours (Module 2, Reading 3 - *How to Draft a Feasible Project Proposal*) pour dérisquer le projet le plus tôt possible :

1. **Focus sur la Qualité des Données ("Data work is given equal weight to modelling") :** La proposition du cours insiste sur le fait que l'architecture d'un modèle ne peut compenser une mauvaise donnée. Le passage au *Data-Centric* (purification par IA, regex strictes) a permis de débloquer les performances du modèle, passant de 50% à des scores prêts pour la production.
2. **Gestion du Déséquilibre (Class Imbalance) :** Sans l'undersampling à 2094 échantillons et l'augmentation ciblée par données synthétiques pour les *Feature Requests*, les modèles de la Baseline (Antoine) et Deep Learning (Arslan) auraient ignoré les classes minoritaires.
3. **Préparation pour la Modélisation :** L'élimination du bruit garantit une matrice creuse (TF-IDF) de haute qualité pour la régression logistique, et des *embeddings* riches de sens pour les tokens contextuels (`[CLS]`) du Transformer DistilBERT.
