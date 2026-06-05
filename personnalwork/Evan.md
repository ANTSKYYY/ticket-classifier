# 📊 Milestone S14 : Évaluation Finale et Analyse Qualitative des Erreurs

**Responsable :** Evan
**Objectif :** Évaluer rigoureusement les performances finales du modèle DistilBERT (issu de la Phase 2 avec Curriculum Learning), valider l'efficacité du pipeline global, et mener une analyse qualitative approfondie des 9% d'erreurs résiduelles.

## 1. Contexte Théorique et Choix des Métriques (Lien avec le cours)

L'évaluation d'un modèle d'apprentissage automatique face à un dataset déséquilibré requiert des métriques adaptées.

* **Métrique Principale : Le Macro F1-score.**
L'**Accuracy globale** donne le pourcentage total de prédictions correctes. Cependant, si un modèle se contente de prédire majoritairement la classe "Support" (la plus fréquente), son accuracy restera artificiellement haute. Le **Macro F1-score** calcule la moyenne non pondérée des F1-scores de chaque classe.
Pour atteindre notre cible initiale ($\ge 0.82$), le modèle était mathématiquement contraint d'être très performant sur les classes minoritaires (*Feature Request* et *Billing*). C'est cette métrique qui valide la réussite du `WeightedTrainer` mis en place par Arslan.

## 2. Pipeline d'Évaluation Officiel (Code Python)

Voici le script final d'évaluation. Il inclut la recherche automatique du dernier checkpoint de la Phase 2, et la correction du mapping (Texte/Entiers) pour extraire précisément les vraies erreurs.

```python
# src/evaluate_final_model.py
import os
import glob
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_HUB_DISABLE_TOKEN"] = "1"

DEVICE = torch.device("cpu") # Ou "cuda"/"mps" si disponible


# 1. CHARGEMENT DES DONNÉES ET DU MODÈLE
print("Chargement du set de test...")
test_df = pd.read_csv('data/test.csv')

label_mapping = {"Support": 0, "Feature Request": 1, "Billing and Payments": 2}
inverse_mapping = {v: k for k, v in label_mapping.items()}

# Recherche automatique du dernier checkpoint d'Arslan
base_dir = "./curriculum-phase2-final"
checkpoints = glob.glob(f"{base_dir}/checkpoint-*")
if checkpoints:
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    MODEL_PATH = checkpoints[-1]
else:
    MODEL_PATH = base_dir

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# 2. GÉNÉRATION DES PRÉDICTIONS
print("Inférence en cours...")
texts = test_df['text'].tolist()
# Conversion des labels textuels en entiers pour le calcul des métriques
true_labels = test_df['label'].map(label_mapping).tolist()

inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()


# 3. RÉSULTATS GLOBAUX ET MATRICE

macro_f1 = f1_score(true_labels, predictions, average='macro')
accuracy = accuracy_score(true_labels, predictions)

print("\n" + "="*40)
print(f"RÉSULTATS GLOBAUX (PHASE 2)")
print("="*40)
print(f"Macro F1-score : {macro_f1:.4f} (Cible : >= 0.82)")
print(f"Accuracy       : {accuracy:.4f}")
print("\nRAPPORT DÉTAILLÉ :")
print(classification_report(true_labels, predictions, target_names=list(label_mapping.keys())))

# Génération Matrice de Confusion
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(label_mapping.keys()),
            yticklabels=list(label_mapping.keys()))
plt.title(f'Matrice de Confusion Finale (DistilBERT)\nMacro F1: {macro_f1:.4f}')
plt.ylabel('Vraie Classe')
plt.xlabel('Classe Prédite')
plt.tight_layout()
os.makedirs('docs/images', exist_ok=True)
plt.savefig('docs/images/final_confusion_matrix.png')

# 4. EXTRACTION DES VRAIES ERREURS
# Re-traduction des prédictions en texte
test_df['predicted_label_text'] = [inverse_mapping[p] for p in predictions]
# On isole uniquement les lignes où le texte diffère
errors_df = test_df[test_df['label'] != test_df['predicted_label_text']].copy()
errors_df = errors_df.rename(columns={'label': 'vrai_nom', 'predicted_label_text': 'nom_predit'})

errors_path = 'data/qualitative_analysis_errors.csv'
errors_df[['text', 'vrai_nom', 'nom_predit']].to_csv(errors_path, index=False)
print(f"{len(errors_df)} vraies erreurs extraites pour l'audit manuel : {errors_path}")

```

## 3. Résultats et Analyse Qualitative (Autopsie des Erreurs)

Notre objectif initial était d'atteindre un Macro F1-score d'au moins 0.82. Les résultats obtenus pulvérisent cette attente avec un **Macro F1-score de 0.9103** et une **Accuracy de 90.27%**.

Sur les 442 tickets de test aveugle, seules **43 erreurs** ont été recensées. Mon rôle consiste à analyser qualitativement ces échecs pour démontrer qu'ils ne relèvent pas d'une faille architecturale, mais de l'ambiguïté du langage naturel.

### A. La perfection sur "Feature Request" (Preuve de l'approche Data-Centric)

* **Score F1 : 1.00 (100%)**
### A. La perfection sur "Feature Request" : Succès Data-Centric et Biais Synthétique

* **Score F1 : 1.00 (100%)**
* **Analyse :** Le modèle ne fait littéralement **aucune erreur** sur cette classe. D'un côté, cela prouve l'efficacité de la méthode Data-Centric d'Arthur (S4) : l'utilisation de regex strictes couplée à l'injection de données synthétiques ciblées a permis de créer une signature sémantique extrêmement claire là où la classe était quasi-inexistante à l'origine.
* **Nuance et regard critique :** Toutefois, il est impératif de prendre du recul sur ce score absolu de 100%. Cette perfection mathématique s'explique en grande partie par la nature *synthétique* de la donnée. Les tickets ayant été générés par une IA (Gemini), ils possèdent une structure syntaxique propre, logique et stéréotypée (ex: "*I would like to request...*"), qui est extrêmement facile à capter pour DistilBERT. Dans un environnement de production réel (*in the wild*), les utilisateurs humains formulent souvent leurs demandes d'évolution de manière beaucoup plus chaotique, parfois avec du sarcasme, de la frustration ou du vocabulaire métier imprévisible. Face à des formulations humaines réelles et ambiguës, le modèle commettrait inévitablement des erreurs. Ce score de 1.00 illustre donc autant le succès de notre rééquilibrage que le léger biais de "propreté" introduit par la génération artificielle.

### B. Victoire sur la Baseline (TF-IDF)

* Lors du jalon S7, la Baseline d'Antoine plafonnait avec des scores moyens sur les classes minoritaires car le modèle TF-IDF fonctionnait en "sac de mots" (perte du contexte).
* **Analyse :** DistilBERT, grâce à son attention contextuelle (Transfer Learning), a totalement réparé cette faille. Il comprend le sens global de la phrase, confirmant nos hypothèses théoriques posées dans le Proposal.

### C. La limite du Langage : Le chevauchement Sémantique (*Overlap*)

* **Les 43 erreurs restantes :** Elles se concentrent exclusivement sur des confusions entre *Support Technique* (0.90) et *Billing and Payments* (0.83).
* **Autopsie :** Après la lecture manuelle des 43 tickets dans le fichier `qualitative_analysis_errors.csv`, un schéma très clair se dégage. La machine se trompe sur des intentions ambiguës.
* *Exemple type d'erreur :* Un client écrit *"My subscription button is broken, it won't let me pay."* * Le vrai label est souvent *Billing* (lié à un paiement), mais le modèle le classe en *Support* (car il détecte les signaux d'un bug technique : "broken", "won't let me").
* **Conclusion :** Ce n'est pas du sous-apprentissage. C'est un *chevauchement sémantique* inhérent à l'interaction humaine. Même un opérateur du service client hésiterait sur la file de routage de ce genre de ticket.
