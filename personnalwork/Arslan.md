
# 🚀 Milestone S11 : Modèle Profond (Deep Learning) - Fine-tuning de DistilBERT

**Responsable :** Arslan Tetu

**Objectif :** Entraîner (fine-tuner) le modèle `distilbert-base-uncased` pour surpasser les limites de la baseline classique (Macro F1 de 0.42) et résoudre l'incapacité du modèle précédent à détecter la classe minoritaire "Product Support".
**Prérequis :** Fichiers `train.csv` (80%), `val.csv` (10%), et `test.csv` (10%) générés par Arthur (S4).

## 1. Argumentation Théorique (Lien avec le cours DL2026)

Avant de passer à l'implémentation, il est essentiel de justifier nos choix architecturaux à la lumière du cours.

### A. Modélisation de Séquences et choix du Transformer

La baseline TF-IDF de la Phase 2 (S7) a échoué car elle traite le texte comme un "sac de mots" (bag-of-words), détruisant l'ordre et le contexte. D'après le fichier `reading-1-companion-notes-for-goodfellow-ch-10-sequence-modeling.md`, traiter des données temporelles ou ordonnées comme le texte nécessite un modèle capable de comprendre les dépendances séquentielles.
Si historiquement les Réseaux de Neurones Récurrents (RNNs) étaient la norme, le cours souligne que l'architecture **Transformer** a remplacé les RNNs en NLP. En supprimant la contrainte de récurrence grâce au mécanisme d'attention, le Transformer élimine le "goulot d'étranglement" (bottleneck) de l'encodeur. `DistilBERT` est une version compressée de cette architecture, idéale pour notre petit dataset de 339 échantillons.

### B. Mathématiques de la classification : Softmax et Log-Loss (Cross-Entropy)

Notre modèle utilise une tokenisation par sous-mots (WordPiece). L'embedding contextuel du token `[CLS]` (qui agrège le sens de la séquence) est passé dans une couche de classification linéaire suivie d'une fonction **Softmax**.

* **Pourquoi Softmax ?** D'après `reading-2-sigmoid-softmax-and-log-loss-full-derivations.md`, Softmax est la généralisation de la fonction Sigmoid pour $K \ge 2$ classes mutuellement exclusives. Elle garantit que la somme des probabilités des 3 classes vaut 1.
* **Pourquoi la Categorical Cross-Entropy ?** Le cours démontre que minimiser l'entropie croisée catégorielle équivaut à maximiser la vraisemblance (Maximum Likelihood) sous un modèle catégoriel. Surtout, la dérivation mathématique de la perte combinée au Softmax produit un gradient d'une élégance absolue : $\frac{\partial L}{\partial z} = a - y$. Cette simplification mathématique évite la saturation et rend la rétropropagation (backpropagation) extrêmement stable et efficace.

## 2. Pipeline de Fine-tuning DistilBERT (Code Python)

Le script suivant implémente l'entraînement. Il respecte à la lettre les spécifications de notre proposition (Proposal).

```python
# src/train_distilbert.py
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import f1_score, accuracy_score

# ==========================================
# 1. PRÉPARATION DES DONNÉES
# ==========================================
print("Chargement des données S4...")
train_df = pd.read_csv('data/train.csv')
val_df = pd.read_csv('data/val.csv')
test_df = pd.read_csv('data/test.csv')

# Mapping des labels en entiers pour PyTorch
label_mapping = {"Other": 0, "Product Support": 1, "Technical Support": 2}
for df in [train_df, val_df, test_df]:
    df['label'] = df['label'].map(label_mapping)

# Conversion en objets HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# ==========================================
# 2. TOKENIZATION (WordPiece)
# ==========================================
print("Tokenisation avec distilbert-base-uncased...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # Padding et troncature pour uniformiser la taille des séquences
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_tokenized = train_dataset.map(tokenize_function, batched=True)
val_tokenized = val_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)

# ==========================================
# 3. INITIALISATION DU MODÈLE
# ==========================================
print("Initialisation du modèle de classification...")
# Le modèle ajoute automatiquement la couche linéaire et la perte Cross-Entropy (log-loss)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=3
)

# Fonction de calcul des métriques
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    
    macro_f1 = f1_score(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {"macro_f1": macro_f1, "accuracy": acc}

# ==========================================
# 4. HYPERPARAMÈTRES ET ENTRAÎNEMENT
# ==========================================
print("Configuration des hyperparamètres...")
# Hyperparamètres stricts basés sur la proposition du projet
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",      # Évaluation à la fin de chaque époque
    save_strategy="epoch",
    learning_rate=3e-5,               # Dans la fourchette recommandée [2e-5, 5e-5]
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,               # Arrêt précoce possible si validation stagne
    weight_decay=0.01,                # AdamW : régularisation pour limiter l'overfitting
    warmup_steps=50,                  # Scheduler linéaire avec warmup
    load_best_model_at_end=True,      # Conserve le meilleur modèle basé sur la validation
    metric_for_best_model="macro_f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    compute_metrics=compute_metrics,
)

print("Début de l'entraînement (Fine-Tuning)...")
trainer.train()

# ==========================================
# 5. ÉVALUATION FINALE SUR LE SET DE TEST
# ==========================================
print("\n--- ÉVALUATION SUR LE SET DE TEST ---")
test_results = trainer.evaluate(test_tokenized)
print(f"Test Macro F1-score: {test_results['eval_macro_f1']:.4f}")
print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")

# Sauvegarde du modèle final
trainer.save_model("./distilbert-ticket-classifier")
print("Modèle sauvegardé dans le dossier './distilbert-ticket-classifier'")

```

## 3. Objectifs et Attentes

* **Cible (Macro F1) :** L'objectif de ce fine-tuning est d'atteindre un Macro F1-score compris entre **0.80 et 0.88**.
* **Résolution de la faille de la Baseline :** Lors du jalon S7, la régression logistique a obtenu un score de 0.00 sur la classe "Product Support". Grâce au **Transfer Learning**, DistilBERT possède déjà une compréhension riche de la langue anglaise. Le modèle ne cherchera pas de correspondances exactes de mots isolés, mais "l'intention sémantique" agrégée dans le token `[CLS]`. Nous nous attendons donc à ce que le score de la classe minoritaire décolle significativement.
* **Prochaine étape (S14 - Evan) :** Dès que l'entraînement est terminé et le modèle sauvegardé, Evan prendra le relais pour analyser qualitativement les nouvelles prédictions, générer la matrice de confusion finale et inspecter manuellement les erreurs restantes.
