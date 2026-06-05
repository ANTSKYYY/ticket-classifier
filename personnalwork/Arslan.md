# 🚀 Milestone S11 : Modèle Profond (Deep Learning) - Fine-tuning Avancé de DistilBERT

**Responsable :** Arslan Tetu

**Objectif :** Entraîner (fine-tuner) le modèle `distilbert-base-uncased` via une architecture d'entraînement avancée (Curriculum Learning + Adversarial Training) pour surpasser la baseline classique et garantir la robustesse du modèle face aux données ambiguës.
**Prérequis :** Fichiers purifiés `train_augmented.csv`, `val.csv`, et `test.csv` générés par le pipeline Data-Centric (S4).

## 1. Argumentation Théorique (Lien avec le cours DL2026)

L'implémentation de ce pipeline repose sur plusieurs concepts théoriques majeurs abordés dans le cours.

### A. Modélisation de Séquences et choix du Transformer

Contrairement à la baseline TF-IDF qui traite le texte comme un "sac de mots", détruisant le contexte, l'architecture **Transformer** comprend les dépendances séquentielles grâce au mécanisme d'attention (cf. `reading-1-companion-notes-for-goodfellow-ch-10-sequence-modeling.md`). `DistilBERT` est une version compressée idéale pour notre cas d'usage, permettant un Transfer Learning efficace.

### B. Mathématiques : Softmax et Cross-Entropy Pondérée

L'embedding contextuel du token `[CLS]` est passé dans une couche linéaire suivie d'un **Softmax**. Comme vu dans `reading-2-sigmoid-softmax-and-log-loss-full-derivations.md`, la minimisation de la *Categorical Cross-Entropy* garantit un gradient stable ($\frac{\partial L}{\partial z} = a - y$).
Cependant, pour contrer le déséquilibre des classes restant, nous utilisons une fonction de perte personnalisée (`WeightedTrainer`) qui injecte des poids mathématiques calculés dynamiquement directement dans la fonction de perte.

### C. Curriculum Learning & Adversarial Training

Pour rendre le modèle robuste sans détruire ses apprentissages, l'entraînement est scindé en deux phases :

1. **Phase 1 (Apprentissage standard) :** Le modèle apprend à construire ses représentations sémantiques sur des données "faciles" et propres.
2. **Phase 2 (Adversarial Training) :** Injection de données "difficiles" (bruitées, ambiguës) avec un ratio de 1:3. Pour éviter **l'oubli catastrophique** (*catastrophic forgetting*) des patterns appris en Phase 1, le taux d'apprentissage (learning rate) est drastiquement réduit (de 1.5e-5 à 5e-6).

## 2. Configuration de l'Environnement (Prévention des erreurs)

Avant de lancer le pipeline, nous avons mis en place une configuration stricte des variables d'environnement pour forcer la gestion locale du cache et éviter les erreurs de permission (Token bloquant) très fréquentes avec l'API Hugging Face.

```python
import os

# 1. Force Hugging Face à télécharger dans le projet (évite de saturer le disque global)
os.environ["HF_HOME"] = "./hf_cache"

# 2. Force Hugging Face à ignorer le fichier de token local (qui bloque souvent l'accès)
os.environ["HF_HUB_DISABLE_TOKEN"] = "1"

print("Dossier de cache configuré sur :", os.environ["HF_HOME"])
print("Vérification du token désactivée pour éviter l'erreur de permission.")

# Optimisations matérielles
os.environ["TOKENIZERS_PARALLELISM"]      = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

```

## 3. Pipeline de Fine-tuning DistilBERT (Code Python)

Voici l'implémentation complète du pipeline à deux phases, incluant la surcharge de la fonction de perte et l'injection de la donnée adversariale.

```python
# src/train_distilbert_curriculum.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re
import warnings
warnings.filterwarnings("ignore")

if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = lambda: False

from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cpu")


# 1. WeightedTrainer PARAMÉTRISÉ (Gestion du déséquilibre)

class WeightedTrainer(Trainer):
   
    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights 

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss    = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),
    }

# 2. CHARGEMENT ET NETTOYAGE DES DONNÉES

print("\nChargement des données...")
train_df = pd.read_csv('data/train_augmented.csv')
val_df   = pd.read_csv('data/val.csv')
test_df  = pd.read_csv('data/test.csv')

TARGET_COL = 'queue' if 'queue' in train_df.columns else 'label'

# Undersampling classe majoritaire
df_support  = train_df[train_df[TARGET_COL] == 'Support']
df_customer = train_df[train_df[TARGET_COL] == 'Customer Service']
df_billing  = train_df[train_df[TARGET_COL] == 'Billing and Payments']

if len(df_support) > 2000:
    df_support = df_support.sample(n=2000, random_state=SEED)
    train_df   = pd.concat([df_support, df_customer, df_billing]).sample(frac=1, random_state=SEED).reset_index(drop=True)

# Regex de nettoyage avancées 
GREETINGS_RE   = re.compile(r'^(hello|hi|dear customer|hey)[,\s]+', flags=re.IGNORECASE)
SIGNATURE_RE   = re.compile(r'(sincerely|best regards|warm regards|thank you for your time).*', flags=re.IGNORECASE | re.DOTALL)
SOFTWARE_RE    = re.compile(r'\b(microsoft dynamics 365|laravel 8|node\.js|elasticsearch|avid pro tools|microsoft office 2021|hadoop|davinci resolve|asana)\b', flags=re.IGNORECASE)

def clean_text_bert(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = GREETINGS_RE.sub('', text)
    text = SIGNATURE_RE.sub('', text)
    text = SOFTWARE_RE.sub('SOFTWARE_PRODUCT', text)
    return re.sub(r'\s+', ' ', text).strip()

label_mapping = {"Support": 0, "Feature Request": 1, "Billing and Payments": 2}

def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['text'] = df['text'].apply(clean_text_bert)
    col = 'queue' if 'queue' in df.columns else 'label'
    df['label'] = df[col].map(label_mapping)
    df.dropna(subset=['text', 'label'], inplace=True)
    df['label'] = df['label'].astype(int)
    return df[df['text'].str.split().str.len() > 2].reset_index(drop=True)

train_df = apply_cleaning(train_df)
val_df   = apply_cleaning(val_df)
test_df  = apply_cleaning(test_df)

# 3. ADVERSARIAL DATA — Injection (Ratio 1:3)

print("\nChargement des adversarial examples...")
adv_df = apply_cleaning(pd.read_csv('data/adversarial_tickets.csv'))

# Oversampling pour atteindre la cible 
n_adv_target    = len(train_df) // 3
adv_oversampled = adv_df.sample(n=n_adv_target, replace=True, random_state=SEED)

# Création des deux corpus pour le Curriculum Learning
train_df_normal = train_df.copy()  # Phase 1 : données faciles
train_df_full   = pd.concat([train_df, adv_oversampled]).sample(frac=1, random_state=SEED).reset_index(drop=True) # Phase 2


# 4. TOKENISATION
MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def make_hf_dataset(df: pd.DataFrame) -> Dataset:
    ds = Dataset.from_pandas(df[['text', 'label']], preserve_index=False).map(
        lambda b: tokenizer(b['text'], padding='max_length', truncation=True, max_length=128), batched=True
    ).rename_column('label', 'labels')
    ds.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    return ds

train_dataset_normal = make_hf_dataset(train_df_normal)
train_dataset_full   = make_hf_dataset(train_df_full)
val_dataset          = make_hf_dataset(val_df)
test_dataset         = make_hf_dataset(test_df)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label={v: k for k, v in label_mapping.items()}, label2id=label_mapping
).to(DEVICE)

# 5. CURRICULUM LEARNING — PHASE 1
print("\nCURRICULUM — PHASE 1 : Données normales (faciles)")
cw_normal = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=train_df_normal['label'].values)
cw_normal_tensor = torch.tensor(cw_normal, dtype=torch.float).to(DEVICE)

args_phase1 = TrainingArguments(
    output_dir="./curriculum-phase1", num_train_epochs=2, per_device_train_batch_size=16,
    learning_rate=1.5e-5, weight_decay=0.05, warmup_ratio=0.1,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="macro_f1", use_cpu=True, report_to="none"
)

trainer_phase1 = WeightedTrainer(
    model=model, args=args_phase1, train_dataset=train_dataset_normal, eval_dataset=val_dataset,
    compute_metrics=compute_metrics, class_weights=cw_normal_tensor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)
trainer_phase1.train()
trainer_phase1.save_model("./curriculum-phase1-best")


# 6. CURRICULUM LEARNING — PHASE 2 (Normal + Adversarial)

print("\nCURRICULUM — PHASE 2 : + Adversarial examples (durs)")
cw_full = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=train_df_full['label'].values)
cw_full_tensor = torch.tensor(cw_full, dtype=torch.float).to(DEVICE)

args_phase2 = TrainingArguments(
    output_dir="./curriculum-phase2-final", num_train_epochs=2, per_device_train_batch_size=16,
    learning_rate=5e-6, # LR réduit pour éviter le catastrophic forgetting
    weight_decay=0.05, warmup_ratio=0.05,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="macro_f1", use_cpu=True, report_to="none"
)

trainer_phase2 = WeightedTrainer(
    model=model, args=args_phase2, train_dataset=train_dataset_full, eval_dataset=val_dataset,
    compute_metrics=compute_metrics, class_weights=cw_full_tensor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)
trainer_phase2.train()

# 7. ÉVALUATION FINALE
test_output = trainer_phase2.predict(test_dataset)
pred_labels = np.argmax(test_output.predictions, axis=-1)
true_labels = test_output.label_ids

print(f"\nMacro F1-score : {f1_score(true_labels, pred_labels, average='macro'):.4f}")
print(classification_report(true_labels, pred_labels, target_names=list(label_mapping.keys())))

```

## 4. Objectifs et Attentes

* **Robustesse accrue :** L'objectif de ce fine-tuning par Curriculum Learning est d'apprendre au modèle à ne pas sur-réagir à du vocabulaire trompeur (géré par la Phase 2 avec les données adversariales).
* **Résolution de la faille de la Baseline :** Le calcul de poids personnalisé via le `WeightedTrainer` permet de pénaliser beaucoup plus lourdement les erreurs sur la classe minoritaire (*Feature Request* / *Product Support*) et de ne pas la sacrifier au profit de la classe majoritaire.
* **Prochaine étape (S14 - Evan) :** Une fois le modèle `curriculum-phase2-final` exporté, Evan génèrera la matrice de confusion et l'extraction des derniers faux positifs restants afin d'en comprendre l'origine sémantique.
