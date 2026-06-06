import os


os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_HUB_DISABLE_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml
import torch
import random
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
args = parser.parse_args()
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification, 
    TrainingArguments, 
    EarlyStoppingCallback
)

from src.data import apply_cleaning, make_hf_dataset, ID2LABEL, LABEL_MAPPING
from src.model import WeightedTrainer
from src.utils import compute_metrics

def main():
    # 1. Chargement de la configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    SEED = config['training']['seed']
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️  Device : {DEVICE}")

    # 2. Chargement et préparation des données (Phase 1)
    print("\n📂 Chargement des données normales...")
    train_df = pd.read_csv(config['data']['train_augmented']).dropna(subset=['text'])
    val_df   = pd.read_csv(config['data']['val']).dropna(subset=['text'])
    
    # Undersampling de la classe Support
    col = 'queue' if 'queue' in train_df.columns else 'label'
    df_support  = train_df[train_df[col] == 'Support']
    df_customer = train_df[train_df[col] == 'Customer Service']
    df_billing  = train_df[train_df[col] == 'Billing and Payments']
    
    if len(df_support) > 2000:
        df_support = df_support.sample(n=2000, random_state=SEED)
        train_df = pd.concat([df_support, df_customer, df_billing]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    train_df = apply_cleaning(train_df)
    val_df   = apply_cleaning(val_df)
    train_df_normal = train_df.copy()

    # 3. Chargement des données adversariales (Phase 2)
    print("\n⚔️  Préparation des données adversariales...")
    adv_df = pd.read_csv(config['data']['adversarial'])
    adv_df = apply_cleaning(adv_df)
    
    n_adv_target = len(train_df_normal) // 3
    adv_oversampled = adv_df.sample(n=n_adv_target, replace=True, random_state=SEED)
    train_df_full = pd.concat([train_df_normal, adv_oversampled]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # 4. Tokenisation
    print("\n🔤 Initialisation du tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(config['model']['name'])
    
    train_dataset_normal = make_hf_dataset(train_df_normal, tokenizer, config['model']['max_len'])
    train_dataset_full   = make_hf_dataset(train_df_full, tokenizer, config['model']['max_len'])
    val_dataset          = make_hf_dataset(val_df, tokenizer, config['model']['max_len'])

    # 5. Modèle
    model = DistilBertForSequenceClassification.from_pretrained(
        config['model']['name'], 
        num_labels=config['model']['num_labels'], 
        id2label=ID2LABEL, 
        label2id=LABEL_MAPPING
    ).to(DEVICE)

    # ==========================================
    # 🎓 PHASE 1: Curriculum Learning (Normales)
    # ==========================================
    print("\n🎓 Démarrage Phase 1 (Données faciles)...")
    cw_normal = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=train_df_normal['label'].values)
    cw_normal_tensor = torch.tensor(cw_normal, dtype=torch.float).to(DEVICE)

    args_p1 = TrainingArguments(
        output_dir="./checkpoints/phase1",
        num_train_epochs=config['training']['phase1']['epochs'],
        per_device_train_batch_size=config['training']['phase1']['batch_size'],
        learning_rate=float(config['training']['phase1']['learning_rate']),
        warmup_ratio=config['training']['phase1']['warmup_ratio'],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        use_cpu=(DEVICE.type == "cpu"),
        report_to="none"
    )

    trainer_p1 = WeightedTrainer(
        model=model, args=args_p1, train_dataset=train_dataset_normal, eval_dataset=val_dataset,
        compute_metrics=compute_metrics, class_weights=cw_normal_tensor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer_p1.train()

    # ==========================================
    # 🔥 PHASE 2: Curriculum Learning (Adversarial)
    # ==========================================
    print("\n🔥 Démarrage Phase 2 (+ Adversarial)...")
    cw_full = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=train_df_full['label'].values)
    cw_full_tensor = torch.tensor(cw_full, dtype=torch.float).to(DEVICE)

    args_p2 = TrainingArguments(
        output_dir="./checkpoints/phase2",
        num_train_epochs=config['training']['phase2']['epochs'],
        per_device_train_batch_size=config['training']['phase2']['batch_size'],
        learning_rate=float(config['training']['phase2']['learning_rate']),
        warmup_ratio=config['training']['phase2']['warmup_ratio'],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        use_cpu=(DEVICE.type == "cpu"),
        report_to="none"
    )

    trainer_p2 = WeightedTrainer(
        model=model, args=args_p2, train_dataset=train_dataset_full, eval_dataset=val_dataset,
        compute_metrics=compute_metrics, class_weights=cw_full_tensor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer_p2.train()

    # Sauvegarde du modèle final prêt pour l'évaluation
    final_model_path = "./checkpoints/best"
    trainer_p2.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\n✅ Entraînement terminé ! Modèle final sauvegardé dans {final_model_path}")

if __name__ == "__main__":
    main()
