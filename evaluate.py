import os
import json
import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Configuration HF
os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_HUB_DISABLE_TOKEN"] = "1"

from src.data import apply_cleaning, LABEL_MAPPING, ID2LABEL

def main():
    # 1. Config et Device
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # 2. Chargement Modèle (Depuis le dossier consolidé de train.py)
    MODEL_PATH = "./checkpoints/best"
    print(f"🧠 Chargement du modèle depuis : {MODEL_PATH}")
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    except Exception as e:
        print(f"❌ Erreur: Impossible de charger le modèle. Avez-vous lancé 'python train.py' en premier ? Détails : {e}")
        return
    model.eval()

    # 3. Préparation des Données de Test
    print("📂 Chargement et nettoyage du set de test...")
    test_df = pd.read_csv(config['data']['test'])
    test_df = apply_cleaning(test_df)
    
    texts = test_df['text'].tolist()
    true_labels = test_df['label'].tolist()

    print("🔢 Tokenisation et Inférence en cours...")
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=config['model']['max_len'], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    # 4. Métriques
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    accuracy = accuracy_score(true_labels, predictions)

    print("\n" + "="*40)
    print(f"🎯 RÉSULTATS GLOBAUX DU TEST")
    print("="*40)
    print(f"Macro F1-score : {macro_f1:.4f} (Cible : >= 0.82)")
    print(f"Accuracy       : {accuracy:.4f}")
    print("\n📝 RAPPORT DÉTAILLÉ :")
    print(classification_report(true_labels, predictions, target_names=list(LABEL_MAPPING.keys())))

    # Sauvegarde des métriques en JSON
    metrics = {"macro_f1": float(macro_f1), "accuracy": float(accuracy)}
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 5. Matrice de confusion
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(LABEL_MAPPING.keys()),
                yticklabels=list(LABEL_MAPPING.keys()))
    plt.title(f'Matrice de Confusion Finale\nMacro F1: {macro_f1:.4f}')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')
    plt.tight_layout()
    
    cm_path = 'results/figures/final_confusion_matrix.png'
    plt.savefig(cm_path)
    print(f"\n📊 Matrice de confusion sauvegardée sous : {cm_path}")

    # 6. Extraction des erreurs
    test_df['nom_predit'] = [ID2LABEL[p] for p in predictions]
    test_df['vrai_nom'] = [ID2LABEL[l] for l in true_labels]
    
    errors_df = test_df[test_df['vrai_nom'] != test_df['nom_predit']].copy()
    errors_path = 'results/qualitative_analysis_errors.csv'
    errors_df[['text', 'vrai_nom', 'nom_predit']].to_csv(errors_path, index=False)
    
    print(f"🚨 {len(errors_df)} erreurs extraites pour audit manuel : {errors_path}")

if __name__ == "__main__":
    main()