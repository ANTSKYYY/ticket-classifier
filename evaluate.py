import os


os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_HUB_DISABLE_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

    # 2. Gestion et sélection du checkpoint (Dossier principal vs Dossier de secours)
    PRIMARY_PATH = "./checkpoints/best"
    FALLBACK_PATH = "./checkpoints/github"
    MODEL_PATH = PRIMARY_PATH

    # Vérification de l'existence du fichier de poids dans le dossier principal
    if not os.path.exists(os.path.join(PRIMARY_PATH, "model.safetensors")):
        if os.path.exists(os.path.join(FALLBACK_PATH, "model.safetensors")):
            MODEL_PATH = FALLBACK_PATH
            print(f"⚠️ Aucun entraînement trouvé dans {PRIMARY_PATH}.")
            print(f"🔄 Bascule automatique sur votre checkpoint de sauvegarde : {MODEL_PATH}")
        else:
            print(f"❌ Erreur: Aucun fichier 'model.safetensors' trouvé dans {PRIMARY_PATH} ni dans {FALLBACK_PATH}.")
            print("Veuillez lancer l'entraînement ('python train.py') ou placer vos fichiers dans le dossier './checkpoint/'.")
            return
    else:
        print(f"🧠 Chargement du modèle depuis l'entraînement récent : {MODEL_PATH}")

    try:
        # Chargement du tokenizer (local si disponible, sinon distant/cache avec le nom de base)
        if os.path.exists(os.path.join(MODEL_PATH, "tokenizer_config.json")):
            tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        else:
            tokenizer = DistilBertTokenizerFast.from_pretrained(config['model']['name'])
            
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    except Exception as e:
        print(f"❌ Erreur lors du chargement des composants du modèle : {e}")
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
    os.makedirs("results/figures", exist_ok=True)
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

    # 6. Extraction des erreurs pour l'analyse qualitative
    test_df['nom_predit'] = [ID2LABEL[p] for p in predictions]
    test_df['vrai_nom'] = [ID2LABEL[l] for l in true_labels]
    
    errors_df = test_df[test_df['vrai_nom'] != test_df['nom_predit']].copy()
    errors_path = 'results/qualitative_analysis_errors.csv'
    errors_df[['text', 'vrai_nom', 'nom_predit']].to_csv(errors_path, index=False)
    
    print(f"🚨 {len(errors_df)} erreurs extraites pour audit manuel : {errors_path}")

if __name__ == "__main__":
    main()