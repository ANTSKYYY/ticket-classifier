import os

os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_HUB_DISABLE_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import torch
import numpy as np
import json
import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


from src.data import apply_cleaning, LABEL_MAPPING, ID2LABEL

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def main():

    parser = argparse.ArgumentParser(description="Evaluation of the ticket classification model.")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["local", "hub"], 
        default="local", 
        help="Choose 'local' for the train.py model ou 'hub' for the Huggingface Best model"
    )
    parser.add_argument(
        "--hub_repo", 
        type=str, 
        default="Antskyyy/ticket-classifier", 
        help="Repository Name on huggingface"
    )
    args = parser.parse_args()


    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


    try:
        if args.mode == "local":
            MODEL_PATH = "./checkpoints/final_model"
            if not os.path.exists(MODEL_PATH) or not os.path.exists(os.path.join(MODEL_PATH, "model.safetensors")):
                print(f"Error ! No model Found on {MODEL_PATH}.")
                print("Try to run 'make train' before !")
                return
            
            print(f"🧠 Local mode : load from {MODEL_PATH}")
            tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
            model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)

        elif args.mode == "hub":
            print(f"Hub Mode : Loading from HuggingFace:  ({args.hub_repo})...")
            
            # Utilisation du tokenizer de base (Option 2)
            base_model_name = config['model']['name']
            print(f"🔤 Config tokenizer : {base_model_name}")
            tokenizer = DistilBertTokenizerFast.from_pretrained(base_model_name)
            
            # Chargement de tes poids d'entraînement
            model = DistilBertForSequenceClassification.from_pretrained(args.hub_repo).to(DEVICE)

    except Exception as e:
        print(f"❌Error : {e}")
        return
        
    model.eval()

    # 3. Préparation des données de Test
    print("📂 Loading and cleaning the test set...")
    test_df = pd.read_csv(config['data']['test'])
    test_df = apply_cleaning(test_df)
    
    texts = test_df['text'].tolist()
    true_labels = test_df['label'].tolist()

    print("🔢 Tokenization and Inference in progress...")
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=config['model']['max_len'], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

  
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    accuracy = accuracy_score(true_labels, predictions)

    print("\n" + "="*40)
    print(f"🎯 GLOBAL TEST RESULTS")
    print("="*40)
    print(f"Macro F1-score : {macro_f1:.4f} (Target : >= 0.82)")
    print(f"Accuracy       : {accuracy:.4f}")
    print("\n📝 DETAILED REPORT:")
    print(classification_report(true_labels, predictions, target_names=list(LABEL_MAPPING.keys())))


    os.makedirs("results/figures", exist_ok=True)
    metrics = {"macro_f1": float(macro_f1), "accuracy": float(accuracy)}
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(LABEL_MAPPING.keys()),
                yticklabels=list(LABEL_MAPPING.keys()))
    plt.title(f'Final Confusion Matrix\nMacro F1: {macro_f1:.4f}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    
    cm_path = 'results/figures/final_confusion_matrix.png'
    plt.savefig(cm_path)
    print(f"\n📊 Confusion matrix saved to: {cm_path}")


    test_df['predicted_name'] = [ID2LABEL[p] for p in predictions]
    test_df['true_name'] = [ID2LABEL[l] for l in true_labels]
    
    errors_df = test_df[test_df['true_name'] != test_df['predicted_name']].copy()
    errors_path = 'results/qualitative_analysis_errors.csv'
    errors_df[['text', 'true_name', 'predicted_name']].to_csv(errors_path, index=False)
    
    print(f"🚨 {len(errors_df)} errors extracted for manual audit: {errors_path}")


if __name__ == "__main__":
    main()
