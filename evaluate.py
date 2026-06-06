import os
import random
import torch
import numpy as np

os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_HUB_DISABLE_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# HF Configuration
os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_HUB_DISABLE_TOKEN"] = "1"

from src.data import apply_cleaning, LABEL_MAPPING, ID2LABEL

def main():
    # 0. Command line arguments management
    parser = argparse.ArgumentParser(description="Evaluation of the ticket classification model.")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best.pt", help="Path to the checkpoint folder or file to evaluate.")
    args = parser.parse_args()

    # 1. Config and Device
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # 2. Selection and validation of the checkpoint path
    checkpoint_path = args.checkpoint
    
    # If the path targets a file (e.g., best.pt or model.safetensors), we extract its parent folder
    if os.path.isfile(checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)
    else:
        checkpoint_dir = checkpoint_path

    # Fallback folder if the specified folder does not exist or does not contain the weights
    FALLBACK_PATH = "./checkpoint"
    MODEL_PATH = checkpoint_dir

    if not os.path.exists(os.path.join(MODEL_PATH, "model.safetensors")):
        if MODEL_PATH != FALLBACK_PATH and os.path.exists(os.path.join(FALLBACK_PATH, "model.safetensors")):
            MODEL_PATH = FALLBACK_PATH
            print(f"⚠️ No 'model.safetensors' file found in {checkpoint_dir}.")
            print(f"🔄 Automatically switching to your fallback checkpoint: {MODEL_PATH}")
        else:
            print(f"❌ Error: No 'model.safetensors' file found in {MODEL_PATH}.")
            print("Please run the training ('python train.py') or check your checkpoint files.")
            return
    else:
        print(f"🧠 Loading the model from: {MODEL_PATH}")

    try:
        # Loading the tokenizer (local if available, otherwise remote/cache with the base name)
        if os.path.exists(os.path.join(MODEL_PATH, "tokenizer_config.json")):
            tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        else:
            tokenizer = DistilBertTokenizerFast.from_pretrained(config['model']['name'])
            
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    except Exception as e:
        print(f"❌ Error while loading model components: {e}")
        return
        
    model.eval()

    # 3. Test Data Preparation
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

    # 4. Metrics
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    accuracy = accuracy_score(true_labels, predictions)

    print("\n" + "="*40)
    print(f"🎯 GLOBAL TEST RESULTS")
    print("="*40)
    print(f"Macro F1-score : {macro_f1:.4f} (Target : >= 0.82)")
    print(f"Accuracy       : {accuracy:.4f}")
    print("\n📝 DETAILED REPORT:")
    print(classification_report(true_labels, predictions, target_names=list(LABEL_MAPPING.keys())))

    # Saving metrics in JSON
    os.makedirs("results/figures", exist_ok=True)
    metrics = {"macro_f1": float(macro_f1), "accuracy": float(accuracy)}
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 5. Confusion Matrix
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

    # 6. Error extraction for qualitative analysis
    test_df['predicted_name'] = [ID2LABEL[p] for p in predictions]
    test_df['true_name'] = [ID2LABEL[l] for l in true_labels]
    
    errors_df = test_df[test_df['true_name'] != test_df['predicted_name']].copy()
    errors_path = 'results/qualitative_analysis_errors.csv'
    errors_df[['text', 'true_name', 'predicted_name']].to_csv(errors_path, index=False)
    
    print(f"🚨 {len(errors_df)} errors extracted for manual audit: {errors_path}")

if __name__ == "__main__":
    main()
