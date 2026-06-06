## 📊 Milestone S14: Final Evaluation and Qualitative Error Analysis

**Lead:** Evan

**Objective:** Rigorously evaluate the final performance of the DistilBERT model (from Phase 2 with Curriculum Learning), validate the effectiveness of the overall pipeline, and conduct an in-depth qualitative analysis of the 9% residual errors.

## 1. Theoretical Context and Choice of Metrics (Link with the course)

Evaluating a machine learning model against an imbalanced dataset requires appropriate metrics.

* **Primary Metric: The Macro F1-score.**
**Global Accuracy** provides the total percentage of correct predictions. However, if a model simply predicts the "Support" class (the most frequent one) the majority of the time, its accuracy will remain artificially high. The **Macro F1-score** calculates the unweighted mean of the F1-scores for each class.
To reach our initial target (≥ 0.82), the model was mathematically forced to perform very well on the minority classes (*Feature Request* and *Billing*). It is this metric that validates the success of the `WeightedTrainer` implemented by Arslan.

## 2. Official Evaluation Pipeline (Python Code)

Here is the final evaluation script. It includes automatic detection of the latest Phase 2 checkpoint, and mapping correction (Text/Integers) to accurately extract the true errors.

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

DEVICE = torch.device("cpu") # Or "cuda"/"mps" if available


# 1. DATA AND MODEL LOADING
print("Loading the test set...")
test_df = pd.read_csv('data/test.csv')

label_mapping = {"Support": 0, "Feature Request": 1, "Billing and Payments": 2}
inverse_mapping = {v: k for k, v in label_mapping.items()}

# Automatic search for Arslan's latest checkpoint
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

# 2. GENERATING PREDICTIONS
print("Inference in progress...")
texts = test_df['text'].tolist()
# Conversion of textual labels to integers for metric calculation
true_labels = test_df['label'].map(label_mapping).tolist()

inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()


# 3. GLOBAL RESULTS AND MATRIX

macro_f1 = f1_score(true_labels, predictions, average='macro')
accuracy = accuracy_score(true_labels, predictions)

print("\n" + "="*40)
print(f"GLOBAL RESULTS (PHASE 2)")
print("="*40)
print(f"Macro F1-score : {macro_f1:.4f} (Target : >= 0.82)")
print(f"Accuracy       : {accuracy:.4f}")
print("\nDETAILED REPORT:")
print(classification_report(true_labels, predictions, target_names=list(label_mapping.keys())))

# Confusion Matrix Generation
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(label_mapping.keys()),
            yticklabels=list(label_mapping.keys()))
plt.title(f'Final Confusion Matrix (DistilBERT)\nMacro F1: {macro_f1:.4f}')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
os.makedirs('docs/images', exist_ok=True)
plt.savefig('docs/images/final_confusion_matrix.png')

# 4. TRUE ERRORS EXTRACTION
# Re-translating predictions into text
test_df['predicted_label_text'] = [inverse_mapping[p] for p in predictions]
# Isolating only the rows where the text differs
errors_df = test_df[test_df['label'] != test_df['predicted_label_text']].copy()
errors_df = errors_df.rename(columns={'label': 'true_name', 'predicted_label_text': 'predicted_name'})

errors_path = 'data/qualitative_analysis_errors.csv'
errors_df[['text', 'true_name', 'predicted_name']].to_csv(errors_path, index=False)
print(f"{len(errors_df)} true errors extracted for manual audit : {errors_path}")


```

## 3. Results and Qualitative Analysis (Error Autopsy)

Our initial goal was to achieve a Macro F1-score of at least 0.82. The obtained results shatter this expectation with a **Macro F1-score of 0.9103** and an **Accuracy of 90.27%**.

Out of the 442 blind test tickets, only **43 errors** were recorded. My role is to qualitatively analyze these failures to demonstrate that they are not due to an architectural flaw, but to the ambiguity of natural language.

### A. Perfection on "Feature Request": Data-Centric Success and Synthetic Bias

* **F1 Score: 1.00 (100%)**
* **Analysis:** The model literally makes **zero errors** on this class. On one hand, this proves the effectiveness of Arthur's Data-Centric method (S4): the use of strict regex coupled with the injection of targeted synthetic data made it possible to create an extremely clear semantic signature where the class was originally almost non-existent.
* **Nuance and critical perspective:** However, it is imperative to step back from this absolute score of 100%. This mathematical perfection is largely explained by the *synthetic* nature of the data. Because the tickets were AI-generated (Gemini), they possess a clean, logical, and stereotyped syntactic structure (e.g., "*I would like to request...*"), which is extremely easy for DistilBERT to capture. In a real production environment (*in the wild*), human users often formulate their feature requests in a much more chaotic manner, sometimes with sarcasm, frustration, or unpredictable business vocabulary. Faced with real and ambiguous human phrasing, the model would inevitably make errors. This 1.00 score therefore illustrates both the success of our rebalancing and the slight "cleanliness" bias introduced by artificial generation.

### B. Victory over the Baseline (TF-IDF)

* During the S7 milestone, Antoine's Baseline plateaued with average scores on minority classes because the TF-IDF model operated as a "bag of words" (loss of context).
* **Analysis:** DistilBERT, thanks to its contextual attention (Transfer Learning), completely repaired this flaw. It understands the overall meaning of the sentence, confirming our theoretical hypotheses laid out in the Proposal.

### C. The Limit of Language: Semantic Overlap

* **The remaining 43 errors:** They are exclusively concentrated on confusions between *Technical Support* (0.90) and *Billing and Payments* (0.83).
* **Autopsy:** After a manual review of the 43 tickets in the `qualitative_analysis_errors.csv` file, a very clear pattern emerges. The machine stumbles on ambiguous intents.
* *Typical error example:* A customer writes *"My subscription button is broken, it won't let me pay."* * The true label is often *Billing* (related to a payment), but the model classifies it under *Support* (because it detects signals of a technical bug: "broken", "won't let me").
* **Conclusion:** This is not underfitting. It is a *semantic overlap* inherent to human interaction. Even a customer service operator would hesitate on the routing queue for this kind of ticket.
