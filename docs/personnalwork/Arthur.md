# 📊 Milestone S4: Data Preparation and Data-Centric Approach

**Lead:** Arthur Le Coroller

**Objective:** Acquire, clean, augment, and purify the dataset to ensure the success of the Machine Learning (Baseline) and Deep Learning (DistilBERT) models, applying the data engineering principles covered in the DL2026 course.

## 1. Context and Initial Problem

During the initial exploration phase, our original dataset (Kaggle) contained only about 339 samples. Initial training tests showed that models plateaued at an accuracy of around 50%, even with advanced architectures like DistilBERT.

In-depth data audits revealed that the problem did not stem from the models, but from the **ground truth**:

* Vastly insufficient data volume for Deep Learning.
* Significant noise in the raw texts.
* Critical class imbalance.
* Frequent human errors in the original labeling.

Therefore, I decided to pivot to a **Data-Centric** approach, considering that data optimization was a higher priority than algorithm optimization.

## 2. Data-Centric Pipeline (Methodology and Code)

To solve these issues, I implemented a data processing pipeline in four main steps.

### Step 1: Data Augmentation and Fusion

To compensate for the lack of volume, the corpus was expanded to over 4,000 samples. **It is important to note that the vast majority of this data comes directly from the Kaggle dataset and consists of real (non-synthetic) tickets.** However, the *Feature Request* class was critically underrepresented. To specifically fill this gap, tickets were AI-generated (Gemini) based strictly on existing ticket templates. **Only the feature request category underwent synthetic augmentation.** The whole set was then merged and shuffled to avoid any learning order bias.

```python
import pandas as pd

# Loading the real dataset (Kaggle) and targeted synthetic dataset
df_old = pd.read_csv('data/new/data4.csv') # 100% real data
df_new = pd.read_csv('data/new/synthetic_feature_requests_gemini.csv') # Only Feature Requests

# Standardizing columns and cleaning null values
df_old = df_old[['queue', 'body']].rename(columns={'text': 'body', 'label': 'queue'}, errors='ignore')
df_new = df_new[['queue', 'body']]

# Merging and Shuffling
df_final = pd.concat([df_old, df_new], ignore_index=True)
df_final = df_final.dropna(subset=['queue', 'body']).drop_duplicates(subset=['body'])
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

df_final.to_csv('train_final_3_classes.csv', index=False)

```

### Step 2: Surgical Extraction (Regex N-Grams)

To ensure the integrity of the *Feature Request* class (whether real or synthetic), I designed an extractor based on strict, human-validated regular expressions. This allows the true feature requests to be mathematically isolated from the rest of the noise.

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

### Step 3: AI Label Auto-Correction

Many original tickets from Kaggle were miscategorized by human operators. I deployed a mass inference script so that a pre-trained AI model could re-evaluate the entire dataset. If the AI contradicted the human label with extreme confidence (≥ 90%), the label was automatically corrected.

```python
from transformers import pipeline

# Mass inference on the cleaned corpus
classifier = pipeline("text-classification", model="./distilbert-ticket-classifier-final", device=-1)
results = classifier(train_df['clean_text'].to_list(), truncation=True, max_length=128)

train_df['Prediction_IA'] = [res['label'] for res in results]
train_df['Confidence'] = [res['score'] for res in results]

# Overwriting obvious human errors
CONFIDENCE_THRESHOLD = 0.90
mask_to_correct = (train_df['queue'] != train_df['Prediction_IA']) & (train_df['Confidence'] >= CONFIDENCE_THRESHOLD)

train_df.loc[mask_to_correct, 'queue'] = train_df.loc[mask_to_correct, 'Prediction_IA']
train_df.drop(columns=['clean_text', 'Prediction_IA', 'Confidence']).to_csv('data/train_autocorrected.csv', index=False)

```

### Step 4: Balancing (Undersampling)

Even after augmentation, the *Support* class (100% from Kaggle) remained overrepresented. To prevent the model from developing a majority prediction bias, I applied strict undersampling to this class before the final stratified split.

```python
# Capping the majority class
support = df_final[df_final["queue"] == "Support"].sample(n=2094, random_state=42)
others = df_final[df_final["queue"] != "Support"]

# Reconstructing the balanced dataset
df_balanced = pd.concat([support, others], ignore_index=True)
print(df_balanced["queue"].value_counts())

```

## 3. Link with the DL2026 Course

The completion of this milestone rigorously validates the concepts covered in the course (Module 2, Reading 3 - *How to Draft a Feasible Project Proposal*) to de-risk the project as early as possible:

1. **Focus on Data Quality ("Data work is given equal weight to modelling"):** The course proposal emphasizes that a model's architecture cannot compensate for bad data. The shift to a *Data-Centric* approach (AI purification, strict regex) unlocked the model's performance, jumping from 50% to production-ready scores.
2. **Imbalance Management (Class Imbalance):** Without the undersampling to 2094 samples and the targeted synthetic data augmentation for *Feature Requests*, both the Baseline (Antoine) and Deep Learning (Arslan) models would have ignored the minority classes.
3. **Preparation for Modeling:** Noise elimination guarantees a high-quality sparse matrix (TF-IDF) for logistic regression, and meaningful *embeddings* for the contextual tokens (`[CLS]`) of the DistilBERT Transformer.
