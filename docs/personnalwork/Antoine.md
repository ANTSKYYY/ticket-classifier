# 🚀 Milestone S7: Base Model (Baseline) - TF-IDF & Logistic Regression

**Lead:** Antoine Legall

**Objective:** Implement the end-to-end base model (*baseline*) using TF-IDF vectorization combined with Logistic Regression.
**Prerequisites:** The data was cleaned and split (80% Train, 10% Validation, 10% Test) during the previous phase (S4) by Arthur.

## 1. Baseline Pipeline (Python Code)

The script below loads the prepared data, extracts textual features (TF-IDF), and trains a Logistic Regression model. The evaluation focuses on the **Macro F1-score**, the chosen metric to handle class imbalance.

```python
# src/run_baseline.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA LOADING (Arthur's Split)
# ==========================================
print("Loading data...")
train_df = pd.read_csv('data/train.csv')
val_df = pd.read_csv('data/val.csv')
test_df = pd.read_csv('data/test.csv')

# Separating features (X) and labels (y)
X_train, y_train = train_df['text'].fillna(""), train_df['label']
X_val, y_val = val_df['text'].fillna(""), val_df['label']
X_test, y_test = test_df['text'].fillna(""), test_df['label']

# ==========================================
# 2. TF-IDF VECTORIZATION
# ==========================================
print("Extracting features with TF-IDF...")
# Extracting unigrams and bigrams, ignoring stopwords
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# ==========================================
# 3. LOGISTIC REGRESSION TRAINING
# ==========================================
print("Training Logistic Regression...")
# class_weight='balanced' handles the class imbalance observed during EDA
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train_tfidf, y_train)

# ==========================================
# 4. EVALUATION (Validation & Test)
# ==========================================
print("\n--- EVALUATION ON THE VALIDATION SET ---")
y_val_pred = model.predict(X_val_tfidf)
val_macro_f1 = f1_score(y_val, y_val_pred, average='macro')
print(f"Validation Macro F1-score: {val_macro_f1:.4f}")

print("\n--- EVALUATION ON THE TEST SET ---")
y_test_pred = model.predict(X_test_tfidf)
test_macro_f1 = f1_score(y_test, y_test_pred, average='macro')
print(f"Test Macro F1-score: {test_macro_f1:.4f}")

print("\nDetailed classification report (Test):")
print(classification_report(y_test, y_test_pred))

# ==========================================
# 5. CONFUSION MATRIX
# ==========================================
cm = confusion_matrix(y_test, y_test_pred, labels=model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title(f'Confusion Matrix - Baseline TF-IDF + LR\nMacro F1: {test_macro_f1:.2f}')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig('docs/images/baseline_confusion_matrix.png')
print("Confusion matrix saved to 'docs/images/baseline_confusion_matrix.png'.")


```

## 2. Theoretical Argumentation (Link with DL2026 course)

In the context of the course, it is essential to justify the presence of this Baseline before moving on to Deep Learning architectures.

### A. Why start with Classical Machine Learning?

As explained in the course reading `reading-2-deep-learning-vs-classical-machine-learning.md`, Deep Learning is not always the default solution.

* **Data Size:** Classical Machine Learning is particularly effective on small datasets. Our dataset contains only 339 samples, which is well below the tens of thousands of samples typically required to train a high-performing deep model from scratch.
* **Manual vs. Learned Feature Engineering:** Unlike Deep Learning, which learns its own hierarchical representations from raw data, the classical approach requires feature engineering. Here, we use the TF-IDF algorithm, which creates a **sparse lexical representation**, assigning a weight to words based on their frequency in the document and their rarity across the overall corpus.
* **The Role of the Baseline:** The course explicitly states that one must *always train a classical baseline* to verify that the deep model (DistilBERT) provides real added value. If the deep model is not better than logistic regression, it means the deep model is broken, not better.

### B. Logistic Regression as a Primitive Neural Network

According to `reading-1-logistic-regression-as-a-neural-network-companion-to-d.md`, Logistic Regression can be seen as the smallest possible neural network (a single-neuron network with no hidden layer).

* **Mathematical Structure:** The model performs a linear transformation ($z = w \cdot x + b$) followed by a non-linear activation function (Softmax, as we have multi-class classification).
* **Loss and Interpretability:** The model minimizes a **Cross-Entropy** (log-loss) loss function, giving it very good convergence properties (the problem is convex). Furthermore, unlike opaque deep networks, this baseline is highly interpretable (the weights $w$ tell us exactly which words influence which category).

## 3. Objectives and Expected Results

* **Target:** As defined in the Proposal, the goal for this baseline is to achieve a **Macro F1-score between 0.65 and 0.75**.
* **Next Step:** Once this score is achieved, the Baseline value will be frozen. The next milestone (S11 - Arslan) will consist of applying a pre-trained language model (DistilBERT) to extract contextual embeddings (rather than the sparse TF-IDF embeddings) in an attempt to beat this limit and reach a 0.80 - 0.88 Macro F1-score.

## 4. Analysis of Baseline Results

The results obtained on the test set (Macro F1-score of 0.42) are lower than our theoretical target (0.65 - 0.75). The analysis of the classification report highlights a major failure:

* **Inability to detect the minority class:** The model obtains an F1-score of 0.00 on the `Product Support` class.
* **Overfitting:** The significant gap between validation (0.64) and test (0.42) shows that the TF-IDF approach struggles to generalize on such a small dataset (339 samples).

**S7 Conclusion:** This classical baseline demonstrates that simple word occurrence (TF-IDF) fails to capture the semantic nuances required to route ambiguous tickets, especially for underrepresented classes. This fully validates the need to move to the next phase (S11): using a pre-trained contextual language model (DistilBERT) via Transfer Learning.
