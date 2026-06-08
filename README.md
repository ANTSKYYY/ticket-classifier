# 🎫 Automatic Customer Support Ticket Routing
> A robust NLP classifier (DistilBERT) achieving a **0.9178 Macro F1-score** through a Data-Centric approach, Adversarial Training, and Curriculum Learning.

🚀 **Test our model live:** [Interactive Web Application (XAI)](https://antskyyy.github.io/deeplearning/)  
☁️ **Hugging Face Model:** [Antskyyy/ticket-classifier](https://huggingface.co/Antskyyy/ticket-classifier)

⏱️ **Hardware & Training Time:** Model trained on an APPLE M3 CPU in approximately 30 minutes.

### 🎥 Live Web App Demo
*(Testing the XAI routing capabilities in real-time)*
<video src="https://antskyyy.github.io/deeplearning/docs/demo_site_compressed.mp4" controls="controls" muted="muted" width="100%"></video>

---

## ⚙️ Setup & Reproduction

### 📼 Video Installation Guide
<video src="https://antskyyy.github.io/deeplearning/docs/final_compressed.mp4" controls="controls" muted="muted" width="100%"></video>

<br>

```bash
# 1. Clone the repository
git clone https://github.com/ANTSKYYY/deeplearning.git
cd deeplearning

# 2. Install dependencies
make install

# ==========================================
# OPTION A: Quick Evaluation (Hugging Face)
# ==========================================
# Instantly download and evaluate our pre-trained model from the Hugging Face Hub
make eval-hub

# ==========================================
# OPTION B: Full Training Pipeline (Local)
# ==========================================
# 3. Train the full model from scratch (Phase 1 & Phase 2)
make train

# 4. Evaluate your locally trained model
make eval


```

**Expected terminal output (`make eval-hub`):**

```bash
# ☁️ Mode Hub : Téléchargement du modèle depuis HF (Antskyyy/ticket-classifier)...
# 🔤 Configuration du tokenizer par défaut : distilbert-base-uncased
# 📂 Loading and cleaning the test set...
# 🔢 Tokenization and Inference in progress...
# 
# ========================================
# 🎯 GLOBAL TEST RESULTS
# ========================================
# Macro F1-score : 0.9121 (Target: >= 0.82)
# Accuracy       : 0.9048
# 
# 📝 DETAILED REPORT:
#                     precision    recall  f1-score   support
# 
#              Support       0.89      0.91      0.90       209
#      Feature Request       1.00      1.00      1.00       102
# Billing and Payments       0.85      0.82      0.84       130
# 
#              accuracy                           0.90       441
#             macro avg       0.91      0.91      0.91       441
#          weighted avg       0.90      0.90      0.90       441
# 
# 📊 Confusion matrix saved to: results/figures/final_confusion_matrix.png
# 🚨 42 errors extracted for manual audit: results/qualitative_analysis_errors.csv


```

---

## 📝 Problem Description

In large companies, support teams waste valuable time manually sorting incoming tickets to route them to the correct departments (Billing, Technical, Feature Requests). Traditional keyword-based systems fail when faced with the ambiguity of natural language. This project implements a production-grade NLP classifier to automatically route these raw text tickets with high accuracy.

## 👥 "Ti Pip" Team and Roles

* **Arthur Le Coroller (S4):** Data-Centric approach, acquisition, regex cleaning, and stratified splitting.
* **Antoine Legall (S7):** Baseline implementation (TF-IDF + Logistic Regression) and production code structuring.
* **Arslan Tetu (S11):** Deep Learning architecture, DistilBERT fine-tuning via a robust pipeline (2-phase Curriculum Learning, Adversarial Training, and `WeightedTrainer`).
* **Evan Tangatchy (S14):** Final pipeline evaluation, metrics extraction, and qualitative analysis of residual errors (semantic overlap).
* **Enzo Dziewulski (S15):** Preparation of the final defense, defense of architectural choices, and risk management.

🔗 [View the logbooks and detailed missions for each member](https://github.com/ANTSKYYY/deeplearning/blob/main/docs/missions.md)

## 📊 Dataset (Data-Centric Approach)

As the initial dataset (339 samples) was too noisy, we made a major methodological pivot towards data engineering:

* **Volume:** Expansion to approximately 4,000 samples.
* **Target classes:** `Support` (Technical), `Feature Request` (Evolution), and `Billing and Payments` (Billing).
* **Purification:** Application of strict regular expression filters and auto-correction of labeling errors.
* **Balancing:** Synthetic generation for the minority class (*Feature Request*) and strict undersampling of the majority class (*Support*).

## 🧠 Architecture and Models

To validate the added value of Deep Learning, we compared two approaches:

1. **Baseline (Classical Machine Learning):** TF-IDF vectorization + Logistic Regression. This model fails to capture context, leading to very low scores on minority classes.
2. **Deep Model (DistilBERT + Curriculum Learning):** Fine-tuning of the `distilbert-base-uncased` Transformer architecture.

* **Innovation:** Training takes place in two phases. **Phase 1** learns concepts on easy data. **Phase 2** (Adversarial Training) injects 25% ambiguous data with a reduced learning rate to refine decision boundaries without causing catastrophic forgetting.
* **Imbalance Management:** Use of a custom weighted loss function (`WeightedTrainer`).

## 🏆 Final Results and Evaluation

The initial goal of our proposal was to achieve a Macro F1-score >= 0.82. The final model far exceeded these expectations:

* **Macro F1-score: 0.9178**
* **Overall Accuracy: 91.16%**

**Qualitative Analysis:** Detection of *Feature Requests* is perfect (F1 = 1.00). The remaining 9% of errors are located exclusively at the boundary between "Support" and "Billing", reflecting the natural ambiguity of human language (e.g., a customer reporting a "technical bug" preventing them from "paying"), where even a human operator would hesitate.

## ⚠️ Risk Management

* **Insufficient data:** Mitigated by synthetic augmentation and the Data-Centric approach.
* **Majority bias:** Mitigated by undersampling and the mathematical modification of the *Cross-Entropy Loss* via class weight injection.
* **Keyword fragility (False positives):** Mitigated by Phase 2 of training (Adversarial) forcing the model to focus on global contextual attention.

---

## Acknowledgements & DL Disclosure

As per syllabus requirements, we disclose the use of the following Deep Learning tools during this project:

* Google Gemini: Used to brainstorm and synthetically generate minority class tickets (Feature Requests) during the data augmentation phase.
* ChatGPT / Claude: Used to debug specific PyTorch/Hugging Face cache permission errors, brainstorm dataset purification strategies, and refactor our exploratory Jupyter Notebooks into production-grade.

## 📄 License

* **Source code:** This project is made available under the **[MIT License](https://github.com/ANTSKYYY/deeplearning/blob/main/LICENSE)**.
* **Dataset:** The cleaned and augmented dataset produced for this project is made available under the **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/deed.fr)**. If you use this dataset, please cite this repository: *> Ti Pip Team - "Ticket Classification Data-Centric Dataset", 2026.*
