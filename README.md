# Credit-Card-Fraud-Detection-System
# Credit-Card-Fraud-Detection-System

#  Credit Card Fraud Detection with Machine Learning

Detecting fraudulent credit card transactions is a real-world, high-impact problem — and in this project, we’ve built a smart system to do just that!

This project uses **machine learning techniques** to identify potentially fraudulent transactions based on historical credit card usage data. We compare a simple baseline model (Logistic Regression) with an optimized approach (Random Forest + SMOTE + Scaling) to see how performance improves.

---

##  Dataset

We use the popular [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains:
- **284,807 transactions**
- **492 frauds** (0.172% of total)
- **30 numerical input features** (PCA-transformed)
- A **binary target column**: `Class` (0 = Legit, 1 = Fraud)

---

##  What This Project Covers

###  Step-by-Step Breakdown:

1. **Data Loading & Exploration**
   - Read CSV file and examine key statistics and correlations.

2. **Feature Engineering**
   - Standardize `Amount` and `Time` using `StandardScaler`.

3. **Data Splitting**
   - Train-test split using stratification to preserve class balance.

4. **Baseline Model: Logistic Regression**
   - Simple but effective model with `class_weight="balanced"` to handle class imbalance.

5. **Optimized Model: Random Forest + SMOTE**
   - Apply **SMOTE** (Synthetic Minority Oversampling) to balance classes.
   - Train a **Random Forest** for better handling of non-linear patterns.

6. **Model Evaluation**
   - Evaluate both models using:
     - Accuracy
     - Precision
     - Recall
     - F1-Score
     - AUC (Area Under ROC Curve)
   - ROC curve comparison for both models.

---

##  Why Class Imbalance Matters

Most transactions are **not fraudulent**, so a naive model could predict “legit” every time and still be 99.8% accurate — but useless.

That’s why we focus on metrics like **recall**, **F1-score**, and **AUC**, which are much more meaningful in fraud detection.

---

##  Results Summary

| Model                          | Accuracy | Recall (Fraud) | F1-Score | AUC     |
|-------------------------------|----------|----------------|----------|---------|
| Logistic Regression (Baseline)| High     | Medium         | Medium   | ~0.93   |
| Random Forest + SMOTE         | High     | **High**       | **High** | **~0.98** |

>  Our improved model catches more frauds with fewer false alarms!

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

---

##  How to Run

1. Clone the repo or upload the notebook to **Google Colab**.
2. Download the dataset from Kaggle and upload it (`creditcard.csv`).
3. Follow the cells in order — everything is modular and explained.

---




