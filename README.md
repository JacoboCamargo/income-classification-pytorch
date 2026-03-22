# Income Prediction with Neural Networks
### UCI Adult Census Dataset · PyTorch · Binary Classification

> **Can demographic and socioeconomic data predict whether someone earns over $50K/year?**  
> This project answers that question by comparing a logistic regression baseline against a regularized MLP — trained on 48,000+ census records from the UCI Adult dataset.

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.857 | 0.738 | 0.614 | 0.670 | 0.908 |
| **MLP w/ Regularization** | **0.811** | **0.566** | **0.864** | **0.684** | **0.914** |

**Key finding:** The MLP detects 86.4% of high-income individuals (recall) vs 61.4% for logistic regression — a 25-point improvement — at the cost of lower precision. Higher ROC-AUC (0.914 vs 0.908) confirms better overall class separation. Model choice depends on the cost of false negatives vs false positives.

---

## Architecture

```
Input (features after OHE + StandardScaler)
    └── Linear(n_features → 256) + ReLU + Dropout(0.5)
    └── Linear(256 → 128) + ReLU + Dropout(0.5)
    └── Linear(128 → 64) + ReLU + Dropout(0.5)
    └── Linear(64 → 1) + Sigmoid
```

**Training config:**
- Optimizer: Adam · LR: 0.001 · Batch size: 128
- Loss: BCEWithLogitsLoss with `pos_weight` (handles class imbalance)
- Regularization: Dropout(0.5) + Weight decay + EarlyStopping(patience=10)

---

## What This Project Covers

**1. Data Pipeline (no leakage)**
- Dataset: [UCI Adult Census Income](https://archive.ics.uci.edu/dataset/2/adult) (1994 U.S. Census, ~48K records)
- Missing values in `workclass`, `occupation`, `native-country` imputed as `Unknown` (preserves signal)
- Categorical features: One-Hot Encoding fitted on train only (`handle_unknown='ignore'`)
- Numerical features: StandardScaler fitted on train only
- Split: `adult.data` → train | `adult.test` → 50/50 stratified → validation + test

**2. Baseline Model**
- Logistic Regression trained and evaluated across train/val/test
- Used as benchmark for all subsequent neural network experiments

**3. Neural Network Experiments**
- 5+ MLP configurations tested varying depth, width, learning rate, batch size
- Loss vs. epoch curves generated for each experiment
- Overfitting detected and addressed via Dropout + Weight decay + EarlyStopping

**4. Analysis**
- Compared regularized vs unregularized MLP (clear overfitting in unregularized)
- Compared MLP vs logistic regression on all binary classification metrics
- Discussed threshold selection: default 0.5 suboptimal — Precision-Recall curve recommended

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/JacoboCamargo/income-classification-pytorch.git
cd Parcial-2-IA

# Install dependencies
pip install torch pandas scikit-learn matplotlib numpy

# Open notebook
jupyter notebook notebook.ipynb
# Or open directly in Google Colab
```

Run all cells top to bottom. Final cell outputs best MLP metrics on the test set.

---

## Tech Stack

`Python` `PyTorch` `scikit-learn` `pandas` `NumPy` `Matplotlib` `Google Colab`

---

## Authors

**Jacobo Camargo** · [linkedin.com/in/jacobocamargo](https://linkedin.com/in/jacobocamargo)  
**Nicolás Rich**

*Universidad de los Andes — Applied Artificial Intelligence · 2025*
