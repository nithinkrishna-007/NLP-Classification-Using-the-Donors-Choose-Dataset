# NLP-Classification-Using-the-Donors-Choose-Dataset


Here is the complete project `README.md` content formatted cleanly for your project. You can copy and save this as `README.md` in your repository or download it directly:

---

# DonorsChoose Project Approval Prediction: Classical ML vs. Deep Learning

This project builds an end-to-end Machine Learning and Natural Language Processing (NLP) pipeline to predict whether teacher project proposals on DonorsChoose will be approved (`project_is_approved`). It compares traditional sparse classical models with an advanced deep learning multi-input architecture using pre-trained and custom Word2Vec embeddings.

---

## Project Overview

1. **Data Preprocessing & NLP Pipeline**: Lowercasing, HTML/special character removal, tokenization, POS tagging, and dynamic lemmatization across multiple text fields (`cleaned_titles`, `cleaned_essays`, `cleaned_summary`).
2. **Text Representation**:
* Sparse N-Gram Count / TF-IDF matrices (uni-grams, bi-grams, tri-grams).
* Dense Word2Vec embeddings (Google News pre-trained) and custom CBOW representations.


3. **Classical ML Models**: Logistic Regression and Random Forest classifiers trained on combined high-dimensional text features.
4. **Cross-Validation & Hyperparameter Tuning**: Stratified K-Fold cross-validation paired with efficient parameter optimization (`GridSearchCV`).
5. **Deep Learning (TensorFlow/Keras Functional API)**: A multi-input neural network processing multiple text embedding streams in parallel subnetworks, merging them via concatenation, and utilizing batch normalization, dropout regularization, and learning rate scheduling.
6. **Evaluation & Comparison**: Comprehensive performance reporting using Accuracy, Precision, Recall, F1-Score, Confusion Matrices, ROC curves, and Precision-Recall analysis.

---

## Repository Structure

```text
├── Preprocessed_DonorsChoose_dataset.csv  # Input dataset (hosted in Google Drive)
├── assignment.ipynb                       # Main Google Colab notebook containing code & execution
└── README.md                              # Project documentation

```

---

## Installation & Dependencies

To run this project locally or in Google Colab, ensure you have the required libraries installed:

```bash
pip install pandas numpy scikit-learn nltk gensim tensorflow seaborn matplotlib imbalanced-learn xgboost

```

---

## Running the Pipeline (Section by Section)

### 1. Data Loading & Exploration

Mount Google Drive and load the preprocessed dataset:

```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv("/content/drive/MyDrive/ict/Preprocessed_DonorsChoose_dataset.csv")

```

### 2. NLP Feature Extraction & Word2Vec

Generate sparse count matrices and document-level Word2Vec embeddings:

* Standard Mean Word2Vec
* TF-IDF Weighted Word2Vec
* Custom CBOW Word2Vec model training via Gensim

### 3. Classical Models & Hyperparameter Tuning (Crash-Free Approach)

To avoid Out-Of-Memory (OOM) crashes and long execution times on high-dimensional text features:

* Uses `StratifiedKFold` (3 splits) with memory-efficient linear solvers (`solver='saga'`).

```python
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
param_grid = {'C': [0.1, 1.0], 'solver': ['saga']}

grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=500, random_state=42, n_jobs=-1),
    param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1
)
grid_search.fit(X_train, y_train)

```

### 4. Deep Learning Multi-Input Model (TensorFlow Functional API)

Constructs parallel subnetworks for different text fields (e.g., titles and essays), merges them, and applies regularization:

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.models import Model

# Parallel inputs combined via functional API
input_titles = Input(shape=(input_dim_1,), name='titles_input')
x1 = Dropout(0.3)(BatchNormalization()(Dense(128, activation='relu')(input_titles)))

input_essays = Input(shape=(input_dim_2,), name='essays_input')
x2 = Dropout(0.3)(BatchNormalization()(Dense(128, activation='relu')(input_essays)))

merged = Concatenate()([x1, x2])
output = Dense(1, activation='sigmoid')(Dense(64, activation='relu')(merged))

dl_model = Model(inputs=[input_titles, input_essays], outputs=output)

```

---

## Model Evaluation & Performance Metrics

All models are evaluated using a hold-out test set to report:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix** (visualized via Seaborn heatmaps)
* **ROC-AUC & Precision-Recall Curves**

---

## Key Results & Analysis

* **Logistic Regression (Classical)**: Excels on high-dimensional sparse TF-IDF text features due to linear separability and fast training speeds.
* **Deep Learning (Functional API)**: Effectively captures non-linear semantic interactions across multiple distinct embedding inputs (titles + essays), though it requires careful dropout and learning rate tuning to avoid overfitting on imbalanced educational datasets.
