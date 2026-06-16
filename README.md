# Game Genre Classification from Synopsis

Classifying video game genres from their English synopsis (short description) using three different approaches and comparing them:

1. Single-label classification with classical machine learning
2. Multi-label classification with classical machine learning
3. Multi-label classification with a fine-tuned Transformer (DistilBERT)

The goal of this research is to evaluate how well a game's genre can be predicted from its textual synopsis, and to compare a classical TF-IDF based pipeline against a contextual Transformer model.

Language versions: English (this file) and Indonesian ([README.id.md](README.id.md)).

---

## Table of Contents

1. [Background and Objective](#1-background-and-objective)
2. [Dataset](#2-dataset)
3. [Repository Structure](#3-repository-structure)
4. [Methodology](#4-methodology)
5. [Comparison of the Three Approaches](#5-comparison-of-the-three-approaches)
6. [Detailed Explanation of Each File](#6-detailed-explanation-of-each-file)
7. [How to Run](#7-how-to-run)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Notes and Limitations](#9-notes-and-limitations)

---

## 1. Background and Objective

Game stores such as Steam describe each title with a short synopsis. This research investigates whether the genre of a game can be predicted automatically from that synopsis text using Natural Language Processing and machine learning.

The main objective is single-label genre classification (predicting the primary genre of a game). Two additional approaches, multi-label classification and a Transformer-based model, are included as comparisons to study how the problem behaves when a game can belong to several genres at once and when a modern contextual language model is used.

---

## 2. Dataset

- Source: Steam Games Dataset by FronkonGames (Kaggle): https://www.kaggle.com/datasets/fronkongames/steam-games-dataset
- The dataset describes Steam games, including their textual descriptions and genre tags.
- Only English-language synopses are used.

### Important: the raw data must be repaired first

The raw dataset contains structural errors: some rows have an inconsistent number of columns (fields shift out of alignment), which makes a direct CSV read fail. The column mismatch problem is documented in `Colomn Dataset tidak sesuai.pdf`.

Before running any model notebook, run:

```
fix_steam_games_csv_from_json.ipynb
```

This notebook reconstructs the data from the source and produces a clean, properly aligned file (`games_fixed.csv`) that the model notebooks load. Do not feed the raw Kaggle file directly into the model notebooks.

---

## 3. Repository Structure

| File | Description |
|------|-------------|
| `fix_steam_games_csv_from_json.ipynb` | Data repair and cleaning. Fixes the misaligned columns of the raw dataset and produces the clean `games_fixed.csv`. Run this first. |
| `TEST03_2.ipynb` | Single-label classification (primary genre). Classical ML pipeline (TF-IDF + Naive Bayes / KNN / Linear SVM). This is the main approach of the research. |
| `TEST03_MULTILABEL.ipynb` | Multi-label classification with classical ML (One-vs-Rest). Same feature pipeline as the single-label notebook. |
| `TEST04_TRANSFORMER_multilabel_2.ipynb` | Multi-label classification using a fine-tuned Transformer (DistilBERT). |
| `Colomn Dataset tidak sesuai.pdf` | Documentation of the column-mismatch problem in the raw dataset. |

---

## 4. Methodology

### 4.1 Text Preprocessing (classical ML notebooks)

The classical notebooks apply standard NLP preprocessing to the synopsis text:

- Case folding (lowercasing)
- Tokenization
- Stopword removal (English)
- Stemming (Porter Stemmer)
- Removal of very short tokens

Three preprocessing scenarios are compared:

- S1: stopword removal only
- S2: stemming only
- S3: stopword removal and stemming

### 4.2 Feature Extraction (classical ML notebooks)

Text is converted to numerical features with TF-IDF using the following configuration (identical in both classical notebooks):

- `max_features = 20000`
- `ngram_range = (1, 2)`
- `sublinear_tf = True`
- `min_df = 3`
- `max_df = 0.95`

### 4.3 Models (classical ML notebooks)

Three algorithms are trained and compared:

- Multinomial Naive Bayes (`alpha = 0.1`)
- K-Nearest Neighbors (`n_neighbors = 11`, cosine distance)
- Linear SVM (`class_weight = 'balanced'`, `max_iter = 5000`)

The best model is selected automatically based on the highest F1-score.

### 4.4 Transformer Model

The Transformer notebook fine-tunes DistilBERT (`distilbert-base-uncased`) for multi-label genre prediction:

- Maximum sequence length: 256 tokens
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 4
- Light text cleaning only (no stopword removal or stemming, since the model uses contextual embeddings)
- Class imbalance handled with undersampling
- Per-genre decision thresholds are tuned on a validation set

---

## 5. Comparison of the Three Approaches

| Aspect | Single-label (`TEST03_2`) | Multi-label Conventional (`TEST03_MULTILABEL`) | Transformer (`TEST04_TRANSFORMER_multilabel_2`) |
|--------|---------------------------|------------------------------------------------|-------------------------------------------------|
| Task type | One genre per game (primary genre) | Several genres per game | Several genres per game |
| Label representation | LabelEncoder (multi-class) | MultiLabelBinarizer + One-vs-Rest | Multi-hot, sigmoid per genre |
| Feature extraction | TF-IDF | TF-IDF | DistilBERT contextual embeddings |
| Feature configuration | max_features=20000, ngram (1,2), sublinear_tf, min_df=3, max_df=0.95 | Identical to single-label | Tokenizer with max length 256 |
| Algorithms | Naive Bayes, KNN, Linear SVM | Naive Bayes, KNN, Linear SVM (each wrapped in One-vs-Rest) | Fine-tuned DistilBERT |
| Hyperparameters | NB alpha=0.1; KNN k=11 (cosine); SVM balanced, max_iter=5000 | Identical to single-label (One-vs-Rest versions) | batch=16, lr=2e-5, epochs=4 |
| Cross-validation | Stratified K-Fold (k=5) | K-Fold (k=5); standard stratified split is not applicable to multi-label | None; uses a hold-out validation set |
| Train/test split | 80/20, stratified | 80/20 | Train/validation split with undersampling |
| Imbalance handling | class_weight='balanced' (SVM) | class_weight='balanced' (SVM) | Undersampling + per-genre threshold tuning |
| Accuracy definition | Standard accuracy | 1 - Hamming Loss | Threshold-based per genre |
| Metrics | Accuracy, Precision, Recall, F1 (macro) + MSE, RMSE | Accuracy, Precision, Recall, F1 (macro) + MSE, RMSE | Precision, Recall, F1 (macro) + tuned thresholds |
| Preprocessing | Case folding, tokenize, stopword, stemming, short-token removal (S1/S2/S3) | Same as single-label | Light cleaning only |
| Inference output | One predicted genre + Top-5 confidence ranking | Multiple predicted genres + per-genre confidence | Multiple predicted genres + sigmoid confidence |
| Data language | English | English | English |

### Key points

- The single-label and multi-label conventional notebooks now share an identical feature pipeline and identical hyperparameters. The difference between them is purely methodological: single-label predicts one genre (LabelEncoder, Stratified K-Fold, standard accuracy), while multi-label predicts several genres (One-vs-Rest, K-Fold, accuracy measured as 1 - Hamming Loss).
- The Transformer approach is fundamentally different: it uses contextual embeddings instead of TF-IDF, does not apply stopword removal or stemming, does not use K-Fold, and relies on undersampling and per-genre threshold tuning.
- The inference output of all three notebooks now follows a consistent style: a list of per-genre confidence scores with a marker indicating the predicted genre(s).

---

## 6. Detailed Explanation of Each File

### 6.1 `fix_steam_games_csv_from_json.ipynb` (data preparation)

The raw Kaggle dataset cannot be read directly because some rows have a different number of columns than the header expects (the columns become misaligned). This notebook reconstructs the records correctly and writes a clean CSV (`games_fixed.csv`). Always run this notebook before the model notebooks.

### 6.2 `TEST03_2.ipynb` (single-label, main approach)

Predicts the primary genre of a game. The pipeline performs preprocessing (S1/S2/S3 scenarios), builds TF-IDF features, trains Naive Bayes, KNN, and Linear SVM, compares them with Stratified K-Fold cross-validation, and reports Accuracy, Precision, Recall, F1, MSE, and RMSE. The best model (highest F1) is selected automatically, and an inference example prints the predicted genre together with a Top-5 confidence ranking.

### 6.3 `TEST03_MULTILABEL.ipynb` (multi-label, conventional)

Uses the same feature pipeline and hyperparameters as the single-label notebook, but treats the problem as multi-label using MultiLabelBinarizer and One-vs-Rest classifiers. Because standard stratified splitting does not apply to multi-label targets, plain K-Fold is used, and accuracy is reported as 1 - Hamming Loss. The inference example prints all predicted genres with their per-genre confidence.

### 6.4 `TEST04_TRANSFORMER_multilabel_2.ipynb` (Transformer)

Fine-tunes DistilBERT for multi-label genre prediction. Instead of TF-IDF, it uses the model's contextual embeddings, with light text cleaning only. It applies undersampling to address class imbalance and tunes a decision threshold per genre on a validation set. The inference example prints the predicted genres with their sigmoid confidence scores.

---

## 7. How to Run

### Requirements

Classical notebooks (`TEST03_2`, `TEST03_MULTILABEL`):

```
python >= 3.9
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
```

Transformer notebook (`TEST04_TRANSFORMER_multilabel_2`) additionally needs:

```
torch
transformers
```

NLTK resources used by the classical notebooks:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Steps

1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/fronkongames/steam-games-dataset
2. Run `fix_steam_games_csv_from_json.ipynb` to repair the data and generate `games_fixed.csv`.
3. Open and run any of the model notebooks:
   - `TEST03_2.ipynb` for single-label classification
   - `TEST03_MULTILABEL.ipynb` for multi-label classification
   - `TEST04_TRANSFORMER_multilabel_2.ipynb` for the Transformer model

A GPU is recommended for the Transformer notebook.

---

## 8. Evaluation Metrics

- Accuracy: proportion of correct predictions. For the multi-label notebook it is reported as 1 - Hamming Loss.
- Precision, Recall, F1-score: reported with macro averaging so that each genre contributes equally regardless of frequency.
- MSE and RMSE: reported in the classical notebooks as additional error measures; genre labels are encoded to integers before these are computed.

---

## 9. Notes and Limitations

- Single-label classification assigns exactly one genre per game; games that genuinely belong to multiple genres are simplified to their primary genre. The multi-label notebooks address this limitation.
- The dataset is imbalanced across genres, which is mitigated through balanced class weights (classical) and undersampling plus threshold tuning (Transformer).
- Only English synopses are used; results may not transfer to other languages.
- All experiments depend on the cleaned dataset produced by `fix_steam_games_csv_from_json.ipynb`; using the unrepaired raw file will cause loading errors.
