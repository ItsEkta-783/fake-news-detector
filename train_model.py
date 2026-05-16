# train_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Run this script ONCE to:
#   1. Load and prepare the Kaggle Fake News dataset
#   2. Preprocess text using NLP pipeline
#   3. Vectorize using TF-IDF
#   4. Train a Logistic Regression classifier
#   5. Evaluate and print metrics
#   6. Save model.pkl and vectorizer.pkl
#
# Usage:
#   python train_model.py
#
# Requirements:
#   Place Fake.csv and True.csv inside the dataset/ folder
# ─────────────────────────────────────────────────────────────────────────────

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# Import our shared preprocessing function
from utils.preprocess import preprocess_text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load Dataset
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading datasets...")
print("=" * 60)

fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

print(f"Fake news articles: {len(fake_df):,}")
print(f"Real news articles: {len(true_df):,}")

# Add labels: 0 = Fake, 1 = Real
fake_df['label'] = 0
true_df['label'] = 1

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Prepare Combined Text Column
# ─────────────────────────────────────────────────────────────────────────────

print("\nSTEP 2: Preparing data...")

# Combine title + text for richer signal
# The title of a fake news article is often sensationalist
fake_df['content'] = fake_df['title'] + " " + fake_df['text']
true_df['content'] = true_df['title'] + " " + true_df['text']

# Merge both datasets
df = pd.concat([fake_df[['content', 'label']],
                true_df[['content', 'label']]], ignore_index=True)

# Shuffle the dataset (important before splitting)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total articles: {len(df):,}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")
df.dropna(inplace=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: NLP Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

print("\nSTEP 3: Preprocessing text (this may take 1-2 minutes)...")

df['cleaned'] = df['content'].apply(preprocess_text)

print("Sample cleaned text:")
print(df['cleaned'].iloc[0][:200])

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Train/Test Split
# ─────────────────────────────────────────────────────────────────────────────

print("\nSTEP 4: Splitting dataset (80/20)...")

X = df['cleaned']
y = df['label']

# stratify=y ensures equal class proportions in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: TF-IDF Vectorization
# ─────────────────────────────────────────────────────────────────────────────

print("\nSTEP 5: Fitting TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    max_features=50000,   # Keep top 50k most informative terms
    ngram_range=(1, 2),   # Unigrams + bigrams (e.g., "fake news")
    sublinear_tf=True     # Apply log(1+tf) to dampen extreme frequencies
)

# IMPORTANT: fit_transform on TRAINING data only
X_train_tfidf = vectorizer.fit_transform(X_train)

# Only transform (not fit) on TEST data — prevents data leakage
X_test_tfidf = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")
print(f"Feature matrix shape (train): {X_train_tfidf.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Train Logistic Regression Model
# ─────────────────────────────────────────────────────────────────────────────

print("\nSTEP 6: Training Logistic Regression model...")

model = LogisticRegression(
    C=1.0,           # Inverse regularization strength (default)
    max_iter=1000,   # Ensure convergence on large datasets
    solver='lbfgs',  # Efficient solver for binary classification
    n_jobs=-1        # Use all CPU cores
)

model.fit(X_train_tfidf, y_train)
print("Model training complete!")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Evaluate Model
# ─────────────────────────────────────────────────────────────────────────────

print("\nSTEP 7: Evaluating model...")

y_pred = model.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 40)
print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1 Score : {f1:.4f}")
print("=" * 40)

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

# Save confusion matrix as image
os.makedirs("screenshots", exist_ok=True)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Fake (0)", "Real (1)"],
            yticklabels=["Fake (0)", "Real (1)"])
plt.title("Confusion Matrix — Fake News Detector")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("screenshots/confusion_matrix.png", dpi=150)
print("Confusion matrix saved to screenshots/confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Save Model and Vectorizer
# ─────────────────────────────────────────────────────────────────────────────

print("\nSTEP 8: Saving model and vectorizer...")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ model.pkl saved!")
print("✅ vectorizer.pkl saved!")
print("\n🎉 Training pipeline complete. Run `streamlit run app.py` to launch the app.")

