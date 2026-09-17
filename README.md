# 🔍 Fake News Detector

A production-style NLP web application that classifies news articles as **Real** or **Fake** using Machine Learning.

[Streamlit App Live Link](https://fake-news-detector-xozqywxappgra4jn76mwlvd.streamlit.app)


## 🎯 Overview

This project demonstrates a complete NLP + ML pipeline built for resume and portfolio purposes. It covers everything from raw text preprocessing to model deployment.

## 📊 Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~99%   |
| Precision | ~99%   |
| Recall    | ~99%   |
| F1-Score  | ~99%   |

## 🧠 Tech Stack

| Layer           | Technology                        |
|-----------------|-----------------------------------|
| Language        | Python 3.10                       |
| NLP             | NLTK (stopwords, stemming)        |
| Feature Eng.    | Scikit-learn TF-IDF               |
| ML Model        | Logistic Regression               |
| Frontend        | Streamlit                         |
| Visualization   | Matplotlib, Seaborn               |
| Deployment      | Streamlit Cloud + GitHub          |

## 🏗️ Architecture# 
User Input → NLP Preprocessing → TF-IDF Vectorization → Logistic Regression → Result + Confidence


## 📁 Project Structure
fake-news-detector/ 
├── app.py ← Streamlit frontend 
├── train_model.py ← Training pipeline 
├── model.pkl ← Trained model 
├── vectorizer.pkl ← Fitted TF-IDF vectorizer 
├── requirements.txt 
├── utils/
│ └── preprocess.py ← Shared NLP preprocessing 
├── dataset/ ← (not committed — Kaggle data) 
└── screenshots/

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Place Fake.csv and True.csv in dataset/
python train_model.py

streamlit run app.py
```

## 📊 Dataset

[Kaggle: Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
~23,000 fake articles + ~21,000 real articles

## 💡 Key Learnings

- Designed reusable NLP preprocessing pipeline
- Understood why TF-IDF + Logistic Regression is a strong baseline for text classification
- Learned train/test separation discipline (no data leakage)
- Built and deployed a Streamlit app from scratch
