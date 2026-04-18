# PathAI Student Success Engine

An end-to-end learning analytics system that models student engagement, predicts early disengagement risk, and recommends personalized course pathways using behavioral and academic data.

---

## 📌 Overview

This project is part of the Studor DS Screening Project. The objective is to design a system that could realistically be deployed within a university setting to:

* Track and score student engagement over time
* Predict disengagement risk early (before Week 6)
* Recommend suitable courses for future semesters

The system leverages behavioral clickstream data from the **OULAD dataset** (or equivalent).

---

## 🚀 Features

### 1. Behavioral Engagement Scoring

* Dynamic engagement score (0–100)
* Week-by-week trajectory tracking
* Feature engineering from clickstream data (recency, frequency, activity patterns, etc.)
* Student archetype analysis

### 2. Disengagement Prediction

* Binary classification model (withdraw/fail vs pass)
* Early prediction using only Week ≤ 6 data
* Evaluation using Precision, Recall, F1, ROC-AUC
* Model calibration and feature importance analysis

### 3. Course Recommendation Engine

* Content-based recommendation system
* Collaborative filtering approach
* Cold-start handling strategy
* Evaluation using ranking metrics

---

## 📂 Project Structure

pathai-student-success-engine/
│
├── data/               # Raw and processed datasets
├── notebooks/          # Exploratory analysis and prototyping
├── src/                # Core source code
│   ├── features/       # Feature engineering pipelines
│   ├── models/         # Training and evaluation scripts
│   ├── scoring/        # Engagement scoring logic
│   └── recommend/      # Recommendation engine
│
├── outputs/            # Plots, results, and artifacts
├── report/             # Final PDF report
├── requirements.txt    # Dependencies
└── README.md

---

## ⚙️ Setup Instructions

1. Clone the repository:

```
git clone https://github.com/<your-username>/pathai-student-success-engine.git
cd pathai-student-success-engine
```

2. Create a virtual environment:

```
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```
pip install -r requirements.txt
```

---

## 📊 Dataset

* Primary: Open University Learning Analytics Dataset (OULAD)
* Alternative: xAPI-Edu-Data

Place datasets inside the `data/` directory before running scripts.

---

## 📈 Evaluation Metrics

* Classification: Precision, Recall, F1-score, ROC-AUC
* Calibration: Reliability analysis
* Recommendation: Precision@K / Coverage

---

## 🎯 Design Principles

* No data leakage (strict temporal constraints)
* Interpretability over complexity
* Product-oriented outputs (actionable insights for staff)
* Reproducibility and clean code

---

## 📌 Deliverables

* GitHub repository
* PDF report (max 6 pages)
* 10-minute walkthrough video

---

## 🧠 Future Improvements

* Real-time engagement tracking dashboard
* Intervention recommendation system
* Advanced sequence models for behavior modeling
* A/B testing framework for recommendations

---

## 👤 Author

Harsha Vardhan
IIT Bombay

---

## 📄 License

This project is for evaluation purposes only.

