# PathAI Student Success Engine

An end-to-end learning analytics system that models student engagement, predicts early disengagement risk, and recommends personalized course pathways using behavioral and academic data.

*Note: All three phases of the Studor DS Screening Project are now completed and functional.*

---

## 📌 Overview

This system is designed to be deployed within a university setting to:

* **[Completed]** Track and score student engagement over time using behavioral clickstream data.
* **[Completed]** Predict disengagement risk early (before Week 6) using an interpretable ML model.
* **[Completed]** Recommend suitable courses for future semesters using a hybrid ensemble model with a proxy holdout evaluation.

---

## 📂 Project Structure

```text
pathai-student-success-engine/
├── archive/                  # Raw OULAD dataset CSVs (ignored in Git)
├── student_engagement.py     # Task 1: Behavioral scoring & archetypes
├── predictive_model.py       # Task 2: Early-warning model + SHAP alerts
├── course_recommender.py     # Task 3: Hybrid recommendation engine + evaluation
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Setup Instructions

**1. Clone the repository:**

```bash
git clone https://github.com/Y-Harsha-Vardhan/pathai-student-success-engine.git
cd pathai-student-success-engine
```

**2. Create and activate a virtual environment:**

```bash
# Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Dataset Placement:**
Download the Open University Learning Analytics Dataset (OULAD).
Create an `archive/` folder in the root directory and place the CSV files inside it.

---

## 🚀 Usage

### Task 1: Behavioral Scoring Framework

```bash
python student_engagement.py
```

**Outputs:**

* Prints learned feature weights (Spearman-based)
* Generates `engagement_archetypes.png`

---

### Task 2: Early Warning Predictive Engine

```bash
python predictive_model.py
```

**Outputs:**

* Trains a calibrated XGBoost model
* Optimizes threshold using F2-score (recall-focused)
* Generates `calibration_curve.png`
* Prints SHAP-based alert explanations for advisors

---

### Task 3: Hybrid Course Recommendation Engine

```bash
python course_recommender.py
```

**Outputs:**

* Hybrid model combining Content-Based (Logistic Regression) and Collaborative Filtering (KNN)
* Proxy holdout evaluation (predicting next enrollment)
* Diversity-aware recommendations across domains
* Prints evaluation report (~15.2 point uplift in Precision@3 over baseline)

---

## 🎯 Design Principles

* **Zero-Leakage Design:** Day-41 cutoff + temporal splits ensure realistic deployment
* **Interpretability:** SHAP explanations + transparent feature design
* **Product Thinking:** Focus on actionable alerts and usable recommendations

---

## 👤 Author
```
Harsha Vardhan
B.Tech CSE, IIT Bombay
```