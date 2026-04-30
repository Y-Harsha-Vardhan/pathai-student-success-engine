# PathAI Student Success Engine

An end-to-end learning analytics system that models student engagement, predicts early disengagement risk, and recommends personalized course pathways using behavioral and academic data.

*Note: This repository is being built iteratively for the Studor DS Screening Project.*

---

## 📌 Overview

This system is designed to be deployed within a university setting to:

* **[Completed]** Track and score student engagement over time using behavioral clickstream data.
* **[Completed]** Predict disengagement risk early (before Week 6) using an interpretable ML model.
* **[WIP]** Recommend suitable courses for future semesters using a hybrid ensemble model.

---

## 📂 Project Structure

```text
pathai-student-success-engine/
├── archive/                  # Raw OULAD dataset CSVs (ignored in Git)
├── student_engagement.py     # Task 1: Behavioral scoring & archetypes
├── predictive_model.py       # Task 2: Early-warning model + SHAP alerts
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
Download the Open University Learning Analytics Dataset (OULAD) from Kaggle.
Create a folder named `archive/` in the root directory and place the CSV files inside it.

Required files:

* studentVle.csv
* studentAssessment.csv
* studentInfo.csv
* assessments.csv

---

## 🚀 Usage

### Task 1: Behavioral Scoring Framework

```bash
python student_engagement.py
```

**Outputs:**

* Prints learned feature weights (Spearman-based)
* Generates `engagement_archetypes.png` showing student trajectories

---

### Task 2: Early Warning Predictive Engine

```bash
python predictive_model.py
```

**Outputs:**

* Applies strict Day-41 cutoff (no future leakage)
* Uses temporal cohort split (train → past, test → future)
* Optimizes threshold using F2 score (recall-focused)
* Generates `calibration_curve.png`
* Prints example alert messages with SHAP-based explanations

---

## 🎯 Design Principles

* **No Data Leakage:** Time-aware feature engineering and proper temporal splits
* **Interpretability:** SHAP explanations + transparent feature logic
* **Product Thinking:** Outputs designed as actionable alerts, not just predictions

---

## 👤 Author

Harsha Vardhan
B.Tech CSE, IIT Bombay
