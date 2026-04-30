# PathAI Student Success Engine

An end-to-end learning analytics system that models student engagement, predicts early disengagement risk, and recommends personalized course pathways using behavioral and academic data. 

*Note: This repository is being built iteratively for the Studor DS Screening Project.*

---

## 📌 Overview

This system is designed to be deployed within a university setting to:
* **[Completed]** Track and score student engagement over time using behavioral clickstream data.
* **[WIP]** Predict disengagement risk early (before Week 6).
* **[WIP]** Recommend suitable courses for future semesters using a hybrid ensemble model.

---

## 📂 Project Structure

```
pathai-student-success-engine/
│
├── archive/                  # Raw OULAD dataset CSVs (Ignored in Git)
├── student_engagement.py     # Task 1: Behavioral Scoring & Archetype pipeline
├── requirements.txt          # Frozen Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
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
Download the Open University Learning Analytics Dataset (OULAD) from Kaggle. Create a folder named `archive` in the root directory and place the extracted CSV files inside it.
*Required files: `studentVle.csv`, `studentAssessment.csv`, `studentInfo.csv`, `assessments.csv`.*

---

## 🚀 Usage 

### Task 1: Behavioral Scoring Framework
To generate the dynamic engagement scores and extract the week-by-week trajectories for the student archetypes, run:
```bash
python student_engagement.py
```
**Outputs:** 
* Prints the data-driven Spearman correlation weights to the terminal.
* Generates `engagement_archetypes.png` in the root directory, visualizing the "Steady Engager", "Early Dropout", and "Late Recoverer" profiles.

---

## 🎯 Design Principles

* **No Data Leakage:** Strict implementation of expanding-window min-max scalers for time-series features.
* **Interpretability:** Preference for data-driven, mathematically transparent weights over arbitrary assertions.
* **Product-Oriented:** Outputs are designed to translate complex models into actionable, plain-English insights for university staff.

---

## 👤 Author
```
Harsha Vardhan
B.Tech CSE, IIT Bombay
```