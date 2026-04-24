# Bias-Aware AI Resume Screening System

Built a bias-aware ATS system using ML and DL to improve fair hiring decisions.

---

## Overview  
This project is an AI-powered resume screening system that ranks candidates using machine learning, semantic similarity, and ATS-based scoring. It also incorporates fairness-aware evaluation to reduce bias in hiring decisions.

---

## Features  
- ATS score using skill matching  
- ML models (Random Forest, XGBoost)  
- Semantic similarity using Sentence Transformers  
- Hybrid scoring system  
- Fairness-aware ranking  
- Top candidate selection  

---

## Tech Stack  
- Python  
- Scikit-learn  
- XGBoost  
- Sentence Transformers  
- Pandas  
- NumPy  

---

## How It Works  
1. Input: Resume + Job Description  
2. TF-IDF feature extraction  
3. ML model prediction  
4. Semantic similarity scoring  
5. Final hybrid score generation  

---

## Sample Code  

```python
# Final hybrid scoring formula
df["final_score"] = (
    0.5 * df["ATS_score"] +
    0.3 * df["ML_score"] +
    0.2 * df["match_score"]
)
