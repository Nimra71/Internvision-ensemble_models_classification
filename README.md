# Ensemble Models for Breast Cancer Classification

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Models Implemented](#-models-implemented)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results](#-results)
- [Ensemble Method Comparison](#-ensemble-method-comparison)
- [Technologies Used](#-technologies-used)
- [How to Run](#-how-to-run)
- [Author](#-author)

---

## 📌 Project Overview
This project implements and compares multiple ensemble learning techniques to classify breast tumors as **malignant** or **benign**.

The goal is to evaluate how different ensemble methods improve prediction performance compared to single models.

Implemented methods:
- Random Forest (Bagging)
- Gradient Boosting (Boosting)
- Stacking (Meta-learning)

---

## 📊 Dataset
- Source: Scikit-learn Breast Cancer Dataset  
- Samples: 569  
- Features: 30 numerical features  

Target:
- 0 → Malignant  
- 1 → Benign  

---

## ⚙️ Models Implemented

### Random Forest
Multiple decision trees trained on random subsets of data and combined predictions.

### Gradient Boosting
Sequential learning where each model corrects previous errors.

### Stacking
Multiple base models combined using a final meta-model.

---

## 🧪 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC Score  
- Confusion Matrix  
- ROC Curve  

---

## 📈 Results
All ensemble methods performed strongly.

The **Stacking model achieved the best performance**, showing the advantage of combining predictions from multiple models.

---

## ⚖️ Ensemble Method Comparison

### Random Forest
Advantages:
- Reduces overfitting
- Stable performance

Disadvantages:
- Less interpretable
- Computational cost

---

### Gradient Boosting
Advantages:
- High predictive accuracy
- Learns complex patterns

Disadvantages:
- Sensitive to noise
- Slower training

---

### Stacking
Advantages:
- Combines strengths of multiple models
- Often best performance

Disadvantages:
- Complex implementation
- Higher computation

---

## 🛠 Technologies Used
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## ▶️ How to Run
1. Clone repository  
2. Install dependencies  
3. Open notebook  
4. Run all cells  

Install libraries:

pip install -r requirements.txt

---

## 📌 Author
Machine Learning Internship Project
