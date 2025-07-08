# 🧠 Multiclass Classification & Logistic Regression — From Scratch

This project is a complete hands-on implementation of a **multiclass classification model**, built entirely from scratch in Python — without using any machine learning libraries. It demonstrates full ownership of the data science pipeline, from raw data analysis to prediction.

---

## 🔍 Project Summary

The goal is to **predict a categorical outcome** based on several input features using a **logistic regression model**. While the dataset is inspired by fictional student grades (e.g., from the Harry Potter universe), the technical framework is fully applicable to real-world problems such as:

- Customer segmentation  
- Medical diagnosis  
- Risk prediction  
- Fraud detection  

Everything is built manually — including model training, loss functions, gradient descent, and predictions — to showcase a **deep understanding of ML fundamentals**.

---

## 🎯 Learning Objectives

- Apply core concepts of **supervised learning**
- Build a **multiclass logistic regression model** (One-vs-All)
- Clean, explore, and preprocess **real-world-like datasets**
- Implement core algorithms from scratch: no `scikit-learn`, `TensorFlow`, or `PyTorch`
- Evaluate, visualize, and interpret model performance

---

## 📊 Workflow Overview

### 1. Exploratory Data Analysis  
- Automated statistical summaries (`describe.py`)  
- Histograms, scatter plots, pairplots  
- Feature selection based on distribution and correlation

### 2. Data Preprocessing  
- Missing data handling  
- Manual feature scaling (normalization)  
- Categorical encoding and transformation

### 3. Model Implementation  
- Custom implementation of **logistic regression**  
- **One-vs-All** strategy for multiclass classification  
- Manual **gradient descent** optimization  
- Softmax/sigmoid activation logic

### 4. Evaluation & Prediction  
- Accuracy metrics  
- Confusion matrix interpretation  
- Final model weights exported (`weights.csv`)  
- Prediction results saved (`houses.csv`)

---

## 📈 Visual Examples


*Distribution of key variables across target classes*


*Scatter plot showing correlation between key features*


*Comprehensive pairwise comparison of all numerical features*

---

## 🧠 Skills Demonstrated

- Advanced Python (modular code, data parsing, math-heavy logic)
- Algorithm design: **gradient descent**, **loss minimization**, **classification scoring**
- Data wrangling with **Pandas / NumPy**
- Data visualization and insight extraction
- Problem-solving mindset: everything built without high-level ML APIs

---

## ⚙️ Tech Stack

- **Python 3.10+**
- **Pandas & NumPy** for data handling
- **Matplotlib & Seaborn** for visualizations
- **No ML frameworks used** (built from the ground up)

---

## 💼 Real-World Applications

This project structure is adaptable to a wide range of industry cases where explainability and direct control over the model are critical:

- Predicting customer churn  
- Risk scoring for financial profiles  
- Multi-category classification in product catalogs  
- Educational analytics (student profiling, outcome prediction)

---
