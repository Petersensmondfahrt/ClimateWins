# ClimateWins
I apply several Machine Learning algorithms in order to figure out how to predict "pleasant weather" in various weather stations across Europe.
# 🌦️ ClimateWins – Machine Learning for Weather Prediction

This repository contains my work for **Exercise 1.5: Supervised Learning Algorithms Part 2**  
from the *Basics of Machine Learning for Analysts* bootcamp.  
The goal is to explore and compare different supervised machine learning algorithms  
to predict whether a day will be **pleasant** or **unpleasant** based on historical weather data.

---

## 🧠 Project Overview

**Objective:**  
Use supervised learning techniques to classify weather days as “pleasant” or “unpleasant”  
using temperature, humidity, wind speed, and other features from multiple European weather stations.

**Algorithms Implemented:**
1. **K-Nearest Neighbors (KNN)** – baseline model  
2. **Decision Tree Classifier** – interpretable, but prone to overfitting  
3. **Artificial Neural Network (ANN)** – scalable and best generalization when features are scaled

**Key Findings:**
- After fixing data leakage, the Decision Tree reached ~60% test accuracy.  
- The scaled ANN achieved the best result (~96% test accuracy).  
- KNN performed consistently well (~80%) as a simple benchmark.  
- Feature scaling and cleaning were critical to realistic results.  

---

## 📊 Data Description

**Data Sources:**
- `Dataset-weather-prediction-dataset-processed.csv`  
  → Daily numeric weather features (temperature, humidity, wind, etc.)
- `Dataset-Answers-Weather_Prediction_Pleasant_Weather.csv`  
  → Binary target labels (`0` = unpleasant, `1` = pleasant)

**Preprocessing Steps:**
- Removed low-data stations (`TOURS`, `ROMA`, `GDANKS`)
- Dropped non-numeric columns (`DATE`, `MONTH`, `STATION`)
- Scaled numeric features using `StandardScaler` (for ANN)
- Verified there was no data leakage between features and labels
- Split data into training (70%) and testing (30%) sets with stratification

---

## ⚙️ Installation & Requirements

**Environment:**  
- Python 3.10 or later  
- Anaconda or venv recommended

**Required Libraries:**
```bash
pip install pandas numpy scikit-learn matplotlib
