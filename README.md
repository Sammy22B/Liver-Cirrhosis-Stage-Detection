# Liver-Cirrhosis-Stage-Detection
A machine learning project that predicts liver cirrhosis stage (1–3) based on biochemical and clinical test data from the Mayo Clinic PBC study using Logistic Regression, Random Forest, and SVM models.
# 🩺 Liver Cirrhosis Stage Detection using Machine Learning

This project predicts the **stage of liver cirrhosis** in patients based on clinical and biochemical test data. Using machine learning algorithms, the system classifies patients into different stages of liver damage, helping assess disease progression and assist medical decision-making.

---

## 🎯 Objective
To build a machine learning model capable of predicting the **level of liver damage (cirrhosis stage)** based on patient data from the Mayo Clinic’s study on primary biliary cirrhosis (PBC) conducted between **1974 and 1984**.

---

## 📘 Dataset Overview

### 📍 Source
Mayo Clinic study on **Primary Biliary Cirrhosis (PBC)** of the liver.

### 🧩 Description of Columns
| Feature | Description |
|----------|-------------|
| N_Days | Days between registration and death/transplant/study analysis |
| Status | Patient status – C (censored), CL (censored due to transplant), D (death) |
| Drug | Type of drug administered – D-penicillamine or placebo |
| Age | Age of patient (in days) |
| Sex | M = Male, F = Female |
| Ascites | Presence of ascites – N (No), Y (Yes) |
| Hepatomegaly | Presence of hepatomegaly – N (No), Y (Yes) |
| Spiders | Presence of spider nevi – N (No), Y (Yes) |
| Edema | Edema status – N (none), S (resolved/without diuretics), Y (with diuretics) |
| Bilirubin | Serum bilirubin (mg/dl) |
| Cholesterol | Serum cholesterol (mg/dl) |
| Albumin | Serum albumin (gm/dl) |
| Copper | Urine copper (μg/day) |
| Alk_Phos | Alkaline phosphatase (U/L) |
| SGOT | SGOT enzyme level (U/ml) |
| Tryglicerides | Triglycerides (mg/dl) |
| Platelets | Platelet count (per ml/1000) |
| Prothrombin | Prothrombin time (seconds) |
| Stage | **Target variable** – histologic stage of disease (1, 2, or 3) |

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing
- Loaded and inspected the dataset for missing values and data types  
- Converted categorical variables (Drug, Sex, Ascites, etc.) into numeric values  
- Dropped irrelevant columns (`N_Days`, `Status`, `Stage`) for training features  
- Standardized numerical features using **StandardScaler**

### 2️⃣ Exploratory Data Analysis (EDA)
- Visualized **stage distribution** using Seaborn count plots  
- Displayed **feature histograms** for numerical variables  
- Generated **correlation heatmaps** for major clinical indicators  

### 3️⃣ Model Training and Evaluation
Trained multiple supervised learning models:

| Model | Description | Accuracy (%) |
|--------|--------------|--------------|
| Logistic Regression | Multinomial logistic model | ~78% |
| Random Forest (Baseline) | Ensemble classifier with 100 trees | ~85% |
| Random Forest (Tuned) | GridSearchCV optimized hyperparameters | **~88%** |
| SVM (RBF Kernel) | Non-linear classifier for complex decision boundaries | ~80% |

Each model was evaluated using:
- **Accuracy score**  
- **Confusion matrix visualization**  
- **Classification report (Precision, Recall, F1-score)**  

### 4️⃣ Feature Importance
Used Random Forest to identify key biochemical indicators affecting liver cirrhosis stage:
- Bilirubin, Albumin, SGOT, Prothrombin, and Copper were among the most important features.

### 5️⃣ ROC Curve Analysis
Generated **One-vs-Rest ROC curves** for each class to evaluate the discriminative power of the tuned Random Forest model.

---

## 🧠 Technologies Used
- **Python 3.8+**
- **Libraries:**
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`
  - `pickle` (for model saving)

---
