# 🩺 Diabetes Prediction with Multi‑Model Machine Learning

## 📌 Overview
This project predicts the likelihood of diabetes in patients using multiple machine learning algorithms.  
The pipeline includes **data cleaning, outlier processing, feature scaling**, and **multi‑model evaluation**.  
Given the medical context, **recall** for the positive class (diabetes) is prioritized to minimize false negatives.

## 🚀 Features
- Data preprocessing:
  - Outlier detection & transformation
  - Handling skewed distributions
  - Feature scaling via `StandardScaler`
- Model training & evaluation:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - SVM (RBF Kernel)
  - XGBoost
  - LightGBM
- Metrics:
  - Cross‑validation Accuracy
  - Test Accuracy
  - F1 Score
  - **Recall (priority metric)**
  - ROC AUC
- Clean training output with suppressed verbose logs

## 📊 Results Summary
| Model                | CV Accuracy | Test Accuracy | F1 Score | ROC AUC | Recall (Class 1) |
|----------------------|-------------|--------------|----------|---------|------------------|
| Logistic Regression  | 0.7818      | 0.6948       | 0.5347   | 0.8165  | 0.50              |
| Random Forest        | 0.7704      | 0.7143       | 0.5769   | 0.8086  | 0.56              |
| Gradient Boosting    | 0.7444      | 0.7143       | 0.5600   | 0.8161  | 0.52              |
| SVM (RBF)            | 0.7655      | 0.7273       | 0.5800   | 0.7970  | 0.54              |
| **XGBoost**          | 0.7379      | 0.7338       | 0.6095   | **0.8248** | 0.59            |
| **LightGBM** ✅       | 0.7460      | **0.7532**   | **0.6415** | 0.8078  | **0.63**         |

> **Note:** LightGBM achieved the highest recall, making it the preferred choice for this detection task.

## ⚙️ Tech Stack
- **Python**: Pandas, NumPy, Scikit-learn
- **Machine Learning**: XGBoost, LightGBM
- **Visualization**: Matplotlib, Seaborn (EDA phase)
- **Development**: Jupyter Notebooks / VS Code

## 📈 How It Works
1. **Data Processing** — clean the dataset, handle outliers, transform skewed features.
2. **Train-Test Split** — `test_size=0.2` with stratification to preserve class distribution.
3. **Pipeline Creation** — scaling + model wrapped in `Pipeline` for consistency.
4. **Evaluation** — cross-validation on the training set, predictions & metrics on the test set.
5. **Model Selection** — sorting results by **recall for Class 1** (diabetes).





---
**Author:** Ali Abdollahi  
📧 *[Your email or contact info if you wish]*  
