# 🩺 AI-Based Blood Glucose Prediction System

A machine learning project for predicting future blood glucose levels using the **OhioT1DM Dataset**. The system compares multiple machine learning and deep learning models to identify the most accurate approach for continuous glucose prediction in Type 1 Diabetes management.

---

# 🎯 Motivation

Continuous monitoring and prediction of blood glucose levels play an important role in diabetes management. Accurate prediction can help patients take preventive actions against hypoglycemia and hyperglycemia.

This project explores multiple machine learning approaches to determine the most effective model for forecasting blood glucose levels using historical Continuous Glucose Monitoring (CGM) data.

---

# ✨ Features

- Blood glucose prediction using real CGM data
- Comparison of multiple ML and DL models
- Time-series feature engineering
- Performance evaluation using RMSE and MAE
- Data preprocessing and visualization
- Modular machine learning pipeline

---

# 🤖 Models Evaluated

The following models were implemented and compared:

- Linear Regression
- Random Forest
- XGBoost
- Long Short-Term Memory (LSTM)

---

# 📂 Dataset

**OhioT1DM Dataset**

The dataset contains:

- Continuous Glucose Monitoring (CGM) data
- Insulin records
- Meal information
- Time-series patient data

---

# ⚙️ Methodology

1. Data preprocessing and cleaning
2. Feature engineering
3. Train-test split
4. Model training
5. Hyperparameter tuning
6. Model evaluation
7. Performance comparison

---

# 📊 Evaluation Metrics

The models were evaluated using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

---

# 📈 Results

The project compares model performance across multiple patients and prediction horizons.

Key observations include:

- Random Forest achieved strong performance on structured features.
- XGBoost provided competitive predictive accuracy.
- LSTM effectively captured temporal dependencies in glucose data.
- Performance varied depending on prediction horizon and patient-specific data.

---

# 🛠️ Tech Stack

### Programming

- Python

### Machine Learning

- Scikit-learn
- XGBoost
- PyTorch

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib

---

# 📂 Project Structure

```text
AI-Based-Blood-Glucose-Prediction-System/
│
├── dataset/
├── notebooks/
├── models/
├── results/
├── figures/
├── src/
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🚀 Future Improvements

- Transformer-based time-series models
- Real-time glucose prediction
- Explainable AI (SHAP/LIME)
- Mobile deployment
- Personalized prediction models
- Integration with wearable devices

---

# 📄 Research Background

This project was developed as part of my undergraduate work in Medical Electronics and focuses on applying Artificial Intelligence to healthcare.

---

# 👨‍💻 Author

**Arnav Gupta**

- LinkedIn: https://www.linkedin.com/in/arnavgupta1470
- GitHub: https://github.com/arnavgupta177013
