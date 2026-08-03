**🏦 Banking Fraud Detection**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikitlearn">
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit">
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge">
</p>

---

# 📌 Project Overview

The **Banking Fraud Detection System** is a Machine Learning powered web application that predicts whether a banking transaction is **Fraudulent** or **Legitimate**.

This project applies multiple Machine Learning algorithms and a Deep Learning model to analyze transaction patterns and classify fraudulent banking transactions with high accuracy.

The application is deployed using **Streamlit**, enabling users to perform fraud detection directly from their browser.

---

# 🚀 Live Demo

### 🔗 Try the Web Application

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://banking-fraud-detection-myssrah5gyqfl96qrkvpzx.streamlit.app/)

---

# ✨ Features

- 🏦 Banking Transaction Fraud Detection
- 🤖 Logistic Regression Model
- 🌲 Random Forest Classifier
- 🧠 TensorFlow Deep Learning Model
- 📊 Interactive Plotly Visualizations
- 📈 Banking Time-Series Analysis
- 📉 Confusion Matrix Evaluation
- 🌐 Streamlit Web Application
- ⚡ Real-time Fraud Prediction
- 💾 Pre-trained Models
# 🛠️ Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Scikit-Learn | Machine Learning |
| TensorFlow | Deep Learning |
| Streamlit | Web Application |
| Plotly | Interactive Charts |
| NetworkX | Graph Visualization |
| Joblib | Model Serialization |

---

# 🧠 Machine Learning Workflow

```mermaid
flowchart TD

A[Raw Banking Dataset]
-->B[Data Cleaning]

B-->C[Feature Engineering]

C-->D[Data Preprocessing]

D-->E[Train Models]

E-->F[Logistic Regression]

E-->G[Random Forest]

E-->H[TensorFlow Neural Network]

F-->I[Model Evaluation]

G-->I

H-->I

I-->J[Best Model Saved]

J-->K[Streamlit Deployment]

K-->L[Fraud Prediction]
```

---

# 🏗️ Project Architecture

```mermaid
graph LR

A[User]

-->B(Streamlit Web App)

B-->C[Input Transaction]

C-->D[Preprocessor]

D-->E[Machine Learning Model]

E-->F{Prediction}

F-->G[Fraud]

F-->H[Legitimate]
```

---

# 📂 Project Structure

```text
Banking-Fraud-Detection/
│
├── 📁 dataset_cleaning/
│   └── 📝 model_creation.ipynb          # Data cleaning & model training
│
├── 📁 IMAGES/
│   ├── 🖼️ confusion_matrix.png
│   ├── 🖼️ confusion_matrix(tensorflow).png
│   └── 🖼️ Time Series Trends & Forecast.png
│
├── 📁 MODELS/
│   ├── 🤖 logistic_regression_model.joblib
│   ├── 🌲 random_forest_model.joblib
│   ├── ⚙️ preprocessor.joblib
│   └── 🧠 tensorflow_fraud_model.keras
│
├── 🐍 app.py
├── 📊 bank_transactions.csv
├── 📦 requirements.txt
├── ⚡ runtime.txt
├── 🚫 .gitignore
└── 📖 README.md
```

# 📈 Models Used

| Model | Purpose |
|--------|---------|
| Logistic Regression | Baseline Binary Classification |
| Random Forest | Ensemble Learning |
| TensorFlow Neural Network | Deep Learning Classification |

### Model Evaluation

✔ Confusion Matrix

✔ Accuracy Score

✔ Precision

✔ Recall

✔ F1-Score

# 🔄 Prediction Pipeline

```mermaid
flowchart LR

A[User Inputs]

-->B[Data Cleaning]

-->C[Preprocessing]

-->D[Feature Scaling]

-->E[Selected Model]

-->F{Prediction}

F-->G[Fraud]

F-->H[Legitimate]
```

---

# 📷 Project Results

## 📊 Random Forest Confusion Matrix

<p align="center">
<img src="IMAGES/confusion_matrix.png" width="650">
</p>

---

## 🧠 TensorFlow Confusion Matrix

<p align="center">
<img src="IMAGES/confusion_matrix(tensorflow).png" width="650">
</p>

---

## 📈 Banking Transaction Time Series Analysis

<p align="center">
<img src="IMAGES/Time%20Series%20Trends%20%26%20Forecast.png" width="750">
</p>

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/rezumistri2026/banking-fraud-detection.git
```

Go to the project directory

```bash
cd banking-fraud-detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

# 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
tensorflow
plotly
networkx
joblib
```

---

# 🎯 Future Improvements

- Explainable AI (SHAP/LIME)
- XGBoost & LightGBM Models
- Real-time Banking API
- Docker Deployment
- Cloud Database Integration
- User Authentication
- Continuous Model Retraining

---

# 👨‍💻 Author

**Rezu Mistri**

🎓 B.Tech CSE (Artificial Intelligence & Machine Learning)

🏫 Brainware University

🔗 GitHub: https://github.com/rezumistri2026

---

# ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub!

---

# 📜 License

This project is developed for educational and learning purposes.
