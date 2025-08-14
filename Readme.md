# Credit Risk Scoring App 🏦

This Streamlit web application predicts the likelihood of loan approval based on applicant financial and credit-related features.The underlying model is a **CatBoostClassifier**, trained on processed data with techniques like **EDA**, **encoding**, **SMOTE**, and evaluated using **AUC-ROC**.

---

## 🚀 Features

- Interactive loan approval prediction tool
- Uses top-performing **CatBoost ML model**
- Displays probability of approval
- Shows most important features influencing decision
- Clean and intuitive Streamlit-based UI

---

## 🧠 ML Workflow Summary

- **Exploratory Data Analysis (EDA):**  
  Identified feature distributions, missing values, and patterns in approval status.
  
- **Data Preprocessing:**  
  - Applied **Label Encoding** and **One-Hot Encoding**  
  - Handled missing values
  - Scaled and transformed features as needed

- **Class Imbalance Handling:**  
  Used **SMOTE** (Synthetic Minority Oversampling Technique) to balance approved/denied classes.

- **Model Selection:**  
  Best model selected using AUC-ROC and classification report was **CatBoostClassifier**.

- **Evaluation Metrics:**  
  - Accuracy, Precision, Recall, F1-score  
  - ROC-AUC Curve

---

## 🏗️ Top Features Used

The app focuses on the following top features:

- `AMT_CREDIT`
- `AMT_INCOME_TOTAL`
- `EXT_SOURCE_1`
- `EXT_SOURCE_2`
- `EXT_SOURCE_3`
- `DAYS_EMPLOYED`
- `REGION_POPULATION_RELATIVE`

These features were selected based on model importance analysis.

---

## 📦 Project Structure
credit-risk-scoring-app/


│

├── app.py # Streamlit application code

├── model2.pkl # Trained CatBoost model

├── features3.pkl # List of selected features

├── requirements.txt # Python dependencies

├── .gitignore # Ignored files

└── README.md # Project overview and instructions



---

## ▶️ How to Run the App Locally

1. **Clone the repository:**
   ```
   git clone https://github.com/Prerna2411/Credit-Risk-Scoring-App.git
   cd credit-risk-scoring-app

 2.Create and activate a virtual environment:
    python -m venv env
  source env/bin/activate     # On Windows: env\Scripts\activate

  3.Install dependencies:

    pip install -r requirements.txt


  4.Start the Streamlit app:
     streamlit run app.py


🛠️ Tech Stack

    Python 3.8+

    Streamlit

    CatBoost,XGBoost,Logistic Regression,SVM,LightBGM,RandomForest

    scikit-learn

    pandas, numpy

    joblib

    matplotlib


Dataset Used-Kaggle-https://www.kaggle.com/competitions/home-credit-default-risk

