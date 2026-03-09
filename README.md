# 🏦 CreditWise: Loan Approval Predictor

CreditWise is an intelligent, AI-powered web application designed to assist in financial decision-making. By leveraging historical loan data, this application predicts the likelihood of loan approval based on user-provided financial and personal indicators.



## 🚀 How it Works
The application uses a **Logistic Regression** model trained on credit data. The workflow is split into two distinct phases:

1.  **Data Processing:** User inputs are collected via an interactive Streamlit interface. The app performs real-time feature engineering (e.g., squaring DTI ratios and Credit Scores) and categorical encoding using a saved `OneHotEncoder` to ensure input parity with the training data.
2.  **Inference:** The processed data is scaled using a saved `StandardScaler` to match the model’s training distribution, resulting in a probability score of loan approval.

## 🛠 Tech Stack
* **Language:** Python 3.x
* **ML Library:** `scikit-learn` (v1.6.1)
* **Frontend:** Streamlit
* **Data Handling:** `pandas`, `numpy`
* **Deployment:** Streamlit Community Cloud

## 📋 Features
* **Real-time Prediction:** Get an instant "Approved" or "Not Approved" verdict.
* **Probability Insights:** View the specific confidence percentage behind the decision.
* **Custom Inputs:** Handles diverse financial profiles, including income, loan terms, credit history, and employment details.

## 📂 Project Structure
* `app.py`: Main application script.
* `requirements.txt`: Project dependencies.
* `model.pkl`: Serialized Logistic Regression model.
* `scaler.pkl`: Serialized StandardScaler for data normalization.
* `ohe.pkl`: Serialized OneHotEncoder for categorical variables.
* `model_columns.pkl`: Mapping of training columns for inference consistency.

## 💡 Deployment Instructions
1. Clone this repository to your local machine.
2. Ensure you have the required dependencies: `pip install -r requirements.txt`.
3. Run the app locally using: `streamlit run app.py`.
4. Deploy to [Streamlit Community Cloud](https://share.streamlit.io/) by connecting your GitHub repository.