# Customer Churn Prediction (Python + ML)

A customer churn analysis project using Python to explore the data, build models and identify high-risk customers.

## 🎯 Objectives

- Understand churn drivers through **EDA and feature analysis**.  
- Build and evaluate a simple ML model for churn prediction.  
- Segment customers into risk groups for potential retention strategies.

## 📊 Data

- Typical telecom-style churn dataset with:  
  - Customer demographics  
  - Contract details (tenure, plan type, charges)  
  - Churn indicator  

*(You can replace this section with the exact source once finalised.)*

## 🧠 Approach

1. **Exploratory Data Analysis (EDA)**  
   - Distribution checks, missing values, correlations.  
   - Churn rates by tenure, contract type, payment method, etc.

2. **Data Preparation**  
   - Handling missing values  
   - Encoding categorical variables  
   - Feature scaling where needed  

3. **Modelling**  
   - Baseline model: Logistic Regression  
   - Additional model(s): Decision Tree / Random Forest  
   - Train/test split, cross-validation

4. **Evaluation**  
   - Accuracy, Precision, Recall, F1-score  
   - Confusion matrix  
   - Feature importance & interpretation

## 🛠️ Tools & Libraries

- Python, Jupyter Notebook  
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

## 📁 Project Structure

- `notebooks/` – EDA and modelling notebooks  
- `data/` – input dataset (or sample / link)  
- `models/` – saved models (optional)  
- `README.md` – documentation  

## 🚀 How to Run

1. Create a virtual environment (optional but recommended).  
2. Install dependencies:  

   ```bash
   pip install -r requirements.txt
