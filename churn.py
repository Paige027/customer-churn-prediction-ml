import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

df = pd.read_csv('churn.csv')
X = df[['tenure_months','monthly_charges','total_charges','support_tickets']]
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, preds))
print('Confusion matrix:\n', confusion_matrix(y_test, preds))

joblib.dump(model, 'rf_churn_model.joblib')
