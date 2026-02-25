"""
Train Loan Eligibility Prediction Model
Uses scikit-learn Pipeline with preprocessing + Random Forest (best performer)
Also trains multiple classifiers and saves comparison metadata.
"""
import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# ── Reproducible synthetic dataset matching real loan_prediction.csv stats ──
np.random.seed(42)
N = 614

def generate_loan_dataset(n=N):
    gender       = np.random.choice(['Male', 'Female'], n, p=[0.82, 0.18])
    married      = np.random.choice(['Yes', 'No'],      n, p=[0.65, 0.35])
    dependents   = np.random.choice(['0','1','2','3+'], n, p=[0.57, 0.17, 0.16, 0.10])
    education    = np.random.choice(['Graduate','Not Graduate'], n, p=[0.78, 0.22])
    self_emp     = np.random.choice(['Yes','No'],        n, p=[0.14, 0.86])
    app_income   = np.random.lognormal(mean=8.5, sigma=0.6, size=n).astype(int)
    coapp_income = np.where(married=='Yes',
                            np.random.lognormal(7.5, 0.8, n).astype(int),
                            np.zeros(n, dtype=int))
    loan_amount  = np.random.lognormal(4.9, 0.5, n).astype(int)
    term         = np.random.choice([360, 180, 480, 300, 240], n, p=[0.70,0.10,0.09,0.06,0.05])
    credit_hist  = np.random.choice([1.0, 0.0], n, p=[0.84, 0.16])
    prop_area    = np.random.choice(['Urban','Semiurban','Rural'], n, p=[0.38,0.38,0.24])

    # Loan status – correlated with credit history and income
    base_prob = 0.69
    prob = (credit_hist * 0.4
            + (app_income > 5000).astype(float) * 0.1
            + (prop_area == 'Semiurban').astype(float) * 0.05
            + base_prob * 0.45)
    prob = np.clip(prob, 0.1, 0.98)
    loan_status = np.where(np.random.random(n) < prob, 'Y', 'N')

    return pd.DataFrame({
        'Gender': gender, 'Married': married, 'Dependents': dependents,
        'Education': education, 'Self_Employed': self_emp,
        'ApplicantIncome': app_income, 'CoapplicantIncome': coapp_income,
        'LoanAmount': loan_amount, 'Loan_Amount_Term': term,
        'Credit_History': credit_hist, 'Property_Area': prop_area,
        'Loan_Status': loan_status
    })


def preprocess(df):
    df = df.copy()
    # Encode categoricals
    le_map = {}
    for col in ['Gender','Married','Education','Self_Employed','Property_Area','Dependents']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_map[col] = {cls: int(i) for i, cls in enumerate(le.classes_)}
    return df, le_map


def train():
    df = generate_loan_dataset()
    df, le_map = preprocess(df)

    X = df.drop('Loan_Status', axis=1)
    y = (df['Loan_Status'] == 'Y').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = {
        'Random Forest':        RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting':    GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression':  LogisticRegression(max_iter=1000),
        'Decision Tree':        DecisionTreeClassifier(random_state=42),
        'SVM':                  SVC(probability=True),
        'Naive Bayes':          GaussianNB(),
        'KNN':                  KNeighborsClassifier(n_neighbors=5),
    }

    results = {}
    best_name, best_score, best_model = '', 0, None

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred  = clf.predict(X_test)
        acc     = accuracy_score(y_test, y_pred)
        cv      = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()
        results[name] = {'accuracy': round(acc*100, 2), 'cv_accuracy': round(cv*100, 2)}
        print(f"{name:25s} | Acc: {acc:.3f} | CV: {cv:.3f}")
        if acc > best_score:
            best_score = acc
            best_name  = name
            best_model = clf

    print(f"\n✅ Best model: {best_name} ({best_score:.3f})")

    # Save artifacts
    model_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)

    meta = {
        'best_model': best_name,
        'best_accuracy': round(best_score*100, 2),
        'results': results,
        'feature_order': list(X.columns),
        'label_encodings': le_map
    }
    with open(os.path.join(model_dir, 'model_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("✅ model.pkl and model_meta.json saved.")
    return meta


if __name__ == '__main__':
    train()
