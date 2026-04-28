# 🏦 LoanIQ — Loan Eligibility Prediction Web App

A production-ready machine learning web application that predicts loan eligibility using 7 different classification algorithms. Built with **Flask**, **SQLite**, and **scikit-learn**.

## ✨ Features

- 🤖 **7 ML Algorithms** — Random Forest, Gradient Boosting, Logistic Regression, Decision Tree, SVM, Naive Bayes, KNN
- 🏆 **Best Model Auto-Selection** — Automatically picks the highest-accuracy model
- 💾 **SQLite Database** — Every prediction is stored and queryable via Flask-SQLAlchemy
- 📊 **Analytics Dashboard** — Charts, stats, paginated history, approval rate tracking
- 🎨 **Premium UI** — DM Serif Display + DM Sans typography, fintech-grade design
- 📱 **Responsive** — Works on mobile and desktop

## 🛠 Tech Stack

| Layer     | Technology |
|-----------|-----------|
| Backend   | Python 3.10+, Flask 3.0 |
| Database  | SQLite via Flask-SQLAlchemy |
| ML        | scikit-learn (7 classifiers) |
| Frontend  | Jinja2 templates, vanilla CSS/JS |
| Charts    | Canvas API (no external libraries) |

## 🚀 Quick Start

```bash
# 1. Clone and enter project
git clone <repo-url>
cd loan_app

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Pre-train the model
python train_model.py

# 5. Run the app
python app.py
```

Open `http://localhost:5000` in your browser.

> **Note**: If `model.pkl` doesn't exist, the app auto-trains on first startup.

## 📁 Project Structure

```
loan_app/
├── app.py              # Flask app + SQLAlchemy models + routes
├── train_model.py      # ML training script (7 classifiers)
├── model.pkl           # Serialized best-performing model
├── model_meta.json     # Model metadata + accuracy comparison
├── requirements.txt
├── instance/
│   └── loans.db        # SQLite database (auto-created)
├── templates/
│   ├── base.html       # Nav, footer, flash messages
│   ├── index.html      # Application form
│   ├── result.html     # Prediction result + model comparison
│   ├── history.html    # Dashboard with charts
│   └── detail.html     # Individual application detail
└── static/
    └── style.css       # Full design system
```

## 🧬 ML Features Used

| Feature | Type |
|---------|------|
| Gender | Categorical |
| Married | Categorical |
| Dependents | Categorical |
| Education | Categorical |
| Self_Employed | Categorical |
| ApplicantIncome | Numerical |
| CoapplicantIncome | Numerical |
| LoanAmount | Numerical |
| Loan_Amount_Term | Numerical |
| Credit_History | Binary |
| Property_Area | Categorical |

## 📊 Model Performance

Typical accuracies on this dataset:
- Random Forest: ~83%
- Gradient Boosting: ~82%
- Logistic Regression: ~81%
- SVM: ~80%
- KNN: ~74%

## 🗄 Database Schema

```sql
loan_applications (
    id, created_at,
    gender, married, dependents, education, self_employed, property_area,
    applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history,
    prediction, confidence, model_used
)
```

## 📜 License

MIT License — see LICENSE file.
## Author: Aayush Suthar
