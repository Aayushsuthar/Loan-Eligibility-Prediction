"""
LoanIQ — Loan Eligibility Prediction Web App
Flask + SQLite (native) + scikit-learn
"""
import os, json, pickle, subprocess, sys, sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, g
import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH  = os.path.join(BASE_DIR, 'instance', 'loans.db')
os.makedirs(os.path.join(BASE_DIR, 'instance'), exist_ok=True)

app = Flask(__name__)
app.secret_key = 'loaniq_secret_2024'

# ── SQLite ────────────────────────────────────────────────────────────────────
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db: db.close()

def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute("""
        CREATE TABLE IF NOT EXISTS loan_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            gender TEXT, married TEXT, dependents TEXT, education TEXT,
            self_employed TEXT, property_area TEXT,
            applicant_income REAL, coapplicant_income REAL,
            loan_amount REAL, loan_amount_term REAL, credit_history REAL,
            prediction TEXT, confidence REAL, model_used TEXT
        )
    """)
    db.commit(); db.close()

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
META_PATH  = os.path.join(BASE_DIR, 'model_meta.json')
MODEL, META = None, None

def get_model():
    global MODEL, META
    if MODEL is None:
        if not os.path.exists(MODEL_PATH):
            print("🔧 Training model…", flush=True)
            subprocess.run([sys.executable, os.path.join(BASE_DIR, 'train_model.py')])
        with open(MODEL_PATH, 'rb') as f: MODEL = pickle.load(f)
        with open(META_PATH)  as f: META  = json.load(f)
    return MODEL, META

ENCODINGS = {
    'Gender':       {'Female':0,'Male':1},
    'Married':      {'No':0,'Yes':1},
    'Dependents':   {'0':0,'1':1,'2':2,'3+':3},
    'Education':    {'Graduate':0,'Not Graduate':1},
    'Self_Employed':{'No':0,'Yes':1},
    'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},
}
FEATURE_ORDER = ['Gender','Married','Dependents','Education','Self_Employed',
    'ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term',
    'Credit_History','Property_Area']

def encode_input(form):
    feats = []
    for f in FEATURE_ORDER:
        v = form.get(f,'')
        if f in ENCODINGS: feats.append(float(ENCODINGS[f].get(str(v),0)))
        else:
            try: feats.append(float(v))
            except: feats.append(0.0)
    return np.array(feats).reshape(1,-1)

# ── DB helpers ────────────────────────────────────────────────────────────────
def db_stats():
    db=get_db()
    total=db.execute("SELECT COUNT(*) FROM loan_applications").fetchone()[0]
    approved=db.execute("SELECT COUNT(*) FROM loan_applications WHERE prediction='Approved'").fetchone()[0]
    rejected=total-approved
    return {'total':total,'approved':approved,'rejected':rejected,
            'approval_rate':round(approved/total*100,1) if total else 0}

def db_recent(n=5):
    db=get_db()
    return [dict(r) for r in db.execute(
        "SELECT * FROM loan_applications ORDER BY created_at DESC LIMIT ?",(n,)).fetchall()]

def db_paginate(page=1,per_page=10):
    db=get_db()
    total=db.execute("SELECT COUNT(*) FROM loan_applications").fetchone()[0]
    items=[dict(r) for r in db.execute(
        "SELECT * FROM loan_applications ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (per_page,(page-1)*per_page)).fetchall()]
    return items,total,(total+per_page-1)//per_page

def db_income_buckets():
    db=get_db()
    b={'<2k':0,'2k-5k':0,'5k-10k':0,'>10k':0}
    for (inc,) in db.execute("SELECT applicant_income FROM loan_applications").fetchall():
        inc=inc or 0
        if inc<2000: b['<2k']+=1
        elif inc<5000: b['2k-5k']+=1
        elif inc<10000: b['5k-10k']+=1
        else: b['>10k']+=1
    return b

def iter_pages(curr,total_pages,left=2,right=2,edge=1):
    pgs=[]
    for p in range(1,total_pages+1):
        if p<=edge or p>total_pages-edge or (p>=curr-left and p<=curr+right):
            if pgs and pgs[-1]!=p-1: pgs.append(None)
            pgs.append(p)
    return pgs

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    _,meta=get_model()
    return render_template('index.html',meta=meta,stats=db_stats(),recent=db_recent(5))

@app.route('/predict',methods=['POST'])
def predict():
    model,meta=get_model()
    form=request.form
    try:
        X=encode_input(form)
        pred_label=int(model.predict(X)[0])
        result='Approved' if pred_label==1 else 'Rejected'
        confidence=None
        if hasattr(model,'predict_proba'):
            confidence=round(float(max(model.predict_proba(X)[0]))*100,1)
        db=get_db()
        cursor=db.execute("""
            INSERT INTO loan_applications
            (created_at,gender,married,dependents,education,self_employed,property_area,
             applicant_income,coapplicant_income,loan_amount,loan_amount_term,credit_history,
             prediction,confidence,model_used)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            form.get('Gender'),form.get('Married'),form.get('Dependents'),
            form.get('Education'),form.get('Self_Employed'),form.get('Property_Area'),
            float(form.get('ApplicantIncome',0)),float(form.get('CoapplicantIncome',0)),
            float(form.get('LoanAmount',0)),float(form.get('Loan_Amount_Term',360)),
            float(form.get('Credit_History',0)),result,confidence,meta['best_model']))
        db.commit()
        return render_template('result.html',result=result,confidence=confidence,
            form_data=form,record_id=cursor.lastrowid,meta=meta)
    except Exception as e:
        flash(f'Error: {e}','error')
        return redirect(url_for('index'))

@app.route('/history')
def history():
    _,meta=get_model()
    page=request.args.get('page',1,type=int)
    items,total,pages=db_paginate(page=page)
    stats=db_stats()
    return render_template('history.html',applications=items,
        total=total,approved=stats['approved'],rejected=stats['rejected'],
        page=page,pages=pages,has_prev=(page>1),has_next=(page<pages),
        prev_num=page-1,next_num=page+1,
        page_range=iter_pages(page,pages),
        income_buckets=json.dumps(db_income_buckets()),meta=meta)

@app.route('/application/<int:record_id>')
def application_detail(record_id):
    db=get_db()
    row=db.execute("SELECT * FROM loan_applications WHERE id=?",(record_id,)).fetchone()
    if not row: flash('Not found.','error'); return redirect(url_for('history'))
    _,meta=get_model()
    return render_template('detail.html',record=dict(row),meta=meta)

@app.route('/delete/<int:record_id>',methods=['POST'])
def delete_application(record_id):
    db=get_db()
    db.execute("DELETE FROM loan_applications WHERE id=?",(record_id,)); db.commit()
    flash('Record deleted.','success')
    return redirect(url_for('history'))

@app.route('/api/stats')
def api_stats(): return jsonify(db_stats())

if __name__=='__main__':
    init_db(); get_model()
    print("✅ LoanIQ → http://localhost:5000",flush=True)
    app.run(debug=True,host='0.0.0.0',port=5000)
