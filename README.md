# ⚖️ DriftSafe — Algorithmic Bias & Drift Monitoring Dashboard

A post-deployment fairness monitoring system for loan approval ML models. DriftSafe detects when a frozen production model becomes unfair over time due to data drift — without any model changes — using Disparate Impact and Approval Gap metrics across multiple sensitive attributes.

---

## 🏗️ Architecture

```
Indian / US Loan Data → Feature Engineering → Logistic Regression (frozen at T0)
  → Time-windowed Fairness Evaluation (T0 → T1 → T2 → T3)
    → Disparate Impact + Approval Gap → Alert System → Flask Dashboard
```

**Key design principle:** The model is trained once on the baseline period (T0) and never retrained. Any fairness degradation observed in T1–T3 is caused purely by data drift, not model changes — demonstrating why continuous post-deployment monitoring is necessary.

---

## ⚙️ Tech Stack

| Component        | Technology                            |
|------------------|---------------------------------------|
| Language         | Python 3.11                           |
| ML Model         | scikit-learn (LogisticRegression)     |
| Data             | pandas, NumPy                         |
| Backend API      | Flask                                 |
| Dashboard        | HTML/CSS/JS (vanilla, no framework)   |
| Fairness Metrics | Custom (Disparate Impact, Approval Gap) |
| Dataset (India)  | 4,269 loan applications               |
| Dataset (USA)    | BLS / synthetic loan data             |

---

## 📊 Dataset

- **Source:** Indian loan application dataset (`IndianData_final.csv`)
- **Size:** 4,269 records, 21 features
- **Date range:** 2018–2024 (sorted chronologically into 4 time windows)
- **Time windows:** T0 (baseline), T1, T2, T3 — ~1,067 records each

**Monitored sensitive attributes:**
- Age Group: Young (18–35) vs Middle (36–65)
- Income Segment: Low (<₹40K) vs Medium (40–80K)
- Credit Score Group: Low (300–550) vs Medium (551–700)
- Product Type: Credit Card vs Personal Loan

---

## 📈 Results

### Quick Test — Threshold Drift (Age Group)

Run `python bias_quick_test.py` for an instant reproducible result:

| Metric              | Before Drift | After Drift |
|---------------------|--------------|-------------|
| Approval Rate (Young) | 24.9%      | 16.6%       |
| Approval Rate (Senior)| 25.7%      | 25.7%       |
| **Disparate Impact**  | **0.9696** ✅ | **0.6464** 🚨 |
| Approval Gap          | 0.0078     | 0.0910      |
| 80% Rule Status       | FAIR       | **VIOLATION** |

DI dropped by **−0.3232** after threshold tightening, crossing the 0.80 fairness threshold. Young applicants' approval rate fell from 24.9% → 16.6% while senior applicants were unaffected.

---

### Time-Series Monitoring — Logistic Regression Model (frozen at T0)

#### Age Group (Young vs Middle)

| Period | Young Approval | Middle Approval | DI     | Gap   | Status      |
|--------|---------------|-----------------|--------|-------|-------------|
| T0     | 22.3%         | 25.7%           | 0.867  | 0.034 | ⚠️ Baseline  |
| T1     | 12.1%         | 20.8%           | 0.582  | 0.087 | 🚨 ALERT    |
| T2     | 16.1%         | 25.0%           | 0.646  | 0.088 | 🚨 ALERT    |
| T3     | 14.0%         | 24.4%           | 0.575  | 0.104 | 🚨 ALERT    |

#### Income Segment (Low vs Medium)

| Period | Low Approval | Medium Approval | DI     | Gap   | Status      |
|--------|-------------|-----------------|--------|-------|-------------|
| T0     | 25.6%       | 22.6%           | 1.136  | 0.031 | ✅ FAIR     |
| T1     | 13.5%       | 20.6%           | 0.654  | 0.071 | 🚨 ALERT    |
| T2     | 13.8%       | 23.7%           | 0.584  | 0.099 | 🚨 ALERT    |
| T3     | 12.9%       | 22.5%           | 0.573  | 0.096 | 🚨 ALERT    |

> **Key finding:** The model is fair at deployment (T0). By T1, data drift causes DI to collapse below 0.80 for both age and income dimensions — with no model changes. This demonstrates that post-deployment monitoring is essential even for models that pass pre-deployment fairness checks.

---

## ▶️ How to Run

```bash
pip install pandas numpy scikit-learn flask

# Quick standalone fairness test (no server needed)
python bias_quick_test.py

# Full time-series pipeline (India dataset)
python india.py

# Launch Flask dashboard
python app.py
# Visit: http://localhost:5000
```

---

## 📁 Project Structure

```
driftsafe/
├── app.py                   # Flask backend (routes for India + USA dashboards)
├── india.py                 # Full fairness monitoring pipeline (4 attributes × 4 time windows)
├── usa_model.py             # USA loan fairness model
├── bias_quick_test.py       # Standalone quick evaluation script
├── templates/
│   ├── dashboard.html       # Landing page with country selector
│   ├── india.html           # India fairness dashboard
│   └── usa.html             # USA fairness dashboard
└── data/
    ├── IndianData_final.csv
    └── Dated_Loan_Approval_Data.xlsx
```

---

## 🔬 Fairness Metrics Explained

| Metric | Formula | Threshold | Meaning |
|--------|---------|-----------|---------|
| Disparate Impact | P(approval\|unprivileged) / P(approval\|privileged) | ≥ 0.80 (80% rule) | Core fairness signal |
| Approval Gap | \|rate_protected − rate_reference\| | < 0.10 | Absolute disparity |

The **80% rule** (EEOC guidelines) is the industry-standard threshold: a DI below 0.80 indicates the disadvantaged group receives favorable outcomes at less than 80% the rate of the advantaged group.

---

## 🔮 Future Work

- [ ] Statistical significance testing (bootstrap CI on DI estimates)
- [ ] Counterfactual fairness analysis
- [ ] Automated retraining triggers when DI falls below threshold
- [ ] Support for additional fairness metrics (Equalized Odds, Calibration)
- [ ] Real-time streaming data ingestion
