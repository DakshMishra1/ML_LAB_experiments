import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score, roc_curve, auc
)

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ── OUTPUT FOLDER ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 58)
print("       CUSTOMER CHURN PREDICTION EXPERIMENT")
print("=" * 58)

# ── 1. LOAD KAGGLE CSV ────────────────────────────────────────
CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CSV_PATH = os.path.join(SCRIPT_DIR, CSV_NAME)

print(f"\n📂 Loading dataset: {CSV_PATH}")

# Read CSV manually (no pandas)
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    raw_rows = [row for row in reader]

print(f"✅ Loaded {len(raw_rows)} customer records.")
print(f"   Columns: {list(raw_rows[0].keys())}")

# ── 2. PARSE & CLEAN DATA ─────────────────────────────────────
print("\n🔧 Parsing and cleaning data...")

def parse_yes_no(val):
    val = val.strip().lower()
    if val == "yes":   return 1
    if val == "no":    return 0
    return -1   # unknown / No internet service / No phone service

def parse_contract(val):
    val = val.strip().lower()
    if "month" in val:  return 0
    if "one"   in val:  return 1
    if "two"   in val:  return 2
    return 0

def parse_payment(val):
    val = val.strip().lower()
    if "electronic" in val: return 0
    if "mailed"     in val: return 1
    if "bank"       in val: return 2
    if "credit"     in val: return 3
    return 0

rows_X, rows_y = [], []
skipped = 0

for row in raw_rows:
    try:
        # Target
        churn = 1 if row["Churn"].strip().lower() == "yes" else 0

        # Numeric features
        tenure          = float(row["tenure"])
        monthly_charges = float(row["MonthlyCharges"])
        total_charges   = float(row["TotalCharges"]) if row["TotalCharges"].strip() else monthly_charges

        # Binary features
        gender          = 1 if row["gender"].strip().lower() == "male" else 0
        senior          = int(row["SeniorCitizen"])
        partner         = parse_yes_no(row["Partner"])
        dependents      = parse_yes_no(row["Dependents"])
        phone_service   = parse_yes_no(row["PhoneService"])
        paperless_bill  = parse_yes_no(row["PaperlessBilling"])
        online_security = parse_yes_no(row["OnlineSecurity"])
        tech_support    = parse_yes_no(row["TechSupport"])
        streaming_tv    = parse_yes_no(row["StreamingTV"])

        # Categorical features
        contract        = parse_contract(row["Contract"])
        payment         = parse_payment(row["PaymentMethod"])

        rows_X.append([
            tenure, monthly_charges, total_charges,
            gender, senior, partner, dependents,
            phone_service, paperless_bill, online_security,
            tech_support, streaming_tv, contract, payment
        ])
        rows_y.append(churn)

    except (ValueError, KeyError):
        skipped += 1

X = np.array(rows_X, dtype=float)
y = np.array(rows_y, dtype=int)

feature_names = [
    "Tenure",  "Monthly Charges", "Total Charges",
    "Gender",  "Senior Citizen",  "Partner",       "Dependents",
    "Phone Service", "Paperless Bill", "Online Security",
    "Tech Support",  "Streaming TV",   "Contract",  "Payment Method"
]

n_churn  = int(y.sum())
n_stayed = int((y == 0).sum())
churn_pct = n_churn / len(y) * 100

print(f"✅ Parsed {len(X)} records  (skipped {skipped} incomplete rows)")
print(f"\n   Stayed  (0) : {n_stayed}  ({100-churn_pct:.1f}%)")
print(f"   Churned (1) : {n_churn}  ({churn_pct:.1f}%)")

# Feature averages by group
print(f"\n--- Feature Averages by Group ---")
print(f"  {'Feature':<20} {'Stayed':>10} {'Churned':>10}")
print("  " + "-" * 42)
for i, name in enumerate(feature_names):
    avg_stay  = X[y == 0, i].mean()
    avg_churn = X[y == 1, i].mean()
    print(f"  {name:<20} {avg_stay:>10.2f} {avg_churn:>10.2f}")

# ── 3. PREPROCESS ─────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✂️  Train: {len(X_train)} | Test: {len(X_test)}")

# ── 4. TRAIN MODELS ───────────────────────────────────────────
models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"       : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
print("\n" + "=" * 58)
print("               MODEL RESULTS")
print("=" * 58)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred    = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
    results[name] = {
        "model"   : model,
        "y_pred"  : y_pred,
        "accuracy": accuracy,
        "cv_mean" : cv_scores.mean(),
        "cv_std"  : cv_scores.std(),
    }
    print(f"\n🔹 {name}  —  Test Acc: {accuracy*100:.2f}%"
          f"  |  CV: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(classification_report(y_test, y_pred,
                                target_names=["Stayed", "Churned"],
                                zero_division=0))

# ── 5. PLOTS ──────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_pred = results[best_name]["y_pred"]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Customer Churn Prediction — Telco Dataset", fontsize=15, fontweight="bold")

# Plot 1 — Churn Distribution
ax = axes[0, 0]
bars = ax.bar(["Stayed", "Churned"], [n_stayed, n_churn],
              color=["#2ecc71", "#e74c3c"], edgecolor="white", width=0.5)
ax.set_title("Customer Distribution", fontweight="bold")
ax.set_ylabel("Number of Customers")
for bar, val in zip(bars, [n_stayed, n_churn]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            str(val), ha="center", fontweight="bold", fontsize=11)
ax.set_ylim(0, max(n_stayed, n_churn) * 1.15)

# Plot 2 — Confusion Matrix
ax = axes[0, 1]
cm = confusion_matrix(y_test, best_pred)
ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"]).plot(
    ax=ax, colorbar=False, cmap="RdYlGn"
)
ax.set_title(f"Confusion Matrix\n{best_name}", fontweight="bold")

# Plot 3 — ROC Curves
ax = axes[1, 0]
line_colors = ["#3498db", "#e74c3c", "#2ecc71"]
for (name, res), color in zip(results.items(), line_colors):
    m = res["model"]
    prob = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") \
           else m.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models", fontweight="bold")
ax.legend(fontsize=9)

# Plot 4 — Feature Importance
ax = axes[1, 1]
rf_model    = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
sorted_idx  = np.argsort(importances)
ax.barh(
    [feature_names[i] for i in sorted_idx],
    importances[sorted_idx],
    color=["#e74c3c" if i == sorted_idx[-1] else "#3498db" for i in sorted_idx],
    edgecolor="white"
)
ax.set_title("Feature Importance\n(Random Forest)", fontweight="bold")
ax.set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "churn_results.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Plots saved.")

# ── 6. RISK SEGMENTATION ──────────────────────────────────────
print("\n" + "=" * 58)
print("         CUSTOMER RISK SEGMENTATION")
print("=" * 58)

best_model = results[best_name]["model"]
all_probs  = best_model.predict_proba(X_scaled)[:, 1]

high_risk   = int((all_probs >= 0.70).sum())
medium_risk = int(((all_probs >= 0.40) & (all_probs < 0.70)).sum())
low_risk    = int((all_probs < 0.40).sum())

print(f"\n  🔴 High Risk   (≥70% churn prob) : {high_risk:>5} customers")
print(f"  🟡 Medium Risk (40–70%)           : {medium_risk:>5} customers")
print(f"  🟢 Low Risk    (<40%)             : {low_risk:>5} customers")
print(f"\n  💡 Target the {high_risk} HIGH RISK customers for retention offers!")

# ── 7. LIVE PREDICTOR ─────────────────────────────────────────
print("\n" + "=" * 58)
print("            LIVE CHURN PREDICTOR")
print("=" * 58)

def predict_churn(tenure, monthly, total, gender, senior, partner,
                  dependents, phone, paperless, security,
                  tech_support, streaming, contract, payment):
    customer = np.array([[tenure, monthly, total, gender, senior, partner,
                          dependents, phone, paperless, security,
                          tech_support, streaming, contract, payment]])
    scaled   = scaler.transform(customer)
    pred     = best_model.predict(scaled)[0]
    prob     = best_model.predict_proba(scaled)[0][1] * 100
    risk     = "🔴 HIGH" if prob >= 70 else "🟡 MEDIUM" if prob >= 40 else "🟢 LOW"
    label    = "⚠️  WILL CHURN" if pred == 1 else "✅ WILL STAY"
    print(f"\n   Tenure={tenure}mo  Monthly=₹{monthly}  Contract="
          f"{'M2M' if contract==0 else '1yr' if contract==1 else '2yr'}")
    print(f"   → {label}  |  Churn Prob: {prob:.1f}%  |  Risk: {risk}")

print("\n👤 Customer 1 — New, high bill, month-to-month:")
predict_churn(2,  95, 190,  1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0)

print("\n👤 Customer 2 — Long-term, 2-year contract, low bill:")
predict_churn(60, 45, 2700, 0, 0, 1, 1, 1, 0, 1, 1, 1, 2, 2)

print("\n👤 Customer 3 — Mid-tenure, moderate plan:")
predict_churn(18, 70, 1260, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1)

print("\n👤 Customer 4 — Senior, no security, no support:")
predict_churn(5, 105, 525,  0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0)

# ── 8. SUMMARY ────────────────────────────────────────────────
print("\n" + "=" * 58)
print("               FINAL SUMMARY")
print("=" * 58)
print(f"  {'Model':<22} {'Test Acc':>9} {'CV Acc':>14}")
print("  " + "-" * 48)
for name, res in results.items():
    print(f"  {name:<22} {res['accuracy']*100:>8.2f}%"
          f"  {res['cv_mean']*100:>7.2f}% ± {res['cv_std']*100:.2f}%")
top3_idx = np.argsort(rf_model.feature_importances_)[::-1][:3]
print(f"\n🏆 Best Model : {best_name}")
print(f"\n📌 Top 3 Churn Factors (from Random Forest):")
for rank, idx in enumerate(top3_idx, 1):
    print(f"   {rank}. {feature_names[idx]}  ({rf_model.feature_importances_[idx]:.3f})")
print("\n✅ Experiment complete!")