# ============================================================
#  DISEASE DIAGNOSIS FROM MEDICAL IMAGES
#  Applied ML Lab Experiment
#  Dataset : Digits dataset (sklearn built-in, simulates
#            medical image classification)
#  Techniques: SVM, Random Forest, Logistic Regression
#  Compatible: Python 3.13 (no download, no pandas)
#  Author  : B.Tech CSE - Semester 4
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score
)

import warnings
warnings.filterwarnings("ignore")

# ── OUTPUT FOLDER ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 58)
print("   DISEASE DIAGNOSIS FROM MEDICAL IMAGES")
print("=" * 58)

# ── 1. LOAD DATASET ───────────────────────────────────────────
# sklearn's Digits dataset: 1797 grayscale 8x8 images (64 pixels)
# Each image = handwritten digit 0-9
# For this experiment we simulate a BINARY medical diagnosis:
#   Class 0 = "No Disease"  (digits 0-4)
#   Class 1 = "Disease"     (digits 5-9)
# This mirrors the exact same ML pipeline used for real X-ray
# classification (image flattening -> feature extraction -> classify)

print("\n  Loading built-in image dataset (sklearn digits)...")
digits      = load_digits()
X_images    = digits.images          # shape: (1797, 8, 8)
X           = digits.data            # shape: (1797, 64) — flattened pixels
y_raw       = digits.target          # 0-9

# Convert to binary: 0-4 = No Disease (0),  5-9 = Disease (1)
y = (y_raw >= 5).astype(int)

n_total    = len(X)
n_healthy  = int((y == 0).sum())
n_diseased = int((y == 1).sum())

print(f"  Dataset    : {n_total} medical images  (8x8 pixels each)")
print(f"  No Disease (0) : {n_healthy}")
print(f"  Disease    (1) : {n_diseased}")
print(f"  Features   : {X.shape[1]} pixels per image")

# ── 2. SHOW SAMPLE IMAGES ─────────────────────────────────────
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
fig.suptitle("Sample Medical Images\n(8x8 Grayscale — Simulated X-Ray Grid)",
             fontsize=13, fontweight="bold")

for col, label in enumerate([0, 1]):
    label_name  = "No Disease" if label == 0 else "Disease"
    color       = "#2ecc71"    if label == 0 else "#e74c3c"
    sample_idxs = np.where(y == label)[0][:8]
    for row, idx in enumerate(sample_idxs):
        ax = axes[col, row]
        ax.imshow(X_images[idx], cmap="gray")
        ax.axis("off")
        if row == 0:
            ax.set_title(label_name, fontweight="bold",
                         color=color, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_images.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("\n  Sample images plot saved.")

# ── 3. PREPROCESS ─────────────────────────────────────────────
# Normalize pixel values (0-16 range in digits) to 0-1
X_norm   = X / 16.0

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_norm)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train: {len(X_train)} images | Test: {len(X_test)} images")

# ── 4. TRAIN MODELS ───────────────────────────────────────────
models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "SVM (RBF)"           : SVC(kernel="rbf", C=2.0, random_state=42,
                                probability=True),
    "Random Forest"       : RandomForestClassifier(n_estimators=100,
                                                   random_state=42),
}

results = {}
print("\n" + "=" * 58)
print("               MODEL RESULTS")
print("=" * 58)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv       = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
    results[name] = {
        "model": model, "y_pred": y_pred,
        "accuracy": accuracy, "cv_mean": cv.mean(), "cv_std": cv.std()
    }
    print(f"\n  {name}  --  Test Acc: {accuracy*100:.2f}%"
          f"  |  CV: {cv.mean()*100:.2f}% +/- {cv.std()*100:.2f}%")
    print(classification_report(y_test, y_pred,
                                target_names=["No Disease", "Disease"],
                                zero_division=0))

# ── 5. RESULT PLOTS ───────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_pred = results[best_name]["y_pred"]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Disease Diagnosis from Medical Images -- Results",
             fontsize=14, fontweight="bold")

# Plot 1 -- Class Distribution
ax = axes[0]
bars = ax.bar(["No Disease", "Disease"], [n_healthy, n_diseased],
              color=["#2ecc71", "#e74c3c"], edgecolor="white", width=0.5)
ax.set_title("Dataset Distribution", fontweight="bold")
ax.set_ylabel("Number of Images")
ax.set_ylim(0, max(n_healthy, n_diseased) * 1.2)
for bar, val in zip(bars, [n_healthy, n_diseased]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(val), ha="center", fontweight="bold", fontsize=12)

# Plot 2 -- Confusion Matrix
ax = axes[1]
cm = confusion_matrix(y_test, best_pred)
ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"]).plot(
    ax=ax, colorbar=False, cmap="RdYlGn"
)
ax.set_title(f"Confusion Matrix\n{best_name}", fontweight="bold")

# Plot 3 -- Accuracy Comparison
ax = axes[2]
names  = list(results.keys())
accs   = [results[m]["accuracy"] * 100 for m in names]
colors = ["#e74c3c" if m == best_name else "#95a5a6" for m in names]
bars   = ax.bar(names, accs, color=colors, edgecolor="white", width=0.5)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(85, 102)
ax.set_title("Model Accuracy Comparison", fontweight="bold")
ax.tick_params(axis="x", rotation=15)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{acc:.2f}%", ha="center", fontweight="bold", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "diagnosis_results.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  Results plot saved.")

# ── 6. PIXEL IMPORTANCE HEATMAP ───────────────────────────────
rf_model   = results["Random Forest"]["model"]
importance = rf_model.feature_importances_.reshape(8, 8)

plt.figure(figsize=(5, 4))
plt.imshow(importance, cmap="hot", interpolation="nearest")
plt.colorbar(label="Pixel Importance Score")
plt.title("Pixel Importance Heatmap\n(Which regions matter most?)",
          fontweight="bold")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pixel_importance.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  Pixel importance heatmap saved.")

# ── 7. LIVE DIAGNOSIS ON TEST IMAGES ──────────────────────────
print("\n" + "=" * 58)
print("          LIVE DIAGNOSIS ON TEST IMAGES")
print("=" * 58)

best_model  = results[best_name]["model"]
class_names = {0: "No Disease", 1: "Disease Detected"}

print()
for i in range(min(8, len(X_test))):
    img_feat = X_test[i].reshape(1, -1)
    pred     = best_model.predict(img_feat)[0]
    true     = y_test[i]
    prob     = best_model.predict_proba(img_feat)[0][pred] * 100
    correct  = "CORRECT" if pred == true else "WRONG"
    icon     = "✓" if pred == true else "✗"
    print(f"  [{icon}] Image {i+1}: "
          f"True={class_names[true]:<16}  "
          f"Pred={class_names[pred]:<16}  "
          f"Conf={prob:.1f}%  [{correct}]")

# ── 8. SUMMARY ────────────────────────────────────────────────
print("\n" + "=" * 58)
print("               FINAL SUMMARY")
print("=" * 58)
print(f"  {'Model':<22} {'Test Acc':>9} {'CV Acc':>14}")
print("  " + "-" * 48)
for name, res in results.items():
    print(f"  {name:<22} {res['accuracy']*100:>8.2f}%"
          f"  {res['cv_mean']*100:>7.2f}% +/- {res['cv_std']*100:.2f}%")
print(f"\n  Best Model : {best_name}")
print("\n  Experiment complete!")