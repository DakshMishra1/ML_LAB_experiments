import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score
)

import warnings
warnings.filterwarnings("ignore")

# -- OUTPUT FOLDER
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 58)
print("        FAKE NEWS DETECTION EXPERIMENT")
print("=" * 58)

FAKE_PATH = os.path.join(SCRIPT_DIR, "Fake.csv")
TRUE_PATH = os.path.join(SCRIPT_DIR, "True.csv")

for path in [FAKE_PATH, TRUE_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n File not found: {path}\n"
            f"   Download from: https://www.kaggle.com/datasets/"
            f"clmentbisaillon/fake-and-real-news-dataset\n"
            f"   Place 'Fake.csv' and 'True.csv' in the same folder as this script."
        )

print("\nLoading Fake.csv and True.csv ...")

def load_csv(path, label, max_rows=5000):
    rows = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            title    = row.get("title", "").strip()
            text     = row.get("text",  "").strip()
            combined = (title + " " + text).strip()
            if len(combined) > 30:
                rows.append((combined, label))
    return rows

fake_data = load_csv(FAKE_PATH, label=1, max_rows=5000)
real_data = load_csv(TRUE_PATH, label=0, max_rows=5000)

min_size  = min(len(fake_data), len(real_data))
fake_data = fake_data[:min_size]
real_data = real_data[:min_size]

all_data  = fake_data + real_data
texts     = [d[0] for d in all_data]
labels    = [d[1] for d in all_data]

n_fake = sum(1 for l in labels if l == 1)
n_real = sum(1 for l in labels if l == 0)

print(f"Loaded {len(texts)} articles  (balanced)")
print(f"   Real News (0) : {n_real}")
print(f"   Fake News (1) : {n_fake}")

# -- 2. TEXT PREPROCESSING
STOPWORDS = {
    "a","an","the","is","it","in","on","at","to","for","of","and",
    "or","but","not","are","was","were","be","been","by","with","as",
    "this","that","from","have","has","had","do","does","did","will",
    "can","could","would","should","its","he","she","they","we","our",
    "said","also","one","two","new","us","just","like","more","than"
}

def clean(text):
    text  = text.lower()
    text  = re.sub(r"http\S+|www\S+", " ", text)
    text  = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split()
             if w not in STOPWORDS and len(w) > 2]
    return " ".join(words[:200])

cleaned = [clean(t) for t in texts]
print("\nText preprocessing done.")
print(f"   Sample: {cleaned[0][:80]}...")

# -- 3. TF-IDF + TRAIN/TEST SPLIT
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    cleaned, labels, test_size=0.2, random_state=42, stratify=labels
)

tfidf   = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                           sublinear_tf=True, min_df=2)
X_train = tfidf.fit_transform(X_train_raw)
X_test  = tfidf.transform(X_test_raw)

print(f"\nTrain: {len(y_train)} | Test: {len(y_test)}")
print(f"TF-IDF features: {X_train.shape[1]}")

# -- 4. TRAIN MODELS
models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Naive Bayes"         : MultinomialNB(alpha=0.1),
    "Linear SVM"          : LinearSVC(max_iter=2000, random_state=42),
}

results = {}
print("\n" + "=" * 58)
print("               MODEL RESULTS")
print("=" * 58)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv       = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    results[name] = {
        "model"   : model, "y_pred": y_pred,
        "accuracy": accuracy, "cv_mean": cv.mean(), "cv_std": cv.std()
    }
    print(f"\n  {name}  --  Acc: {accuracy*100:.2f}%"
          f"  |  CV: {cv.mean()*100:.2f}% +/- {cv.std()*100:.2f}%")
    print(classification_report(y_test, y_pred,
                                target_names=["Real", "Fake"],
                                zero_division=0))

# -- 5. PLOTS
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_pred = results[best_name]["y_pred"]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Fake News Detection -- Kaggle Dataset Results",
             fontsize=14, fontweight="bold")

ax = axes[0]
bars = ax.bar(["Real News", "Fake News"], [n_real, n_fake],
              color=["#2ecc71", "#e74c3c"], edgecolor="white", width=0.5)
ax.set_title("Dataset Distribution", fontweight="bold")
ax.set_ylabel("Number of Articles")
ax.set_ylim(0, max(n_real, n_fake) * 1.2)
for bar, val in zip(bars, [n_real, n_fake]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            str(val), ha="center", fontweight="bold", fontsize=12)

ax = axes[1]
cm = confusion_matrix(y_test, best_pred)
ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"]).plot(
    ax=ax, colorbar=False, cmap="RdYlGn"
)
ax.set_title(f"Confusion Matrix\n{best_name}", fontweight="bold")

ax = axes[2]
names  = list(results.keys())
accs   = [results[m]["accuracy"] * 100 for m in names]
colors = ["#e74c3c" if m == best_name else "#95a5a6" for m in names]
bars   = ax.bar(names, accs, color=colors, edgecolor="white", width=0.5)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(80, 102)
ax.set_title("Model Accuracy Comparison", fontweight="bold")
ax.tick_params(axis="x", rotation=15)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{acc:.2f}%", ha="center", fontweight="bold", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fake_news_results.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Results plot saved.")

# -- 6. TOP KEYWORDS
lr_model   = results["Logistic Regression"]["model"]
feat_names = np.array(tfidf.get_feature_names_out())
coefs      = lr_model.coef_[0]
top_fake   = feat_names[np.argsort(coefs)[-15:][::-1]]
top_real   = feat_names[np.argsort(coefs)[:15]]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Top Keywords -- Real vs Fake News", fontweight="bold", fontsize=13)

axes[0].barh(top_real[::-1], np.sort(np.abs(coefs[:15]))[::-1],
             color="#2ecc71", edgecolor="white")
axes[0].set_title("Top Real News Keywords", fontweight="bold")
axes[0].set_xlabel("Coefficient Magnitude")

axes[1].barh(top_fake[::-1], np.sort(coefs)[-15:],
             color="#e74c3c", edgecolor="white")
axes[1].set_title("Top Fake News Keywords", fontweight="bold")
axes[1].set_xlabel("Coefficient Value")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fake_news_keywords.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Keywords plot saved.")

# -- 7. LIVE DETECTOR
print("\n" + "=" * 58)
print("           LIVE FAKE NEWS DETECTOR")
print("=" * 58)

best_model = results[best_name]["model"]

def detect(headline):
    cleaned_input = clean(headline)
    features      = tfidf.transform([cleaned_input])
    pred          = best_model.predict(features)[0]
    label         = "FAKE NEWS" if pred == 1 else "REAL NEWS"
    print(f"\n  {headline[:70]}")
    print(f"   -> {label}")

detect("NASA successfully tests new rocket engine for Mars mission 2026")
detect("SHOCKING: Politicians drinking children energy to stay young!")
detect("Reserve bank increases repo rate by 25 basis points this quarter")
detect("Doctors HATE this man for curing diabetes with one kitchen spice")
detect("Scientists find microplastics in human blood for the first time")

# -- 8. SUMMARY
print("\n" + "=" * 58)
print("               FINAL SUMMARY")
print("=" * 58)
print(f"  {'Model':<22} {'Test Acc':>9} {'CV Acc':>14}")
print("  " + "-" * 48)
for name, res in results.items():
    print(f"  {name:<22} {res['accuracy']*100:>8.2f}%"
          f"  {res['cv_mean']*100:>7.2f}% +/- {res['cv_std']*100:.2f}%")
print(f"\n  Best Model : {best_name}")
print("Experiment complete!")