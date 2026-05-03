import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
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

# ── 1. NLTK SETUP ─────────────────────────────────────────────
print("=" * 55)
print("        SPAM DETECTION EXPERIMENT")
print("=" * 55)
nltk.download("stopwords", quiet=True)
print("✅ NLTK ready.")

CSV_PATH = os.path.join(SCRIPT_DIR, "spam.csv")
print(f"\n📂 Loading: {CSV_PATH}")

# Kaggle CSV columns: v1 = label (ham/spam), v2 = message text
df = pd.read_csv(CSV_PATH, encoding="latin-1", usecols=[0, 1])
df.columns = ["label_str", "text"]
df["label"] = df["label_str"].map({"ham": 0, "spam": 1})
df = df[["text", "label"]].dropna().reset_index(drop=True)

print(f"\n📊 Total Messages : {len(df)}")
print(f"🚨 Spam  (1)      : {(df.label == 1).sum()}")
print(f"✅ Ham   (0)      : {(df.label == 0).sum()}")
print(f"\n--- Sample rows ---")
print(df.sample(5, random_state=1).to_string(index=False))

# ── 3. TEXT PREPROCESSING ─────────────────────────────────────
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text  = text.lower()
    text  = re.sub(r"[^a-z\s]", " ", text)        # remove punctuation & numbers
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

df["cleaned"] = df["text"].apply(clean_text)
print("\n✅ Text preprocessing done.")
print(f"   Original : {df['text'][0]}")
print(f"   Cleaned  : {df['cleaned'][0]}")

# ── 4. TRAIN / TEST SPLIT ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)
print(f"\n✂️  Train: {len(X_train)} | Test: {len(X_test)}")

# ── 5. TF-IDF FEATURE EXTRACTION ──────────────────────────────
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
print(f"✅ TF-IDF features: {X_train_tfidf.shape[1]}")

# ── 6. TRAIN MODELS ───────────────────────────────────────────
models = {
    "Naive Bayes"         : MultinomialNB(),
    "Logistic Regression" : LogisticRegression(max_iter=1000),
    "Linear SVM"          : LinearSVC(max_iter=2000),
}

results = {}
print("\n" + "=" * 55)
print("              MODEL RESULTS")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred   = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {"model": model, "y_pred": y_pred, "accuracy": accuracy}
    print(f"\n🔹 {name}  —  Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred,
                                target_names=["Ham", "Spam"],
                                zero_division=0))

# ── 7. PLOTS ──────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_pred = results[best_name]["y_pred"]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Spam Detection — SMS Spam Collection Dataset", fontsize=14, fontweight="bold")

# Plot 1 — Class Distribution
ax = axes[0]
counts = df["label"].value_counts().sort_index()
ax.pie(counts.values,
       labels=["Ham ✅", "Spam 🚨"],
       colors=["#2ecc71", "#e74c3c"],
       autopct="%1.1f%%", startangle=90,
       wedgeprops=dict(edgecolor="white", linewidth=2))
ax.set_title(f"Dataset Distribution\n(Total: {len(df)} messages)", fontweight="bold")

# Plot 2 — Confusion Matrix (best model)
ax = axes[1]
cm = confusion_matrix(y_test, best_pred)
ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"]).plot(
    ax=ax, colorbar=False, cmap="RdYlGn"
)
ax.set_title(f"Confusion Matrix\n{best_name}", fontweight="bold")

# Plot 3 — Model Accuracy Comparison
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
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{acc:.2f}%", ha="center", fontweight="bold", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "spam_detection_results.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Plot saved.")

# ── 8. LIVE SPAM DETECTOR ─────────────────────────────────────
print("\n" + "=" * 55)
print("           LIVE SPAM DETECTOR")
print("=" * 55)

best_model = results[best_name]["model"]

def detect_spam(message):
    cleaned  = clean_text(message)
    features = tfidf.transform([cleaned])
    pred     = best_model.predict(features)[0]
    result   = "🚨 SPAM" if pred == 1 else "✅ HAM (Not Spam)"
    print(f"\n📧 {message[:65]}")
    print(f"   → {result}")

detect_spam("Congratulations! You won a free iPhone. Click here to claim now!")
detect_spam("Hey, can we catch up for coffee tomorrow afternoon?")
detect_spam("URGENT: Your bank account is at risk. Verify your details now.")
detect_spam("Please submit your assignment before the deadline on Friday.")
detect_spam("Win cash prizes every week! Sign up for free and start earning.")

# ── 9. SUMMARY ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("              FINAL SUMMARY")
print("=" * 55)
for name in results:
    print(f"  {name:<22} → {results[name]['accuracy']*100:.2f}%")
print(f"\n🏆 Best Model : {best_name}")
print("✅ Experiment complete!")