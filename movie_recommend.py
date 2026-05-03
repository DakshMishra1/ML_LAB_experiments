import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore")

# ── OUTPUT FOLDER ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("     MOVIE RECOMMENDATION SYSTEM EXPERIMENT")
print("=" * 55)

# ── 1. DATASET ────────────────────────────────────────────────
# Built-in sample of 10 users and 15 movies with ratings (1–5)
# 0 means the user has NOT rated that movie

ratings_data = {
    "User":            [1,   2,   3,   4,   5,   6,   7,   8,   9,   10],
    "Inception":       [5,   4,   0,   2,   5,   0,   4,   3,   0,   5 ],
    "The Dark Knight": [4,   5,   4,   0,   5,   3,   0,   5,   4,   4 ],
    "Interstellar":    [5,   4,   5,   3,   0,   4,   5,   0,   5,   5 ],
    "Avengers":        [3,   0,   4,   5,   4,   5,   3,   4,   5,   0 ],
    "Titanic":         [2,   3,   0,   5,   3,   4,   2,   5,   3,   2 ],
    "The Matrix":      [5,   5,   4,   0,   5,   0,   5,   4,   0,   5 ],
    "Iron Man":        [4,   0,   5,   4,   3,   5,   4,   3,   5,   3 ],
    "Forrest Gump":    [3,   4,   3,   5,   0,   5,   3,   5,   4,   3 ],
    "The Godfather":   [4,   5,   4,   3,   4,   0,   5,   4,   3,   4 ],
    "Toy Story":       [2,   3,   0,   5,   2,   5,   2,   4,   5,   2 ],
    "Joker":           [5,   4,   5,   0,   5,   3,   5,   0,   4,   5 ],
    "Spider-Man":      [3,   0,   4,   5,   3,   4,   3,   5,   4,   3 ],
    "The Lion King":   [2,   3,   2,   4,   2,   5,   2,   4,   3,   2 ],
    "Gladiator":       [4,   5,   4,   3,   4,   0,   5,   3,   4,   4 ],
    "Shrek":           [2,   3,   0,   5,   2,   4,   2,   5,   4,   2 ],
}

df = pd.DataFrame(ratings_data).set_index("User")
movies = df.columns.tolist()

print(f"\n📊 Dataset: {df.shape[0]} Users × {df.shape[1]} Movies")
print(f"\n--- Ratings Matrix (0 = not rated) ---")
print(df.to_string())

# ── 2. EDA PLOTS ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Movie Recommendation — EDA", fontsize=14, fontweight="bold")

# Plot 1 — Average rating per movie
ax = axes[0]
avg_ratings = df.replace(0, np.nan).mean()
colors = ["#3498db" if r >= avg_ratings.mean() else "#95a5a6" for r in avg_ratings]
bars = ax.barh(avg_ratings.index, avg_ratings.values, color=colors, edgecolor="white")
ax.set_xlabel("Average Rating")
ax.set_title("Average Rating per Movie", fontweight="bold")
ax.axvline(avg_ratings.mean(), color="#e74c3c", linestyle="--", lw=1.5, label="Overall Mean")
ax.legend()
for bar, val in zip(bars, avg_ratings.values):
    ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", fontsize=8)

# Plot 2 — Ratings heatmap
ax = axes[1]
display_df = df.replace(0, np.nan)
im = ax.imshow(display_df.values, cmap="YlOrRd", aspect="auto",
               vmin=1, vmax=5)
ax.set_xticks(range(len(movies)))
ax.set_xticklabels(movies, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(df)))
ax.set_yticklabels([f"User {u}" for u in df.index], fontsize=8)
ax.set_title("User–Movie Ratings Heatmap\n(blank = not rated)", fontweight="bold")
plt.colorbar(im, ax=ax, label="Rating (1–5)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "recommendation_eda.png"), dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ EDA plot saved.")

# ── 3. COLLABORATIVE FILTERING ────────────────────────────────
# Step 1: Fill unrated movies with each user's average rating
df_filled = df.copy().astype(float)
for user in df_filled.index:
    user_mean = df_filled.loc[user].replace(0, np.nan).mean()
    df_filled.loc[user] = df_filled.loc[user].replace(0, user_mean)

# Step 2: Compute cosine similarity between all users
similarity_matrix = cosine_similarity(df_filled)
sim_df = pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)

print("\n--- User Similarity Matrix (Cosine Similarity) ---")
print(sim_df.round(2).to_string())

# ── 4. RECOMMEND FUNCTION ─────────────────────────────────────
def recommend_movies(target_user, top_n_users=3, top_n_movies=5):
    """
    Recommend movies to target_user using User-Based Collaborative Filtering.
    - Finds top_n_users most similar users
    - Recommends movies they liked but target_user hasn't seen
    """
    print(f"\n{'='*55}")
    print(f"  🎬 Recommendations for User {target_user}")
    print(f"{'='*55}")

    # Movies already rated by target user
    seen_movies = [m for m in movies if df.loc[target_user, m] > 0]
    unseen_movies = [m for m in movies if df.loc[target_user, m] == 0]

    print(f"\n  ✅ Already watched : {', '.join(seen_movies)}")
    print(f"  ❓ Not yet watched : {', '.join(unseen_movies)}")

    # Get top similar users (excluding self)
    user_sim = sim_df[target_user].drop(target_user).sort_values(ascending=False)
    top_users = user_sim.head(top_n_users)

    print(f"\n  👥 Top {top_n_users} similar users:")
    for u, score in top_users.items():
        print(f"     User {u}  →  similarity: {score:.3f}")

    # Score each unseen movie using weighted average of similar users' ratings
    scores = {}
    for movie in unseen_movies:
        weighted_sum = 0
        sim_sum      = 0
        for sim_user, sim_score in top_users.items():
            rating = df.loc[sim_user, movie]
            if rating > 0:
                weighted_sum += sim_score * rating
                sim_sum      += sim_score
        if sim_sum > 0:
            scores[movie] = weighted_sum / sim_sum

    if not scores:
        print("\n  ⚠️  No recommendations found (user has seen all movies).")
        return

    # Sort by score and show top recommendations
    recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n_movies]

    print(f"\n  🌟 Top {top_n_movies} Recommended Movies:")
    for rank, (movie, score) in enumerate(recommended, 1):
        stars = "⭐" * round(score)
        print(f"     {rank}. {movie:<20} (predicted rating: {score:.2f})  {stars}")

    return recommended

# ── 5. GET RECOMMENDATIONS ────────────────────────────────────
recommend_movies(target_user=1)
recommend_movies(target_user=5)
recommend_movies(target_user=8)

# ── 6. SIMILARITY HEATMAP ─────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.imshow(sim_df.values, cmap="Blues", vmin=0.9, vmax=1.0)
plt.colorbar(label="Cosine Similarity")
plt.xticks(range(len(sim_df)), [f"U{u}" for u in sim_df.columns], fontsize=9)
plt.yticks(range(len(sim_df)), [f"U{u}" for u in sim_df.index], fontsize=9)
for i in range(len(sim_df)):
    for j in range(len(sim_df)):
        plt.text(j, i, f"{sim_df.values[i,j]:.2f}",
                 ha="center", va="center", fontsize=7,
                 color="white" if sim_df.values[i, j] > 0.97 else "black")
plt.title("User Similarity Matrix (Cosine Similarity)", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "user_similarity.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Similarity heatmap saved.")

# ── 7. SUMMARY ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("              EXPERIMENT SUMMARY")
print("=" * 55)
print(f"  Technique     : User-Based Collaborative Filtering")
print(f"  Similarity    : Cosine Similarity")
print(f"  Total Users   : {df.shape[0]}")
print(f"  Total Movies  : {df.shape[1]}")
print(f"  Avg Rating    : {df.replace(0, np.nan).mean().mean():.2f} / 5.00")
print("\n✅ Experiment complete!")