import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATASET
#     Place train.csv in the same folder as this script, OR update the path below.
# ──────────────────────────────────────────────────────────────────────────────
CSV_PATH = "train.csv"          # ← change path here if needed

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"\n[ERROR] '{CSV_PATH}' not found!\n"
        "Please download 'train.csv' from:\n"
        "  https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%%201\n"
        "and place it in the same directory as this script."
    )

df = pd.read_csv(CSV_PATH)
print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(df.head(3))
print("\nColumn types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# ──────────────────────────────────────────────────────────────────────────────
# 2.  STYLE SETUP
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f0f1a",
    "axes.facecolor":    "#1a1a2e",
    "axes.edgecolor":    "#555577",
    "axes.labelcolor":   "#e0e0ff",
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   "#ffffff",
    "xtick.color":       "#aaaacc",
    "ytick.color":       "#aaaacc",
    "text.color":        "#e0e0ff",
    "grid.color":        "#2a2a4a",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "DejaVu Sans",
})

COLORS   = ["#6c63ff", "#ff6584", "#43e8d8", "#ffd166", "#06d6a0", "#ef476f"]
GRADIENT = ["#6c63ff", "#8a85ff", "#a8a0ff"]   # for histogram bars

# ──────────────────────────────────────────────────────────────────────────────
# 3.  FIGURE  (2 rows × 3 cols)
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor("#0f0f1a")
fig.suptitle(
    "Titanic Dataset — Distribution Analysis",
    fontsize=20, fontweight="bold", color="#ffffff", y=1.01
)

# ── Helper ──────────────────────────────────────────────────────────────────
def style_ax(ax, title, xlabel, ylabel="Count"):
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.grid(axis="y", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Gender Distribution (Bar Chart)
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0, 0]
gender_counts = df["Sex"].value_counts()
bars = ax.bar(
    gender_counts.index, gender_counts.values,
    color=[COLORS[0], COLORS[1]], edgecolor="#ffffff22", linewidth=0.8, zorder=3
)
for bar, val in zip(bars, gender_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
        f"{val}\n({val/len(df)*100:.1f}%)",
        ha="center", va="bottom", fontsize=10, color="#ffffff", fontweight="bold"
    )
style_ax(ax, "Gender Distribution", "Gender")
ax.set_ylim(0, gender_counts.max() * 1.2)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Age Distribution (Histogram)
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0, 1]
age_data = df["Age"].dropna()
n, bins, patches = ax.hist(
    age_data, bins=20, edgecolor="#ffffff22", linewidth=0.6, zorder=3, color=COLORS[0]
)
# Gradient colouring
norm = plt.Normalize(n.min(), n.max())
for patch, height in zip(patches, n):
    patch.set_facecolor(plt.cm.plasma(norm(height)))

ax.axvline(age_data.mean(),   color=COLORS[1], linestyle="--", lw=1.8, label=f"Mean  {age_data.mean():.1f}")
ax.axvline(age_data.median(), color=COLORS[2], linestyle=":",  lw=1.8, label=f"Median {age_data.median():.1f}")
ax.legend(fontsize=9, framealpha=0.3, labelcolor="white")
style_ax(ax, "Age Distribution (Histogram)", "Age")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Passenger Class (Bar Chart)
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0, 2]
pclass_counts = df["Pclass"].value_counts().sort_index()
bars = ax.bar(
    [f"Class {i}" for i in pclass_counts.index],
    pclass_counts.values,
    color=COLORS[:3], edgecolor="#ffffff22", linewidth=0.8, zorder=3
)
for bar, val in zip(bars, pclass_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
        f"{val}", ha="center", va="bottom", fontsize=11,
        color="#ffffff", fontweight="bold"
    )
style_ax(ax, "Passenger Class Distribution", "Passenger Class")
ax.set_ylim(0, pclass_counts.max() * 1.18)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Survival Count (Bar Chart)
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1, 0]
survival_counts = df["Survived"].value_counts()
labels = {0: "Not Survived", 1: "Survived"}
bars = ax.bar(
    [labels[i] for i in survival_counts.index],
    survival_counts.values,
    color=[COLORS[1], COLORS[4]], edgecolor="#ffffff22", linewidth=0.8, zorder=3
)
for bar, val in zip(bars, survival_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
        f"{val}\n({val/len(df)*100:.1f}%)",
        ha="center", va="bottom", fontsize=10, color="#ffffff", fontweight="bold"
    )
style_ax(ax, "Survival Distribution", "Survival Status")
ax.set_ylim(0, survival_counts.max() * 1.2)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Fare Distribution (Histogram)
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1, 1]
fare_data = df["Fare"].dropna()
n, bins, patches = ax.hist(
    fare_data, bins=30, color=COLORS[3], edgecolor="#ffffff22", linewidth=0.6, zorder=3
)
norm = plt.Normalize(n.min(), n.max())
for patch, height in zip(patches, n):
    patch.set_facecolor(plt.cm.YlOrRd(norm(height)))

ax.axvline(fare_data.mean(),   color=COLORS[0], linestyle="--", lw=1.8, label=f"Mean  £{fare_data.mean():.1f}")
ax.axvline(fare_data.median(), color=COLORS[2], linestyle=":",  lw=1.8, label=f"Median £{fare_data.median():.1f}")
ax.legend(fontsize=9, framealpha=0.3, labelcolor="white")
style_ax(ax, "Fare Distribution (Histogram)", "Fare (£)")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Survival Rate by Gender (Grouped Bar Chart)
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1, 2]
survival_by_gender = df.groupby("Sex")["Survived"].value_counts(normalize=True).unstack() * 100
x     = np.arange(len(survival_by_gender.index))
width = 0.35

bars1 = ax.bar(x - width/2, survival_by_gender[0], width,
               label="Not Survived", color=COLORS[1], edgecolor="#ffffff22", linewidth=0.8, zorder=3)
bars2 = ax.bar(x + width/2, survival_by_gender[1], width,
               label="Survived",     color=COLORS[4], edgecolor="#ffffff22", linewidth=0.8, zorder=3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, color="#ffffff")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, color="#ffffff")

ax.set_xticks(x)
ax.set_xticklabels(["Female", "Male"])
ax.legend(fontsize=9, framealpha=0.3, labelcolor="white")
style_ax(ax, "Survival Rate by Gender (%)", "Gender", "Percentage (%)")
ax.set_ylim(0, 110)

# ──────────────────────────────────────────────────────────────────────────────
# 4.  SAVE & SHOW
# ──────────────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=3.0)
OUTPUT_FILE = "task01_titanic_distribution.png"
plt.savefig(OUTPUT_FILE, dpi=180, bbox_inches="tight", facecolor="#0f0f1a")
print(f"\n✅ Chart saved as '{OUTPUT_FILE}'")
plt.show()
