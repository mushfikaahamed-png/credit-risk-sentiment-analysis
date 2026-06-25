import pandas as pd
import matplotlib.pyplot as plt

# --- LOAD DATA ---
df = pd.read_csv("macro_data.csv", index_col=0, parse_dates=True)

# --- PLOT 1: Sentiment vs Delinquency over time ---
fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.set_xlabel("Year")
ax1.set_ylabel("Consumer Sentiment", color="steelblue")
ax1.plot(df.index, df["sentiment"], color="steelblue", linewidth=2, label="Consumer Sentiment")
ax1.tick_params(axis="y", labelcolor="steelblue")

ax2 = ax1.twinx()
ax2.set_ylabel("Credit Card Delinquency Rate (%)", color="crimson")
ax2.plot(df.index, df["delinquency"], color="crimson", linewidth=2, label="Delinquency Rate")
ax2.tick_params(axis="y", labelcolor="crimson")

plt.title("Consumer Sentiment vs Credit Card Delinquency Rate (1980–2024)", fontsize=14)
fig.tight_layout()
plt.savefig("plot1_sentiment_vs_delinquency.png", dpi=150)
plt.show()
print("✅ Plot 1 saved")

# --- PLOT 2: All variables over time ---
fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

variables = ["sentiment", "delinquency", "unemployment", "cpi", "fed_funds_rate"]
colors = ["steelblue", "crimson", "darkorange", "green", "purple"]
titles = ["Consumer Sentiment", "Delinquency Rate (%)", "Unemployment Rate (%)", "CPI (Inflation)", "Fed Funds Rate (%)"]

for i, (var, color, title) in enumerate(zip(variables, colors, titles)):
    axes[i].plot(df.index, df[var], color=color, linewidth=1.5)
    axes[i].set_ylabel(title, fontsize=9)
    axes[i].grid(True, alpha=0.3)

    # Mark key crises
    for year, label in [("2008", "GFC"), ("2020", "COVID")]:
        axes[i].axvline(pd.Timestamp(year), color="black", linestyle="--", alpha=0.4)

axes[0].set_title("All Macro Indicators (1980–2024)", fontsize=14)
plt.tight_layout()
plt.savefig("plot2_all_indicators.png", dpi=150)
plt.show()
print("✅ Plot 2 saved")

# --- PLOT 3: Correlation heatmap ---
import numpy as np

corr = df.corr()
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
plt.colorbar(im)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)

for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=10)

plt.title("Correlation Matrix of Macro Indicators", fontsize=14)
plt.tight_layout()
plt.savefig("plot3_correlation_heatmap.png", dpi=150)
plt.show()
print("✅ Plot 3 saved")

# --- PRINT SUMMARY STATS ---
print("\n📊 Summary Statistics:")
print(df.describe().round(2))
