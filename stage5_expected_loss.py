import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- LOAD DATA ---
df = pd.read_csv("macro_data.csv", index_col=0, parse_dates=True)

# --- RECREATE THE MODEL FROM STAGE 4 ---
median_delinq = df["delinquency"].median()
df["high_risk"] = (df["delinquency"] > median_delinq).astype(int)
df["sentiment_lag1"] = df["sentiment"].shift(1)
df["sentiment_lag2"] = df["sentiment"].shift(2)
df["unemployment_lag1"] = df["unemployment"].shift(1)
df["cpi_lag1"] = df["cpi"].shift(1)
df["fed_funds_lag1"] = df["fed_funds_rate"].shift(1)
df = df.dropna()

features = ["unemployment_lag1", "cpi_lag1", "fed_funds_lag1",
            "sentiment_lag1", "sentiment_lag2"]
X = df[features]
y = df["high_risk"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression()
model.fit(X_scaled, y)

# --- PROBABILITY OF DEFAULT (PD) ---
# This is the core of credit risk modelling
df["PD"] = model.predict_proba(X_scaled)[:, 1]

# --- LOSS GIVEN DEFAULT (LGD) & EXPOSURE AT DEFAULT (EAD) ---
# Standard assumptions used in Basel III framework
LGD = 0.45   # 45% of loan lost if default occurs (industry standard)
EAD = 1.0    # Normalised to 1 unit of exposure

# --- EXPECTED LOSS ---
# EL = PD x LGD x EAD  (the fundamental credit risk formula)
df["EL"] = df["PD"] * LGD * EAD

# --- PRINT SUMMARY ---
print("=" * 55)
print("EXPECTED LOSS MODEL (Basel III Framework)")
print("=" * 55)
print(f"\nLGD assumption: {LGD:.0%} (Basel III standard)")
print(f"EAD assumption: normalised to 1.0")
print(f"\nAverage PD across all quarters: {df['PD'].mean():.1%}")
print(f"Average Expected Loss:          {df['EL'].mean():.1%}")
print(f"\nHighest risk periods (top 5):")
top5 = df[["PD", "EL", "sentiment", "delinquency"]].nlargest(5, "EL")
print(top5.round(4))

# --- PLOT 1: PD and EL over time ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(df.index, df["PD"], color="crimson", linewidth=2)
ax1.axhline(df["PD"].mean(), color="black", linestyle="--", alpha=0.5, label="Average PD")
ax1.set_ylabel("Probability of Default (PD)")
ax1.set_title("Credit Risk Over Time: PD and Expected Loss", fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

# shade recessions
for start, end in [("2007-01-01", "2009-06-01"), ("2020-01-01", "2020-09-01")]:
    ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, color="red")
    ax2.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, color="red")

ax2.plot(df.index, df["EL"], color="darkorange", linewidth=2)
ax2.axhline(df["EL"].mean(), color="black", linestyle="--", alpha=0.5, label="Average EL")
ax2.set_ylabel("Expected Loss (EL = PD × LGD × EAD)")
ax2.set_xlabel("Year")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plot6_expected_loss.png", dpi=150)
plt.show()

# --- PLOT 2: Sentiment vs PD ---
fig, ax = plt.subplots(figsize=(10, 5))
scatter = ax.scatter(df["sentiment"], df["PD"],
                     c=df["delinquency"], cmap="RdYlGn_r",
                     alpha=0.7, s=60)
plt.colorbar(scatter, label="Actual Delinquency Rate (%)")
ax.set_xlabel("Consumer Sentiment Index")
ax.set_ylabel("Predicted Probability of Default")
ax.set_title("Sentiment vs Predicted Default Risk\n(colour = actual delinquency)", fontsize=12)
plt.tight_layout()
plt.savefig("plot7_sentiment_vs_pd.png", dpi=150)
plt.show()

print("\nStage 5 complete! All plots saved.")
print("\nProject summary:")
print("  stage1_data_collection.py  - Data from FRED API")
print("  stage2_eda.py              - Exploratory analysis")
print("  stage3_econometrics.py     - VAR & Granger causality")
print("  stage4_prediction.py       - Logistic regression (86% accuracy)")
print("  stage5_expected_loss.py    - Basel III Expected Loss model")