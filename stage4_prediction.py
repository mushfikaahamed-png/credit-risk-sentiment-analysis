import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- LOAD DATA ---
df = pd.read_csv("macro_data.csv", index_col=0, parse_dates=True)

# --- CREATE TARGET VARIABLE ---
# "High risk" quarter = delinquency rate above its historical median
median_delinq = df["delinquency"].median()
df["high_risk"] = (df["delinquency"] > median_delinq).astype(int)
print(f"Median delinquency rate: {median_delinq:.2f}%")
print(f"High risk quarters: {df['high_risk'].sum()} out of {len(df)}")

# --- CREATE FEATURES ---
# Use lagged values so we're genuinely predicting the FUTURE
df["sentiment_lag1"] = df["sentiment"].shift(1)
df["sentiment_lag2"] = df["sentiment"].shift(2)
df["unemployment_lag1"] = df["unemployment"].shift(1)
df["cpi_lag1"] = df["cpi"].shift(1)
df["fed_funds_lag1"] = df["fed_funds_rate"].shift(1)
df = df.dropna()

# --- WITHOUT SENTIMENT ---
features_no_sentiment = ["unemployment_lag1", "cpi_lag1", "fed_funds_lag1"]
X_no_sent = df[features_no_sentiment]
y = df["high_risk"]

scaler = StandardScaler()
X_no_sent_scaled = scaler.fit_transform(X_no_sent)

model_no_sent = LogisticRegression()
model_no_sent.fit(X_no_sent_scaled, y)
score_no_sent = model_no_sent.score(X_no_sent_scaled, y)

# --- WITH SENTIMENT ---
features_with_sentiment = ["unemployment_lag1", "cpi_lag1", "fed_funds_lag1",
                            "sentiment_lag1", "sentiment_lag2"]
X_with_sent = df[features_with_sentiment]
X_with_sent_scaled = scaler.fit_transform(X_with_sent)

model_with_sent = LogisticRegression()
model_with_sent.fit(X_with_sent_scaled, y)
score_with_sent = model_with_sent.score(X_with_sent_scaled, y)

# --- COMPARE ---
print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)
print(f"Accuracy WITHOUT sentiment: {score_no_sent:.1%}")
print(f"Accuracy WITH sentiment:    {score_with_sent:.1%}")
improvement = score_with_sent - score_no_sent
print(f"Improvement from sentiment: {improvement:+.1%}")

# --- DETAILED REPORT ---
print("\nClassification Report (WITH sentiment):")
print(classification_report(y, model_with_sent.predict(X_with_sent_scaled),
      target_names=["Low Risk", "High Risk"]))

# --- FEATURE IMPORTANCE PLOT ---
coefs = pd.Series(model_with_sent.coef_[0], index=features_with_sentiment)
coefs = coefs.sort_values()

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["crimson" if c < 0 else "steelblue" for c in coefs]
coefs.plot(kind="barh", ax=ax, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Feature Importance: What Predicts High Credit Risk?", fontsize=13)
ax.set_xlabel("Coefficient (positive = increases risk)")
plt.tight_layout()
plt.savefig("plot5_feature_importance.png", dpi=150)
plt.show()
print("\nStage 4 complete!")