import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

df = pd.read_csv("macro_data.csv", index_col=0, parse_dates=True)
df_diff = df.diff().dropna()

print("STATIONARITY TESTS")
for col in df.columns:
    result = adfuller(df[col].dropna())
    pval = result[1]
    status = "Stationary" if pval < 0.05 else "Non-stationary"
    print(f"{col}: p-value={pval:.4f} {status}")

print("\nGRANGER CAUSALITY TEST")
granger_data = df_diff[["delinquency", "sentiment"]]
results = grangercausalitytests(granger_data, maxlag=4, verbose=False)
for lag, result in results.items():
    pval = result[0]["ssr_ftest"][1]
    sig = "Significant!" if pval < 0.05 else "Not significant"
    print(f"Lag {lag}: p-value={pval:.4f} {sig}")

model = VAR(df_diff)
fit = model.fit(maxlags=4, ic="aic")
print(fit.summary())
print("Done!")