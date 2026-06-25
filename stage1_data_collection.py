cat > stage1_data_collection.py << 'EOF'
from fredapi import Fred
import pandas as pd

API_KEY = "f35fa3c10acf6cecdcddb7e65866549a "
fred = Fred(api_key=API_KEY)

series = {
    "sentiment":      "UMCSENT",
    "delinquency":    "DRCCLACBS",
    "unemployment":   "UNRATE",
    "cpi":            "CPIAUCSL",
    "fed_funds_rate": "FEDFUNDS",
}

data = {}
for name, series_id in series.items():
    print(f"Downloading {name} ({series_id})...")
    data[name] = fred.get_series(series_id)

df = pd.DataFrame(data)
df = df.resample("QE").mean()
df = df["1980":"2024"]
df = df.dropna()
df.to_csv("macro_data.csv")
print("\n✅ Done! Data saved to macro_data.csv")
print(df.tail())
EOF