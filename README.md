# Sentiment-Driven Credit Risk Analysis

## Overview
This project investigates whether consumer sentiment indices can improve the prediction of credit card delinquency rates and expected credit losses, using 44 years of Federal Reserve macroeconomic data (1980–2024).

## Research Question
*Does how people feel predict how they default?*

## Data
All data sourced from the FRED API (Federal Reserve Bank of St. Louis):
- Michigan Consumer Sentiment Index (UMCSENT)
- Credit Card Delinquency Rate (DRCCLACBS)
- Unemployment Rate (UNRATE)
- Consumer Price Index (CPIAUCSL)
- Federal Funds Rate (FEDFUNDS)

## Methodology
- **Exploratory Data Analysis** — time series visualisation across major crises
- **Econometrics** — ADF stationarity tests, VAR model, Granger causality
- **Machine Learning** — logistic regression with and without sentiment features
- **Credit Risk** — Basel III Expected Loss framework (EL = PD × LGD × EAD)

## Key Findings
- Macro variables alone predict high-risk periods with **82.8% accuracy**
- Adding consumer sentiment improves accuracy to **86%** (+3%)
- Highest predicted risk periods: **1991–1992** recession
- Average Expected Loss across 44 years: **22.2%**

## Project Structure
| File | Description |
|------|-------------|
| `stage1_data_collection.py` | FRED API data download |
| `stage2_eda.py` | Exploratory data analysis & visualisation |
| `stage3_econometrics.py` | VAR model & Granger causality tests |
| `stage4_prediction.py` | Logistic regression credit risk model |
| `stage5_expected_loss.py` | Basel III Expected Loss model |

## Requirements
pip install fredapi pandas matplotlib statsmodels scikit-learn

## Author
Syeda Mushfika Ahamed| Economics & Mathematics
