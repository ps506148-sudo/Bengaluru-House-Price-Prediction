# House Price Prediction System — Bengaluru (Updated)

This project compares a regularized Linear Regression (Ridge), Random Forest, and XGBoost to estimate historical Bengaluru property listing prices (INR lakh).

## Dataset
`Dataset/Bengaluru_House_Data.csv` from the public Kaggle Bengaluru House Price Data dataset: https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data

Original Kaggle data has 13,320 rows. The training script retained 12,572 rows after filtering and deduplication; 249 locality labels were used after rare-location grouping.

## Run on Windows
Open Command Prompt inside the HousePricePrediction1 folder:

```
py -m pip install -r requirements.txt
py -m streamlit run Streamlit_App/app.py
```

## Reproduce model training

```
py Notebook/train_bengaluru_three_models.py
```

This writes `Model/house_price_model_bengaluru.pkl` and `bengaluru_model_comparison.csv`. The trained model is already included and need not be retrained for the demonstration.

## Final evaluation (historical holdout)
- Selected algorithm: XGBoost
- MAE: INR 32.14 lakh
- RMSE: INR 88.76 lakh
- R²: 0.6687
- Median absolute percentage error: 17.47%

The linear baseline is **Ridge** (regularized linear regression). Results are on an unseen random 20% test split from the historical dataset, not present-day prices. The target is `price` in lakh. No accuracy claim is made for October 2026.

The old project resources are preserved in `Backup_Original/` and remain separate from the updated Bengaluru application.
