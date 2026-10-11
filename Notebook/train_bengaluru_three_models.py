"""Train Linear Regression, Random Forest and XGBoost on Bengaluru house listings.
Place this file in HousePricePrediction1/Notebook/ and the CSV in Dataset/.
Run: py train_bengaluru_three_models.py
"""
from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name.lower() == 'notebook' else HERE
DATA = ROOT / 'Dataset' / 'Bengaluru_House_Data.csv'
OUT = ROOT / 'Model' / 'house_price_model_bengaluru.pkl'
FEATURES = ['location', 'area_type', 'bhk', 'sqft', 'bath', 'balcony']
NUMS = ['bhk', 'sqft', 'bath', 'balcony']
CATS = ['location', 'area_type']


def sqft_value(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().replace(',', '')
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)', s)
    if match:
        return (float(match.group(1)) + float(match.group(2))) / 2
    try:
        return float(s)
    except ValueError:
        pass
    # Conversion for uncommon area units; skip unrecognized units.
    units = {'sq. meter': 10.7639, 'sq. metres': 10.7639,
             'sq. yards': 9.0, 'sq. yard': 9.0,
             'acres': 43560.0, 'perch': 272.25,
             'cents': 435.6, 'grounds': 2400.0,
             'guntha': 1089.0}
    for unit, factor in units.items():
        m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*' + re.escape(unit), s, re.IGNORECASE)
        if m:
            return float(m.group(1)) * factor
    return np.nan


def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    required = ['area_type', 'location', 'size', 'total_sqft', 'bath', 'balcony', 'price']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    df['bhk'] = pd.to_numeric(df['size'].astype('string').str.extract(r'^(\d+)')[0], errors='coerce')
    df['sqft'] = df['total_sqft'].map(sqft_value)
    df['bath'] = pd.to_numeric(df['bath'], errors='coerce')
    df['balcony'] = pd.to_numeric(df['balcony'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['location'] = df['location'].astype('string').str.strip().str.lower()
    df['area_type'] = df['area_type'].astype('string').str.strip()
    df = df.dropna(subset=['location', 'area_type', 'bhk', 'sqft', 'price'])
    df = df.loc[df['bhk'].between(1, 12) & df['sqft'].between(250, 15000) & df['price'].between(5, 4000)].copy()
    # Deduplicate before split to avoid exact records being present in train and test.
    df = df.drop_duplicates(subset=['location', 'area_type', 'bhk', 'sqft', 'bath', 'balcony', 'price'])
    # Preserve locality detail where supported, group very rare localities as 'other'.
    counts = df['location'].value_counts()
    df.loc[df['location'].map(counts) < 10, 'location'] = 'other'
    df['location'] = df['location'].astype(str)
    df['area_type'] = df['area_type'].astype(str)
    return df


def make_pipeline(estimator):
    pre = ColumnTransformer([
        ('num', Pipeline([('missing', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), NUMS),
        ('cat', Pipeline([('missing', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CATS),
    ])
    return TransformedTargetRegressor(
        regressor=Pipeline([('preprocess', pre), ('model', estimator)]),
        func=np.log1p, inverse_func=np.expm1, check_inverse=False)


def metrics(y, p):
    p = np.maximum(np.asarray(p), 0)
    y = np.asarray(y)
    return {'MAE_lakh': float(mean_absolute_error(y,p)),
            'RMSE_lakh': float(np.sqrt(mean_squared_error(y,p))),
            'R2': float(r2_score(y,p)),
            'Median_APE_percent': float(np.median(np.abs(y-p) / np.maximum(y,0.01)) * 100)}


def candidates():
    return {
        'Linear Regression': Ridge(alpha=10.0), # regularized linear regression for stable categorical encoding
        'Random Forest': RandomForestRegressor(n_estimators=240, max_depth=18,
                                               min_samples_leaf=2, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=500, learning_rate=0.04, max_depth=5,
                                subsample=0.85, colsample_bytree=0.85, reg_lambda=5,
                                objective='reg:squarederror', tree_method='hist',
                                random_state=42, n_jobs=-1),
    }


def main():
    if not DATA.exists():
        raise FileNotFoundError(f'CSV not found: {DATA}\nPut Bengaluru_House_Data.csv in the Dataset folder.')
    raw = pd.read_csv(DATA)
    df = clean_data(raw)
    print(f'Original rows: {len(raw):,} | Cleaned rows: {len(df):,}')
    print('Unique locations after grouping:', df['location'].nunique())
    X,y = df[FEATURES],df['price']
    xd,xt,yd,yt = train_test_split(X,y,test_size=0.20,random_state=42)
    xr,xv,yr,yv = train_test_split(xd,yd,test_size=0.25,random_state=42)
    print(f'Train: {len(xr)} | Validation: {len(xv)} | Test: {len(xt)}')
    rows=[]
    for name, est in candidates().items():
        print(f'\nTraining {name} ...',flush=True)
        model=make_pipeline(est)
        model.fit(xr,yr)
        sc=metrics(yv,model.predict(xv))
        rows.append({'Model':name,**sc})
        print(f"Validation MAE ₹{sc['MAE_lakh']:.2f} lakh | R² {sc['R2']:.4f} | Median APE {sc['Median_APE_percent']:.2f}%",flush=True)
    result=pd.DataFrame(rows).sort_values('MAE_lakh')
    print('\n=== VALIDATION MODEL COMPARISON ===')
    print(result.to_string(index=False))
    winner=str(result.iloc[0]['Model'])
    print('\nSelected:',winner)
    final=make_pipeline(candidates()[winner]); final.fit(xd,yd)
    sc=metrics(yt,final.predict(xt))
    print('\n=== FINAL UNTOUCHED TEST RESULTS ===')
    print(f"MAE: ₹{sc['MAE_lakh']:.2f} lakh | RMSE: ₹{sc['RMSE_lakh']:.2f} lakh")
    print(f"R²: {sc['R2']:.4f} | Median APE: {sc['Median_APE_percent']:.2f}%")
    OUT.parent.mkdir(parents=True,exist_ok=True)
    artifact={'pipeline':final,'model_name':winner,'features':FEATURES,
              'locations':sorted(df['location'].unique().tolist()),
              'area_types':sorted(df['area_type'].unique().tolist()),
              'metrics':sc,'validation_comparison':rows,
              'source':'Kaggle Bengaluru House Price Data; historic prices in INR lakh'}
    joblib.dump(artifact,OUT)
    result.to_csv(ROOT/'bengaluru_model_comparison.csv',index=False)
    print('\nSaved:',OUT)
    print('Saved:',ROOT/'bengaluru_model_comparison.csv')
    sample=pd.DataFrame([{'location':'whitefield','area_type':'Super built-up  Area',
                          'bhk':2,'sqft':1200,'bath':2,'balcony':1}])
    print(f"Example historic listing estimate (Whitefield, 2 BHK, 1200 sqft): ₹{final.predict(sample)[0]:.2f} lakh")
    print('CAUTION: Historical dataset estimates, NOT present-day Bengaluru valuations.')

if __name__=='__main__':
    main()
