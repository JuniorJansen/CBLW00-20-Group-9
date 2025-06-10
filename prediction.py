import os
import sys
import pandas as pd
import numpy as np
from joblib import load
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import preprocess_data


def determine_next_month(model_df):
    latest_month = model_df['Month'].max()
    if latest_month.month == 12:
        next_month = pd.Timestamp(year=latest_month.year + 1, month=1, day=1)
    else:
        next_month = pd.Timestamp(year=latest_month.year, month=latest_month.month + 1, day=1)
    return next_month


def prepare_next_month_data(model_df, next_month):
    lsoas = model_df['LSOA code'].unique()
    next_month_df = pd.DataFrame({'LSOA code': lsoas})
    next_month_df['Month'] = next_month
    next_month_df['Year'] = next_month.year

    # Last observed row per LSOA
    latest_data = (
        model_df
        .sort_values('Month')
        .groupby('LSOA code')
        .last()
        .reset_index()
    )

    # Copy forward constant features (excluding only truly non-feature cols)
    exclude_cols = [
        'LSOA code', 'Month', 'Burglary Count', 'Year', 'IMD Score'
    ]
    constant_features = [c for c in latest_data.columns if c not in exclude_cols]
    features_df = latest_data[['LSOA code'] + constant_features]

    next_month_df = next_month_df.merge(
        features_df,
        on='LSOA code', how='left', validate='many_to_one'
    )

    # Compute reversed features using training maxima
    # (we know these were computed during training)
    # Energy_All_rev
    if 'Energy_All' in model_df.columns and 'Energy_All' in next_month_df.columns:
        ea_max = model_df['Energy_All'].max()
        next_month_df['Energy_All_rev'] = ea_max - next_month_df['Energy_All']
    # Digital Propensity Score_rev
    if 'Digital Propensity Score' in model_df.columns and 'Digital Propensity Score' in next_month_df.columns:
        dp_max = model_df['Digital Propensity Score'].max()
        next_month_df['Digital Propensity Score_rev'] = dp_max - next_month_df['Digital Propensity Score']
    # PTAL_rev
    if 'PTAL' in model_df.columns and 'PTAL' in next_month_df.columns:
        ptal_max = model_df['PTAL'].max()
        next_month_df['PTAL_rev'] = ptal_max - next_month_df['PTAL']

    return next_month_df


def add_moving_averages_per_split(train, test, target_col='Burglary Count', windows=[3, 6]):
    # same as training-time function
    train = train.sort_values(['LSOA code', 'Month']).copy()
    test = test.sort_values(['LSOA code', 'Month']).copy()
    for w in windows:
        col = f'{target_col}_MA{w}'
        train[col] = train.groupby('LSOA code')[target_col].transform(
            lambda x: x.shift(1).ewm(span=w, adjust=False).mean()
        )
        last = train.groupby('LSOA code')[col].last()
        test[col] = test['LSOA code'].map(last)
    # spatial lag if present
    if 'Burglary Count_SpatialLag1' in train.columns:
        last_sl = train.groupby('LSOA code')['Burglary Count_SpatialLag1'].last()
        test['Burglary Count_SpatialLag1'] = test['LSOA code'].map(last_sl)
    return train, test


def load_models(models_dir='models'):
    try:
        reg = load(os.path.join(models_dir, 'burglary_regressor.joblib'))
        clf = load(os.path.join(models_dir, 'burglary_classifier.joblib'))
        features = load(os.path.join(models_dir, 'selected_features.joblib'))
        return reg, clf, features
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)


def make_predictions(next_df, reg, clf, features):
    # Ensure we have all columns
    missing = [c for c in features if c not in next_df.columns]
    if missing:
        raise ValueError(f"Missing features for prediction: {missing}")

    # Impute if necessary
    X = next_df[features].fillna(0)

    next_df['Predicted_Count'] = np.maximum(0, reg.predict(X))
    next_df['Burglary_Probability'] = clf.predict_proba(X)[:, 1]

    # Postcode-level join if available
    try:
        pc = pd.read_csv(
            'boundaries/PCD_OA21_LSOA21_MSOA21_LAD_NOV24_UK_LU.csv',
            usecols=['pcds', 'lsoa21cd'], dtype=str, encoding='latin-1'
        )
        pc.rename(columns={'pcds':'Postcode','lsoa21cd':'LSOA code'}, inplace=True)
        pc = pc.dropna().drop_duplicates()
        by_pc = next_df.merge(pc, on='LSOA code', how='left', validate='one_to_many')
    except:
        by_pc = None
    return next_df, by_pc


def save_and_visualize_predictions(pred_df, pred_pc, next_month, output_dir='data'):
    os.makedirs(output_dir, exist_ok=True)
    month_str = next_month.strftime('%B_%Y').lower()

    # Risk levels
    p = pred_df['Burglary_Probability']
    low, high = p.quantile(0.3), p.quantile(0.9)
    def risk(x):
        if x<=low: return 'Low'
        if x>=high: return 'High'
        return 'Medium'
    pred_df['Risk_Level'] = pred_df['Burglary_Probability'].apply(risk)

    # Save CSVs
    lsoa_file = os.path.join(output_dir, f"{month_str}_predictions_lsoa.csv")
    pred_df.to_csv(lsoa_file, index=False)
    print(f"Saved LSOA-level to {lsoa_file}")
    if pred_pc is not None:
        pc_file = os.path.join(output_dir, f"{month_str}_predictions_postcode.csv")
        pred_pc.to_csv(pc_file, index=False)
        print(f"Saved postcode-level to {pc_file}")


def main():
    hist = preprocess_data(
        'metropolitan_police_data.csv',
        'data/imd2019lsoa.csv',
        'data/population_summary.csv',
        'data/medianenergyefficiencyscoreenglandandwales.xlsx',
        'data/PCD_OA21_LSOA21_MSOA21_LAD_NOV24_UK_LU.csv',
        'data/LSOA2011 AvPTAI2015.csv',
        'boundaries/london_lsoa.geojson',
        'data/digitalpropensityindexlsoas.xlsx'
    )
    next_month = determine_next_month(hist)
    next_df = prepare_next_month_data(hist, next_month)
    _, next_df = add_moving_averages_per_split(hist, next_df)
    reg, clf, feats = load_models()
    pred_lsoa, pred_pc = make_predictions(next_df, reg, clf, feats)
    save_and_visualize_predictions(pred_lsoa, pred_pc, next_month)

if __name__ == '__main__':
    main()