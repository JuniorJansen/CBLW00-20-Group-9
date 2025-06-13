import os
import sys
import pandas as pd
import numpy as np
from joblib import load
from datetime import datetime
import matplotlib.pyplot as plt
from preprocessing import preprocess_data


def determine_next_month(model_df):
    latest_month = model_df['Month'].max()
    if latest_month.month == 12:
        return pd.Timestamp(year=latest_month.year + 1, month=1, day=1)
    return pd.Timestamp(year=latest_month.year, month=latest_month.month + 1, day=1)


def prepare_next_month_data(model_df, next_month):
    lsoas = model_df['LSOA code'].unique()
    next_df = pd.DataFrame({'LSOA code': lsoas})
    next_df['Month'] = next_month
    next_df['Year'] = next_month.year

    latest = (
        model_df.sort_values('Month')
                .groupby('LSOA code').last().reset_index()
    )
    exclude = ['LSOA code', 'Month', 'Burglary Count', 'Year', 'IMD Score']
    const_feats = [c for c in latest.columns if c not in exclude]
    feats_df = latest[['LSOA code'] + const_feats].drop_duplicates('LSOA code')

    next_df = next_df.merge(feats_df, on='LSOA code', how='left', validate='many_to_one')

    for feat in ['Energy_All', 'Digital Propensity Score', 'PTAL']:
        if feat in model_df.columns and feat in next_df.columns:
            max_val = model_df[feat].max()
            next_df[f"{feat}_rev"] = max_val - next_df[feat]

    return next_df


def add_moving_averages_per_split(train_df, test_df, target='Burglary Count', windows=[3,6]):
    train = train_df.sort_values(['LSOA code','Month']).copy()
    test = test_df.sort_values(['LSOA code','Month']).copy()
    for w in windows:
        col = f"{target}_MA{w}"
        train[col] = train.groupby('LSOA code')[target].transform(
            lambda x: x.shift(1).ewm(span=w, adjust=False).mean()
        )
        last_vals = train.groupby('LSOA code')[col].last()
        test[col] = test['LSOA code'].map(last_vals)
    if 'Burglary Count_SpatialLag1' in train.columns:
        last_sl = train.groupby('LSOA code')['Burglary Count_SpatialLag1'].last()
        test['Burglary Count_SpatialLag1'] = test['LSOA code'].map(last_sl)
    return train, test


def load_models(models_dir='models'):
    try:
        reg = load(os.path.join(models_dir, 'burglary_regressor.joblib'))
        clf = load(os.path.join(models_dir, 'burglary_classifier.joblib'))
        feats = load(os.path.join(models_dir, 'selected_features.joblib'))
        return reg, clf, feats
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)


def make_predictions(df, reg, clf, feat_list):
    missing = [c for c in feat_list if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    X = df[feat_list].fillna(0)
    df['Predicted_Count'] = np.maximum(0, reg.predict(X))
    df['Burglary_Probability'] = clf.predict_proba(X)[:,1]

    try:
        pc = pd.read_csv(
            'boundaries/PCD_OA21_LSOA21_MSOA21_LAD_NOV24_UK_LU.csv',
            usecols=['pcds','lsoa21cd'], dtype=str, encoding='latin-1'
        )
        pc.rename(columns={'pcds':'Postcode','lsoa21cd':'LSOA code'}, inplace=True)
        pc = pc.dropna().drop_duplicates()
        by_pc = df.merge(pc, on='LSOA code', how='left', validate='one_to_many')
    except Exception:
        by_pc = None
    return df, by_pc


def save_and_visualize_predictions(pred_df, pred_pc, next_month, output_dir='data'):
    os.makedirs(output_dir, exist_ok=True)
    month_str = next_month.strftime('%B_%Y').lower()

    # Assign risk
    p = pred_df['Burglary_Probability']
    low, high = p.quantile(0.3), p.quantile(0.9)
    pred_df['Risk_Level'] = pred_df['Burglary_Probability'].apply(
        lambda x: 'Low' if x <= low else ('High' if x >= high else 'Medium')
    )

    # Compute custom IMD score
    weights = {
        'b. Income Deprivation Domain': 23.82,
        'c. Employment Deprivation Domain': 23.82,
        'e. Health Deprivation and Disability Domain': 7.15,
        'd. Education, Skills and Training Domain': 7.15,
        'g. Barriers to Housing and Services Domain': 5.91,
        'h. Living Environment Deprivation Domain': 5.91,
        'Digital Propensity Score_rev': 7.15,
        'Energy_All_rev': 5.91,
        'AvPTAI2015': 5.91,
        'Mean Age': 7.15
    }
    valid_weights = {f: w for f, w in weights.items() if f in pred_df.columns}

    # Normalize and scale
    for feat, w in valid_weights.items():
        mn, mx = pred_df[feat].min(), pred_df[feat].max()
        scaled = f"{feat}_scaled"
        if mx > mn:
            pred_df[scaled] = (pred_df[feat] - mn) / (mx - mn)
        else:
            pred_df[scaled] = 0.0

    norm_cols = [f + '_scaled' for f in valid_weights]
    arr = np.array(list(valid_weights.values()))

    # Ensure all scaled columns exist
    for col in norm_cols:
        if col not in pred_df.columns:
            pred_df[col] = 0.0

    # Compute IMD scores
    pred_df['IMD_Custom_Score'] = pred_df[norm_cols].fillna(0).values.dot(arr)
    pred_df['IMD_Custom_Score'] = pred_df['IMD_Custom_Score'].fillna(0)

    pred_df['IMD_Custom_Rank'] = (
        pred_df['IMD_Custom_Score']
               .rank(method='dense', ascending=True)
               .fillna(0)
               .astype(int)
    )

    # Rescale to 1-10 and rank
    min_raw, max_raw = pred_df['IMD_Custom_Score'].min(), pred_df['IMD_Custom_Score'].max()
    if max_raw > min_raw:
        pred_df['IMD_Final_1_to_10'] = 1 + 9 * (pred_df['IMD_Custom_Score'] - min_raw) / (max_raw - min_raw)
    else:
        pred_df['IMD_Final_1_to_10'] = 1.0

    pred_df['IMD_Final_1_to_10'] = pred_df['IMD_Final_1_to_10'].fillna(1)
    pred_df['IMD_Rank_from_Scaled10'] = (
        pred_df['IMD_Final_1_to_10']
               .rank(method='dense', ascending=True)
               .astype(int)
    )

    # Save full dataframe with all IMD columns
    all_cols = pred_df.columns.tolist()
    imd_cols = ['IMD_Custom_Score', 'IMD_Custom_Rank', 'IMD_Final_1_to_10', 'IMD_Rank_from_Scaled10']
    print("Columns written to CSV:", imd_cols)

    lsoa_file = os.path.join(output_dir, f"{month_str}_predictions_lsoa.csv")
    pred_df.to_csv(lsoa_file, columns=all_cols, index=False)
    print(f"LSOA-level predictions saved to {lsoa_file}")

    pc_file = None
    if pred_pc is not None:
        pc_file = os.path.join(output_dir, f"{month_str}_predictions_postcode.csv")
        pred_pc.to_csv(pc_file, index=False)
        print(f"Postcode-level predictions saved to {pc_file}")

    return lsoa_file, pc_file


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
    pred_df, pred_pc = make_predictions(next_df, reg, clf, feats)
    save_and_visualize_predictions(pred_df, pred_pc, next_month)


if __name__ == '__main__':
    main()
