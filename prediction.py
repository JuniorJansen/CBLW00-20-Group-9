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
    latest_data = model_df.sort_values('Month').groupby('LSOA code').last().reset_index()
    exclude_cols = ['LSOA code', 'Month', 'Burglary Count', 'Year', 'IMD Score',
                    'Male', 'Female', 'Mean Male Age', 'Mean Female Age', 'Population',
                    'Custom_IMD_Score','i. Income Deprivation Affecting Children Index (IDACI)',
                    'j. Income Deprivation Affecting Older People Index (IDAOPI)',
                    'Male/Female Ratio','PTAL', 'Lag1', 'Lag2','Custom Deprivation Score']
    constant_features = [col for col in latest_data.columns if col not in exclude_cols and col != 'LSOA code']
    features_df = latest_data[['LSOA code'] + constant_features].copy()

    features_df = features_df.drop_duplicates(subset=['LSOA code'], keep='first')
    next_month_df = next_month_df.merge(
        features_df, on='LSOA code', how='left', validate='many_to_one'
    )
    if 'Energy_All' in next_month_df.columns:
        next_month_df['Energy_All_rev'] = next_month_df['Energy_All'].max() - next_month_df['Energy_All']
    if 'Digital Propensity Score' in next_month_df.columns:
        next_month_df['Digital Propensity Score_rev'] = (
            next_month_df['Digital Propensity Score'].max() - next_month_df['Digital Propensity Score']
        )
    return next_month_df


def load_models(models_dir='models'):
    try:
        reg = load(os.path.join(models_dir, 'burglary_regressor.joblib'))
        clf = load(os.path.join(models_dir, 'burglary_classifier.joblib'))
        selected_features = load(os.path.join(models_dir, 'selected_features.joblib'))
        return reg, clf, selected_features
    except Exception:
        sys.exit(1)


def make_predictions(next_month_df, reg, clf, selected_features):
    exclude = ['LSOA code', 'Month', 'Burglary Count', 'Year', 'IMD Score',
               'Male', 'Female', 'Mean Male Age', 'Mean Female Age', 'Population',
               'f. Crime Domain','Digital Propensity Score','Energy_All',
               'Custom_IMD_Score','i. Income Deprivation Affecting Children Index (IDACI)',
               'j. Income Deprivation Affecting Older People Index (IDAOPI)',
               'Male/Female Ratio','PTAL', 'Lag1', 'Lag2','Custom Deprivation Score']
    features = [c for c in next_month_df.columns if c not in exclude]
    X_reg = next_month_df[features]
    next_month_df['Predicted_Count'] = reg.predict(X_reg)
    next_month_df['Predicted_Count'] = np.maximum(0, next_month_df['Predicted_Count'])
    X_clf = next_month_df[selected_features]
    next_month_df['Burglary_Probability'] = clf.predict_proba(X_clf)[:, 1]
    try:
        pc_lookup = pd.read_csv(
            'boundaries/PCD_OA21_LSOA21_MSOA21_LAD_MAY24_UK_LU.csv',
            usecols=['pcds', 'lsoa21cd'], dtype=str, encoding='latin-1'
        )
        pc_lookup.rename(columns={'pcds':'Postcode','lsoa21cd':'LSOA code'}, inplace=True)
        pc_lookup = pc_lookup.dropna().drop_duplicates()
        predictions_by_pc = next_month_df.merge(pc_lookup, on='LSOA code', how='left', validate='one_to_many')
        return next_month_df, predictions_by_pc
    except Exception:
        return next_month_df, None


def save_and_visualize_predictions(predictions_df, predictions_by_pc, next_month, output_dir='predictions'):
    os.makedirs(output_dir, exist_ok=True)
    month_str = next_month.strftime('%B_%Y').lower()

    predictions_df['Risk_Level'] = predictions_df['Burglary_Probability'].apply(
        lambda x: 'High' if x > 0.9 else ('Low' if x < 0.25 else 'Medium')
    )

    output_file = os.path.join(output_dir, f'{month_str}_predictions_lsoa.csv')
    predictions_df.to_csv(output_file, index=False)
    print(f"LSOA-level predictions saved to {output_file}")

    if predictions_by_pc is not None:
        pc_output_file = os.path.join(output_dir, f'{month_str}_predictions_postcode.csv')
        predictions_by_pc.to_csv(pc_output_file, index=False)
        print(f"Postcode-level predictions saved to {pc_output_file}")
    else:
        pc_output_file = None



    return output_file, pc_output_file


def main():
    historical_df = preprocess_data(
        'data/metropolitan_police_data.csv',
        'data/imd2019lsoa.csv',
        'data/population_summary.csv',
        'data/medianenergyefficiencyscoreenglandandwales.xlsx',
        'data/PCD_OA21_LSOA21_MSOA21_LAD_MAY24_UK_LU.csv',
        'data/LSOA2011 AvPTAI2015.csv',
        'data/london_lsoa_codes.csv',
        'data/digitalpropensityindexlsoas.xlsx'
    )
    next_month = determine_next_month(historical_df)
    next_df = prepare_next_month_data(historical_df, next_month)
    reg, clf, feats = load_models()
    pred_df, pred_pc = make_predictions(next_df, reg, clf, feats)
    save_and_visualize_predictions(pred_df, pred_pc, next_month)

if __name__ == '__main__':
    main()
