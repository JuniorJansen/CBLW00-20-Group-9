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

    # Take the last available row for each LSOA code in historical data
    latest_data = (
        model_df
        .sort_values('Month')
        .groupby('LSOA code')
        .last()
        .reset_index()
    )

    # Columns to exclude when copying constant features forward
    exclude_cols = [
        'LSOA code', 'Month', 'Burglary Count', 'Year', 'IMD Score',
        'Male', 'Female', 'Mean Male Age', 'Mean Female Age', 'Population',
        'Custom_IMD_Score', 'i. Income Deprivation Affecting Children Index (IDACI)',
        'j. Income Deprivation Affecting Older People Index (IDAOPI)',
        'Male/Female Ratio', 'PTAL', 'Lag1', 'Lag2', 'Custom Deprivation Score'
    ]
    constant_features = [
        col for col in latest_data.columns
        if col not in exclude_cols and col != 'LSOA code'
    ]

    features_df = latest_data[['LSOA code'] + constant_features].copy()
    features_df = features_df.drop_duplicates(subset=['LSOA code'], keep='first')

    next_month_df = next_month_df.merge(
        features_df,
        on='LSOA code',
        how='left',
        validate='many_to_one'
    )

    # Create reversed features if they exist
    if 'Energy_All' in next_month_df.columns:
        next_month_df['Energy_All_rev'] = (
            next_month_df['Energy_All'].max() - next_month_df['Energy_All']
        )

    if 'Digital Propensity Score' in next_month_df.columns:
        next_month_df['Digital Propensity Score_rev'] = (
            next_month_df['Digital Propensity Score'].max() - next_month_df['Digital Propensity Score']
        )

    return next_month_df


def load_models(models_dir='models'):
    """
    Loads the trained regressor, classifier, and list of selected features from disk.
    Exits with code 1 if any model file is missing or fails to load.
    """
    try:
        reg = load(os.path.join(models_dir, 'burglary_regressor.joblib'))
        clf = load(os.path.join(models_dir, 'burglary_classifier.joblib'))
        selected_features = load(os.path.join(models_dir, 'selected_features.joblib'))
        return reg, clf, selected_features
    except Exception:
        sys.exit(1)


def make_predictions(next_month_df, reg, clf, selected_features):
    """
    Given next_month_df (with all features filled in) and the loaded models,
    produce:
      1) next_month_df (LSOA-level) augmented with Predicted_Count and Burglary_Probability
      2) predictions_by_pc (postcode-level), if postcode lookup is available;
         otherwise, None.
    """
    exclude = [
        'LSOA code', 'Month', 'Burglary Count', 'Year', 'IMD Score',
        'Male', 'Female', 'Mean Male Age', 'Mean Female Age', 'Population',
        'f. Crime Domain', 'Digital Propensity Score', 'Energy_All',
        'Custom_IMD_Score', 'i. Income Deprivation Affecting Children Index (IDACI)',
        'j. Income Deprivation Affecting Older People Index (IDAOPI)',
        'Male/Female Ratio', 'PTAL', 'Lag1', 'Lag2', 'Custom Deprivation Score'
    ]
    features = [c for c in next_month_df.columns if c not in exclude]

    # 1) Regression: predict the burglary count
    X_reg = next_month_df[features]
    next_month_df['Predicted_Count'] = reg.predict(X_reg)
    next_month_df['Predicted_Count'] = np.maximum(0, next_month_df['Predicted_Count'])

    # 2) Classification: predict probability that burglary > 0
    X_clf = next_month_df[selected_features]
    next_month_df['Burglary_Probability'] = clf.predict_proba(X_clf)[:, 1]

    # 3) Attempt to join with postcode lookup for postcode-level output
    try:
        pc_lookup = pd.read_csv(
            'boundaries/PCD_OA21_LSOA21_MSOA21_LAD_NOV24_UK_LU.csv',
            usecols=['pcds', 'lsoa21cd'],
            dtype=str,
            encoding='latin-1'
        )
        pc_lookup.rename(
            columns={'pcds': 'Postcode', 'lsoa21cd': 'LSOA code'},
            inplace=True
        )
        pc_lookup = pc_lookup.dropna().drop_duplicates()

        predictions_by_pc = next_month_df.merge(
            pc_lookup,
            on='LSOA code',
            how='left',
            validate='one_to_many'
        )
        return next_month_df, predictions_by_pc

    except Exception:
        return next_month_df, None


def save_and_visualize_predictions(predictions_df, predictions_by_pc, next_month, output_dir='data'):
    """
    1) Computes Risk_Level (Low/Medium/High) from Burglary_Probability
       based on bottom 30% / top 10%.
    2) Builds a custom IMD‐like score by:
         a) Normalizing each weighted feature into [0,1]
         b) Taking a weighted sum
         c) Ranking that raw weighted sum (integer rank)
         d) Rescaling the raw sum into a [1,10] range
         e) (Optionally) Ranking that [1,10] value as well
    3) Saves LSOA‐level CSV and (if available) postcode‐level CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    month_str = next_month.strftime('%B_%Y').lower()

    # ─── Step 1) Compute Risk_Level based on quantiles ───────────────────────────
    # We want:
    #   ‣ "Low"  = bottom 30% of Burglary_Probability
    #   ‣ "High" = top 10% of Burglary_Probability
    #   ‣ "Medium" = everything in between

    # First compute the relevant quantile cut‐points:
    prob = predictions_df['Burglary_Probability']
    low_cut  = prob.quantile(0.30)    # 30th percentile
    high_cut = prob.quantile(0.90)    # 90th percentile

    def assign_risk(p):
        if p <= low_cut:
            return 'Low'
        elif p >= high_cut:
            return 'High'
        else:
            return 'Medium'

    predictions_df['Risk_Level'] = predictions_df['Burglary_Probability'].apply(assign_risk)
    # ───────────────────────────────────────────────────────────────────────────────

    # ─── Step 2) Define weights for the custom IMD‐like score ─────────────────────
    weights = {
        'b. Income Deprivation Domain':     23.82,
        'c. Employment Deprivation Domain': 23.82,
        'e. Health Deprivation and Disability Domain': 7.15,
        'd. Education, Skills and Training Domain':    7.15,
        'g. Barriers to Housing and Services Domain':  5.91,
        'h. Living Environment Deprivation Domain':    5.91,
        'Digital Propensity Score_rev':       7.15,
        'Energy_All_rev':                     5.91,
        'AvPTAI2015':                         5.91,
        'Mean Age':                           7.15
    }

    # 3) Keep only those weights whose column actually exists
    valid_weights = {f: w for f, w in weights.items() if f in predictions_df.columns}

    for feature_column in valid_weights:
        col_min = predictions_df[feature_column].min()
        col_max = predictions_df[feature_column].max()
        scaled_name = feature_column + "_scaled"
        if col_max > col_min:
            predictions_df[scaled_name] = (
                (predictions_df[feature_column] - col_min) / (col_max - col_min)
            )
        else:
            predictions_df[scaled_name] = 0.0

    norm_cols = [f + "_scaled" for f in valid_weights]

    weights_array = np.array([valid_weights[f] for f in valid_weights])

    predictions_df['IMD_Custom_Score'] = predictions_df[norm_cols].values.dot(weights_array)

    predictions_df['IMD_Custom_Rank'] = (
        predictions_df['IMD_Custom_Score']
          .rank(method='dense', ascending=False)
          .astype(int)
    )

    raw_min = predictions_df['IMD_Custom_Score'].min()
    raw_max = predictions_df['IMD_Custom_Score'].max()
    if raw_max > raw_min:
        predictions_df['IMD_Final_1_to_10'] = (
            1 + 9 * (predictions_df['IMD_Custom_Score'] - raw_min) / (raw_max - raw_min)
        )
    else:
        predictions_df['IMD_Final_1_to_10'] = 1.0

    predictions_df['IMD_Rank_from_Scaled10'] = (
        predictions_df['IMD_Final_1_to_10']
          .rank(method='dense', ascending=False)
          .astype(int)
    )

    output_file = os.path.join(output_dir, f"{month_str}_predictions_lsoa.csv")
    predictions_df.to_csv(output_file, index=False)
    print(f"LSOA-level predictions saved to {output_file}")

    if predictions_by_pc is not None:
        pc_output_file = os.path.join(output_dir, f"{month_str}_predictions_postcode.csv")
        predictions_by_pc.to_csv(pc_output_file, index=False)
        print(f"Postcode-level predictions saved to {pc_output_file}")
    else:
        pc_output_file = None

    return output_file, pc_output_file



def main():
    # 1) Preprocess historical data into a single DataFrame
    historical_df = preprocess_data(
        'data/metropolitan_police_data.csv',
        'data/imd2019lsoa.csv',
        'data/population_summary.csv',
        'data/medianenergyefficiencyscoreenglandandwales.xlsx',
        'data/PCD_OA21_LSOA21_MSOA21_LAD_NOV24_UK_LU.csv',
        'data/LSOA2011 AvPTAI2015.csv',
        'data/london_lsoa_codes.csv',
        'data/digitalpropensityindexlsoas.xlsx'
    )

    # 2) Determine the next month to predict
    next_month = determine_next_month(historical_df)

    # 3) Create a DataFrame of features for next_month
    next_df = prepare_next_month_data(historical_df, next_month)

    # 4) Load the trained models (regressor, classifier, and selected features)
    reg, clf, feats = load_models()

    # 5) Make LSOA-level and postcode-level predictions
    pred_df, pred_pc = make_predictions(next_df, reg, clf, feats)

    # 6) Compute IMD scores, ranks, and save results
    save_and_visualize_predictions(pred_df, pred_pc, next_month)


if __name__ == '__main__':
    main()
