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
    """
    Determine the next month to predict based on the latest data available.

    Args:
        model_df: Historical data DataFrame

    Returns:
        pd.Timestamp: The next month to predict
    """
    # Get the latest month in the data
    latest_month = model_df['Month'].max()
    print(f"Latest month in data: {latest_month.strftime('%B %Y')}")

    # Calculate next month
    if latest_month.month == 12:
        next_month = pd.Timestamp(year=latest_month.year + 1, month=1, day=1)
    else:
        next_month = pd.Timestamp(year=latest_month.year, month=latest_month.month + 1, day=1)

    print(f"Next month to predict: {next_month.strftime('%B %Y')}")
    return next_month


def prepare_next_month_data(model_df, next_month):
    """
    Prepare data for the next month for prediction.
    Creates a new dataframe with all necessary features for the prediction.
    """
    print(f"Preparing data for {next_month.strftime('%B %Y')}")

    # Get unique LSOA codes from the historical data
    lsoas = model_df['LSOA code'].unique()
    print(f"Found {len(lsoas)} unique LSOAs")

    # Create next month dataframe with LSOA codes
    next_month_df = pd.DataFrame({'LSOA code': lsoas})

    # Add Month and Year columns
    next_month_df['Month'] = next_month
    next_month_df['Year'] = next_month.year

    latest_data = (
        model_df.sort_values('Month')
        .groupby('LSOA code')
        .last()
        .reset_index()
    )

    # Find the two most recent months of data
    recent_months = sorted(model_df['Month'].unique())[-2:]
    print(
        f"Using data from {recent_months[0].strftime('%B %Y')} and {recent_months[1].strftime('%B %Y')} for lag features")

    exclude_cols = [
        'LSOA code', 'Month', 'Burglary Count', 'Year', "IMD Score", "PTAL",
        "Male", "Female", "Mean Male Age", "Mean Female Age", "Population", "Lag1", "Lag2"
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

    print(f"Prepared dataframe with {next_month_df.shape[1]} features for {next_month_df.shape[0]} LSOAs")

    return next_month_df


def load_models(models_dir='models'):
    """
    Load the saved models and feature lists.

    Returns:
        tuple: (regressor, classifier, selected_features)
    """
    print("Loading trained models...")

    try:
        reg = load(os.path.join(models_dir, 'burglary_regressor_improved1.joblib'))
        clf = load(os.path.join(models_dir, 'burglary_classifier_improved1.joblib'))
        selected_features = load(os.path.join(models_dir, 'selected_features.joblib'))

        print("Models loaded successfully")
        return reg, clf, selected_features
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)


def make_predictions(next_month_df, reg, clf, selected_features):
    """
    Make predictions using both regression and classification models.

    Args:
        next_month_df: DataFrame for the next month
        reg: Regression model
        clf: Classification model
        selected_features: Features used by the classification model

    Returns:
        DataFrame with predictions
    """
    print("Making predictions...")

    # Prepare features for regression (excluding non-feature columns)
    exclude = ['LSOA code', 'Month', 'Burglary Count', 'Year', "IMD Score", "PTAL",
               "Male", "Female", "Mean Male Age", "Mean Female Age", "Population", "Lag1", "Lag2"]
    features = [c for c in next_month_df.columns if c not in exclude]

    # Generate regression predictions
    X_reg = next_month_df[features]
    next_month_df['Predicted_Count'] = reg.predict(X_reg)
    next_month_df['Predicted_Count'] = np.maximum(0, next_month_df['Predicted_Count'])  # Ensure non-negative

    # Prepare features for classification (only selected features)
    X_clf = next_month_df[selected_features]

    # Generate classification predictions (probability of having at least one burglary)
    next_month_df['Burglary_Probability'] = clf.predict_proba(X_clf)[:, 1]

    # Try to add postcode mapping if the file exists
    try:
        pc_lookup = pd.read_csv(
            "boundaries/PCD_OA21_LSOA21_MSOA21_LAD_MAY24_UK_LU.csv",
            usecols=['pcds', 'lsoa21cd'],
            dtype=str,
            encoding='latin-1'  # Try latin-1 encoding instead of utf-8
        )
        # rename to match your predictions_df key
        pc_lookup.rename(
            columns={'pcds': 'Postcode',
                     'lsoa21cd': 'LSOA code'},
            inplace=True
        )

        # Remove any duplicates and null values
        pc_lookup = pc_lookup.dropna().drop_duplicates()
        print(f"Loaded {len(pc_lookup)} unique postcode-LSOA mappings")

        predictions_by_pc = next_month_df.merge(
            pc_lookup,
            on='LSOA code',
            how='left',
            validate='one_to_many'
        )

        print(f"Expanded from {len(next_month_df)} LSOAs to {len(predictions_by_pc)} postcodes")

        print("Predictions complete")
        return next_month_df, predictions_by_pc

    except FileNotFoundError:
        print("Postcode lookup file not found - continuing with LSOA-level predictions only")
        print("Predictions complete")
        return next_month_df, None
    except UnicodeDecodeError:
        print("Encoding issue with postcode lookup file - trying alternative encoding...")
        try:
            pc_lookup = pd.read_csv(
                "boundaries/PCD_OA21_LSOA21_MSOA21_LAD_MAY24_UK_LU.csv",
                usecols=['pcds', 'lsoa21cd'],
                dtype=str,
                encoding='cp1252'  # Try cp1252 encoding
            )
            pc_lookup.rename(
                columns={'pcds': 'Postcode',
                         'lsoa21cd': 'LSOA code'},
                inplace=True
            )

            # Remove any duplicates and null values
            pc_lookup = pc_lookup.dropna().drop_duplicates()
            print(f"Loaded {len(pc_lookup)} unique postcode-LSOA mappings")

            predictions_by_pc = next_month_df.merge(
                pc_lookup,
                on='LSOA code',
                how='left',
                validate='one_to_many'
            )

            print(f"Expanded from {len(next_month_df)} LSOAs to {len(predictions_by_pc)} postcodes")
            print("Predictions complete")
            return next_month_df, predictions_by_pc

        except Exception as e:
            print(f"Could not read postcode lookup file: {e}")
            print("Continuing with LSOA-level predictions only")
            print("Predictions complete")
            return next_month_df, None


def save_and_visualize_predictions(predictions_df, predictions_by_pc, next_month, output_dir='predictions'):
    """
    Save predictions to CSV and create visualizations.

    Args:
        predictions_df: DataFrame with LSOA-level predictions
        predictions_by_pc: DataFrame with postcode-level predictions (or None)
        next_month: The month being predicted
        output_dir: Directory to save outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create dynamic filename based on the predicted month
    month_str = next_month.strftime('%B_%Y').lower()

    # Save LSOA-level predictions
    output_file = os.path.join(output_dir, f'{month_str}_predictions_lsoa.csv')
    predictions_df.to_csv(output_file, index=False)
    print(f"LSOA-level predictions saved to {output_file}")

    # Save postcode-level predictions if available
    if predictions_by_pc is not None:
        pc_output_file = os.path.join(output_dir, f'{month_str}_predictions_postcode.csv')
        predictions_by_pc.to_csv(pc_output_file, index=False)
        print(f"Postcode-level predictions saved to {pc_output_file}")
    else:
        pc_output_file = None

    # Create visualizations directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Dynamic title for visualizations
    month_title = next_month.strftime('%B %Y')

    # 1. Distribution of predicted burglary counts
    plt.figure(figsize=(12, 6))
    sns.histplot(predictions_df['Predicted_Count'], bins=30, kde=True)
    plt.title(f'Distribution of Predicted Burglary Counts - {month_title}')
    plt.xlabel('Predicted Burglary Count')
    plt.ylabel('Number of LSOAs')
    plt.savefig(os.path.join(vis_dir, f'predicted_counts_distribution_{month_str}.png'), dpi=300)

    # 2. Distribution of burglary probabilities
    plt.figure(figsize=(12, 6))
    sns.histplot(predictions_df['Burglary_Probability'], bins=30, kde=True)
    plt.title(f'Distribution of Burglary Probabilities - {month_title}')
    plt.xlabel('Probability of at least one Burglary')
    plt.ylabel('Number of LSOAs')
    plt.savefig(os.path.join(vis_dir, f'burglary_probability_distribution_{month_str}.png'), dpi=300)

    # 3. Relationship between predicted count and probability
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Predicted_Count', y='Burglary_Probability', data=predictions_df, alpha=0.6)
    plt.title(f'Relationship Between Predicted Count and Probability - {month_title}')
    plt.xlabel('Predicted Burglary Count')
    plt.ylabel('Probability of at least one Burglary')
    plt.savefig(os.path.join(vis_dir, f'count_vs_probability_{month_str}.png'), dpi=300)

    # 4. Top 20 LSOAs by predicted burglary count
    top_counts = predictions_df.sort_values('Predicted_Count', ascending=False).head(20)
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Predicted_Count', y='LSOA code', data=top_counts)
    plt.title(f'Top 20 LSOAs by Predicted Burglary Count - {month_title}')
    plt.xlabel('Predicted Burglary Count')
    plt.ylabel('LSOA Code')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'top_lsoas_by_count_{month_str}.png'), dpi=300)

    # 5. Top 20 LSOAs by burglary probability
    top_probs = predictions_df.sort_values('Burglary_Probability', ascending=False).head(20)
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Burglary_Probability', y='LSOA code', data=top_probs)
    plt.title(f'Top 20 LSOAs by Burglary Probability - {month_title}')
    plt.xlabel('Probability of at least one Burglary')
    plt.ylabel('LSOA Code')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'top_lsoas_by_probability_{month_str}.png'), dpi=300)

    print(f"Visualizations saved to {vis_dir}")

    # Create a summary of high-risk areas
    high_risk = predictions_df[
        predictions_df['Burglary_Probability'] > 0.8
        ].sort_values('Burglary_Probability', ascending=False)

    # Save high-risk areas to CSV
    high_risk_file = os.path.join(output_dir, f'high_risk_lsoas_{month_str}.csv')
    high_risk.to_csv(high_risk_file, index=False)
    print(f"High-risk LSOAs saved to {high_risk_file}")

    return output_file, pc_output_file


def main():
    """
    Main function to predict burglaries for the next available month.
    """
    print("Starting dynamic burglary prediction")

    # First, get the preprocessed data
    print("Loading and preprocessing historical data...")

    # File paths - same as in train_eval.py
    BURGLARY_CSV = 'data/metropolitan_police_data.csv'
    IMD_CSV = 'data/imd2019lsoa.csv'
    POP_CSV = 'data/population_summary.csv'
    ENERGY_XLSX = 'data/medianenergyefficiencyscoreenglandandwales.xlsx'
    MAPPING_CSV = 'data/PCD_OA21_LSOA21_MSOA21_LAD_NOV24_UK_LU.csv'
    PTAL_CSV = 'data/LSOA2011 AvPTAI2015.csv'
    LONDON_LSOA = 'data/london_lsoa_codes.csv'
    DIGITAL_XSLX = 'data/digitalpropensityindexlsoas.xlsx'

    try:
        # Preprocess data using the existing pipeline
        historical_df = preprocess_data(
            BURGLARY_CSV,
            IMD_CSV,
            POP_CSV,
            ENERGY_XLSX,
            MAPPING_CSV,
            PTAL_CSV,
            LONDON_LSOA,
            DIGITAL_XSLX
        )
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        sys.exit(1)

    # Dynamically determine the next month to predict
    next_month = determine_next_month(historical_df)

    # Load trained models
    reg, clf, selected_features = load_models()

    # Prepare data for the next month
    next_month_df = prepare_next_month_data(historical_df, next_month)

    # Make predictions
    predictions_df, predictions_by_pc = make_predictions(next_month_df, reg, clf, selected_features)

    # Save and visualize predictions
    output_file, pc_output_file = save_and_visualize_predictions(predictions_df, predictions_by_pc, next_month)

    # Print summary statistics
    month_title = next_month.strftime('%B %Y')
    print(f"\nPrediction Summary for {month_title}:")
    print(f"Total LSOAs: {len(predictions_df)}")
    print(f"Average predicted burglary count: {predictions_df['Predicted_Count'].mean():.2f}")
    print(f"Average burglary probability: {predictions_df['Burglary_Probability'].mean():.2f}")

    print("\nTop 5 LSOAs by predicted burglary count:")
    top5 = predictions_df.sort_values('Predicted_Count', ascending=False).head(5)
    for _, row in top5.iterrows():
        print(f"LSOA: {row['LSOA code']}, Predicted Count: {row['Predicted_Count']:.2f}, "
              f"Probability: {row['Burglary_Probability']:.2f}")

    print(f"\nDetailed LSOA predictions saved to {output_file}")
    if pc_output_file:
        print(f"Detailed postcode predictions saved to {pc_output_file}")
    print("Prediction completed successfully!")


if __name__ == '__main__':
    main()