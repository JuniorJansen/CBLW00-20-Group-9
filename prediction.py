import os
import sys
import pandas as pd
import numpy as np
from joblib import load
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import preprocess_data



def prepare_next_month_data(model_df, next_month):
    """
    Prepare data for the next month for prediction.
    Creates a new dataframe with all necessary features for the prediction.
    """
    print(f"Preparing data for {next_month}")

    # Get unique LSOA codes from the historical data
    lsoas = model_df['LSOA code'].unique()
    print(f"Found {len(lsoas)} unique LSOAs")

    # Create next month dataframe with LSOA codes
    next_month_df = pd.DataFrame({'LSOA code': lsoas})

    # Add Month and Year columns
    next_month_df['Month'] = next_month
    next_month_df['Year'] = next_month.year

    # Get the most recent data for each LSOA to use as a base for consistent features
    # Explicitly sort to ensure we get the most recent data
    # Use drop_duplicates to keep only the last occurrence for each LSOA
    latest_data = (
        model_df.sort_values('Month')
        .groupby('LSOA code')
        .last()
        .reset_index()
    )

    # Create lag features based on previous months
    # Latest burglary count becomes Lag1, and previous becomes Lag2

    # Find the two most recent months of data
    recent_months = sorted(model_df['Month'].unique())[-2:]
    print(f"Using data from {recent_months[0]} and {recent_months[1]} for lag features")




    exclude_cols = [
        'LSOA code', 'Month', 'Burglary Count', 'Year', "IMD Score", "PTAL",
               "Male", "Female", "Mean Male Age", "Mean Female Age", "Population", "Lag1", "Lag2"
    ]
    constant_features = [
        col for col in latest_data.columns
        if col not in exclude_cols and col != 'LSOA code'
    ]

    # Select only constant features
    features_df = latest_data[['LSOA code'] + constant_features].copy()

    # Ensure features_df has unique LSOA codes
    features_df = features_df.drop_duplicates(subset=['LSOA code'], keep='first')

    # Merge very carefully, using validate parameter to prevent unexpected duplications
    next_month_df = next_month_df.merge(
        features_df,
        on='LSOA code',
        how='left',
        validate='many_to_one'
    )

    # Verify we have all necessary columns for prediction
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

    print("Predictions complete")
    return next_month_df


def save_and_visualize_predictions(predictions_df, output_dir='predictions'):
    """
    Save predictions to CSV and create visualizations.

    Args:
        predictions_df: DataFrame with predictions
        output_dir: Directory to save outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save predictions to CSV
    output_file = os.path.join(output_dir, 'march_2025_predictions.csv')
    predictions_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    # Create visualizations directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # 1. Distribution of predicted burglary counts
    plt.figure(figsize=(12, 6))
    sns.histplot(predictions_df['Predicted_Count'], bins=30, kde=True)
    plt.title('Distribution of Predicted Burglary Counts - March 2025')
    plt.xlabel('Predicted Burglary Count')
    plt.ylabel('Number of LSOAs')
    plt.savefig(os.path.join(vis_dir, 'predicted_counts_distribution.png'), dpi=300)

    # 2. Distribution of burglary probabilities
    plt.figure(figsize=(12, 6))
    sns.histplot(predictions_df['Burglary_Probability'], bins=30, kde=True)
    plt.title('Distribution of Burglary Probabilities - March 2025')
    plt.xlabel('Probability of at least one Burglary')
    plt.ylabel('Number of LSOAs')
    plt.savefig(os.path.join(vis_dir, 'burglary_probability_distribution.png'), dpi=300)

    # 3. Relationship between predicted count and probability
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Predicted_Count', y='Burglary_Probability', data=predictions_df, alpha=0.6)
    plt.title('Relationship Between Predicted Count and Probability - March 2025')
    plt.xlabel('Predicted Burglary Count')
    plt.ylabel('Probability of at least one Burglary')
    plt.savefig(os.path.join(vis_dir, 'count_vs_probability.png'), dpi=300)

    # 4. Top 20 LSOAs by predicted burglary count
    top_counts = predictions_df.sort_values('Predicted_Count', ascending=False).head(20)
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Predicted_Count', y='LSOA code', data=top_counts)
    plt.title('Top 20 LSOAs by Predicted Burglary Count - March 2025')
    plt.xlabel('Predicted Burglary Count')
    plt.ylabel('LSOA Code')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'top_lsoas_by_count.png'), dpi=300)

    # 5. Top 20 LSOAs by burglary probability
    top_probs = predictions_df.sort_values('Burglary_Probability', ascending=False).head(20)
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Burglary_Probability', y='LSOA code', data=top_probs)
    plt.title('Top 20 LSOAs by Burglary Probability - March 2025')
    plt.xlabel('Probability of at least one Burglary')
    plt.ylabel('LSOA Code')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'top_lsoas_by_probability.png'), dpi=300)

    print(f"Visualizations saved to {vis_dir}")

    # Create a summary of high-risk areas
    high_risk = predictions_df[
        predictions_df['Burglary_Probability'] > 0.8
        ].sort_values('Burglary_Probability', ascending=False)

    # Save high-risk areas to CSV
    high_risk_file = os.path.join(output_dir, 'high_risk_lsoas_march_2025.csv')
    high_risk.to_csv(high_risk_file, index=False)
    print(f"High-risk LSOAs saved to {high_risk_file}")

    return output_file


def main():
    """
    Main function to predict burglaries for March 2025.
    """
    print("Starting burglary prediction for March 2025")

    # Define the next month to predict
    next_month = pd.Timestamp('2025-03-01')
    print(f"Target prediction month: {next_month.strftime('%B %Y')}")

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

    # Load trained models
    reg, clf, selected_features = load_models()

    # Prepare data for March 2025
    next_month_df = prepare_next_month_data(historical_df, next_month)

    # Make predictions
    predictions_df = make_predictions(next_month_df, reg, clf, selected_features)

    # Save and visualize predictions
    output_file = save_and_visualize_predictions(predictions_df)

    # Print summary statistics
    print("\nPrediction Summary for March 2025:")
    print(f"Total LSOAs: {len(predictions_df)}")
    print(f"Average predicted burglary count: {predictions_df['Predicted_Count'].mean():.2f}")
    print(f"Average burglary probability: {predictions_df['Burglary_Probability'].mean():.2f}")
    print(f"Number of high-risk LSOAs (90th percentile): {int(len(predictions_df) * 0.1)}")

    # Print top 5 highest risk LSOAs
    print("\nTop 5 LSOAs by predicted burglary count:")
    top5 = predictions_df.sort_values('Predicted_Count', ascending=False).head(5)
    for _, row in top5.iterrows():
        print(f"LSOA: {row['LSOA code']}, Predicted Count: {row['Predicted_Count']:.2f}, "
              f"Probability: {row['Burglary_Probability']:.2f}")

    print(f"\nDetailed predictions saved to {output_file}")
    print("Prediction completed successfully!")


if __name__ == '__main__':
    main()