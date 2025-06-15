import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from joblib import dump
from preprocessing import preprocess_data
from prediction import predict_next_month_from_history
weights = {
'b. Income Deprivation Domain': 23.82,
'c. Employment Deprivation Domain': 23.82,
'e. Health Deprivation and Disability Domain': 7.15,
'd. Education, Skills and Training Domain': 7.15,
'g. Barriers to Housing and Services Domain': 5.91,
'h. Living Environment Deprivation Domain': 5.91,
'Digital Propensity Score_rev': 7.15,
'Energy_All_rev': 5.91,
'AvPTAI2015': 5.91
}

def add_moving_averages_per_split(train, test, target_col='Burglary Count', windows=[3, 6]):
    """Add exponential moving averages of the target to train and test splits separately."""
    print("Adding moving average features to training and test sets separately...")

    # Sort by LSOA code and Month
    train = train.sort_values(by=['LSOA code', 'Month']).copy()
    test = test.sort_values(by=['LSOA code', 'Month']).copy()

    for window in windows:
        col_name = f'{target_col}_MA{window}'

        # Compute EMA on training data only (shift to avoid leakage)
        train[col_name] = (
            train.groupby('LSOA code')[target_col]
            .transform(lambda x: x.shift(1).ewm(span=window, adjust=False).mean())
        )

        # Initialize test EMA column
        test[col_name] = np.nan

        # Compute EMA for test using combined train+test with shifted values
        for lsoa in test['LSOA code'].unique():
            train_vals = train.loc[train['LSOA code'] == lsoa, [target_col]]
            test_vals = test.loc[test['LSOA code'] == lsoa, [target_col]]

            if train_vals.empty or test_vals.empty:
                continue

            combined = pd.concat([train_vals, test_vals])
            shifted = combined[target_col].shift(1)
            ema = shifted.ewm(span=window, adjust=False).mean()

            # Assign EMA values for the test indices
            test.loc[test['LSOA code'] == lsoa, col_name] = ema.loc[test_vals.index]

    return train, test


def train_eval_models(df, save_models=True):
    """Train and evaluate RandomForest regression and classification models."""
    print("Performing 80/20 chronological split")

    # Chronological train-test split by Month
    months = np.sort(df['Month'].unique())
    cutoff = months[int(len(months) * 0.8) - 1]
    train = df[df['Month'] <= cutoff].copy()
    test = df[df['Month'] > cutoff].copy()
    print(f"Train rows: {train.shape[0]}, Test rows: {test.shape[0]}")

    # Add moving-average features
    train, test = add_moving_averages_per_split(
        train, test, target_col='Burglary Count', windows=[3, 6]
    )

    # Reverse transformations for specific features
    for subset in (train, test):
        subset['PTAL_rev'] = subset['PTAL'].max() - subset['PTAL']
        subset['Energy_All_rev'] = subset['Energy_All'].max() - subset['Energy_All']
        subset['Digital Propensity Score_rev'] = (
                subset['Digital Propensity Score'].max() - subset['Digital Propensity Score']
        )

    # Ensure required weight features exist and fill missing
    for col in weights:
        if col not in train.columns:
            raise ValueError(f"Missing feature '{col}' needed for sample weight calculation.")
        train[col] = train[col].fillna(0)

    # Compute sample weights
    sw = train[list(weights.keys())].mul(pd.Series(weights), axis=1).sum(axis=1)
    sample_weights = sw / sw.mean()

    # Select features, excluding identifiers and target
    exclude = [
        'LSOA code', 'Month', 'Burglary Count', 'Year', 'IMD Score',
        'Digital Propensity Score', 'Energy_All', 'Mean Age', 'Custom_IMD_Score',
        'Male/Female Ratio', 'AvPTAI2015_rev', 'Digital Propensity Score_rev',
        'Energy_All_rev', 'Burglary Count_SpatailLag1', 'PTAL_rev', 'PTAL',
        'AvPTAI2015', 'GIZ','Crime Domain','Female','Male', 'Crime Domain'
    ]
    features = [c for c in train.columns if c not in exclude]
    print(f"Number of features: {len(features)}")

    X_train = train[features]
    X_test = test[features]
    y_train_r = train['Burglary Count']
    y_test_r = test['Burglary Count']
    y_train_c = (y_train_r > 0).astype(int)
    y_test_c = (y_test_r > 0).astype(int)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=features)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=features)

    # ------- REGRESSION -------
    print("\n--- REGRESSION MODEL (RandomForestRegressor) ---")
    reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        min_samples_split=5,
        max_features='sqrt',
        random_state=42
    )
    reg.fit(X_train, y_train_r, sample_weight=sample_weights)
    preds_r = reg.predict(X_test)

    # Regression metrics
    rmse = np.sqrt(mean_squared_error(y_test_r, preds_r))
    mae = mean_absolute_error(y_test_r, preds_r)
    r2 = r2_score(y_test_r, preds_r)
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")

    if save_models:
        os.makedirs('models', exist_ok=True)
        dump(reg, 'models/burglary_regressor.joblib')
        print("Saved RandomForest regression model to models/burglary_regressor.joblib")

    # ------- CLASSIFICATION -------
    print("\n--- CLASSIFICATION MODEL (RandomForestClassifier) ---")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight={0:1,1:1.2},
        random_state=42
    )
    clf.fit(X_train, y_train_c)

    preds_c = clf.predict(X_test)
    preds_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test_c, preds_c)
    auc = roc_auc_score(y_test_c, preds_proba)
    print(f"Accuracy: {acc:.3f}, ROC AUC: {auc:.3f}")
    print(classification_report(y_test_c, preds_c))

    tn, fp, fn, tp = confusion_matrix(y_test_c, preds_c).ravel()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    if save_models:
        dump(clf, 'models/burglary_classifier.joblib')
        dump(features, 'models/selected_features.joblib')
        print("Saved RandomForest classifier and feature list to models/")

    return (
        reg, clf, features, features,
        X_test, y_test_r, y_test_c,
        preds_r, preds_c, preds_proba
    )


def visualize_results(
        reg, clf, features, selected_features,
        X_test, y_test_r, y_test_c,
        preds_r, preds_c, preds_proba
):
    """Create visualizations for model evaluation and save to disk."""
    print("\nCreating visualizations...")
    os.makedirs('visualizations', exist_ok=True)

    # 1. Feature importances (regression)
    imp_reg = pd.Series(reg.feature_importances_, index=features).sort_values()
    top_imp_reg = imp_reg.tail(16)
    plt.figure(figsize=(12, 8))
    top_imp_reg.plot.barh()
    plt.title('Feature Importances - Regression Model')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance_regression.png', dpi=300)

    # 2. Feature importances (classification)
    imp_clf = pd.Series(clf.feature_importances_, index=selected_features).sort_values()
    top_imp_clf = imp_clf.tail(min(16, len(imp_clf)))
    plt.figure(figsize=(12, 8))
    top_imp_clf.plot.barh()
    plt.title('Feature Importance - Classification Model')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance_classification.png', dpi=300)

    # 3. Residual diagnostics
    resid = y_test_r - preds_r
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.histplot(resid, kde=True, ax=axes[0])
    axes[0].set_title('Residuals Distribution')
    axes[0].axvline(0, color='red', linestyle='--')
    sns.scatterplot(x=preds_r, y=y_test_r, alpha=0.5, ax=axes[1])
    max_val = max(preds_r.max(), y_test_r.max())
    axes[1].plot([0, max_val], [0, max_val], color='red', linestyle='--')
    axes[1].set_title('Predicted vs Actual')
    plt.tight_layout()
    plt.savefig('visualizations/regression_diagnostics.png', dpi=300)

    # 4. ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test_c, preds_proba)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Burglary Classification')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/roc_curve.png', dpi=300)

    # 5. Confusion matrix heatmap
    cm = confusion_matrix(y_test_c, preds_c)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=['No Burglary', 'Burglary'],
        yticklabels=['No Burglary', 'Burglary']
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrix.png', dpi=300)

    print("All visualizations saved to 'visualizations' directory")


def main():
    """Main entry point: preprocess data, train models, and visualize results."""
    print("Starting pipeline...")

    # File paths
    BURGLARY_CSV = 'data/metropolitan_police_data.csv'
    IMD_CSV = 'data/imd2019lsoa.csv'
    POP_CSV = 'data/population_summary.csv'
    ENERGY_XLSX = 'data/medianenergyefficiencyscoreenglandandwales.xlsx'
    MAPPING_CSV = 'data/PCD_OA21_LSOA21_MSOA21_LAD_NOV24_UK_LU.csv'
    PTAL_CSV = 'data/LSOA2011 AvPTAI2015.csv'
    LONDON_LSOA = 'boundaries/london_lsoa_shapefile/london_lsoa.shp'
    DIGITAL_XLSX = 'data/digitalpropensityindexlsoas.xlsx'

    try:
        df = preprocess_data(
            BURGLARY_CSV,
            IMD_CSV,
            POP_CSV,
            ENERGY_XLSX,
            MAPPING_CSV,
            PTAL_CSV,
            LONDON_LSOA,
            DIGITAL_XLSX,
        )
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        sys.exit(1)

    print(df.sample(5).to_string())

    reg, clf, features, selected_features, X_test, y_test_r, y_test_c, preds_r, preds_c, preds_proba = (
        train_eval_models(df, save_models=True)
    )

    print(df.head())

    try:
        visualize_results(
            reg, clf, features, selected_features,
            X_test, y_test_r, y_test_c,
            preds_r, preds_c, preds_proba
        )
    except Exception as e:
        print(f"WARNING: Error during visualization: {e}")
    predict_next_month_from_history(df)
    print("Analysis complete.")


if __name__ == '__main__':
    main()