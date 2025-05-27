import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, roc_auc_score, confusion_matrix
)

import seaborn as sns
from preprocessing import preprocess_data
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import geopandas as gpd
from esda.moran import Moran
from libpysal.weights import Queen
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
from sklearn import tree


def train_eval_models(df, save_models=True):
    """Train and evaluate regression and classification models."""
    print("Performing 80/20 chronological split")
    months = df['Month'].sort_values().unique()
    cutoff = months[int(len(months) * 0.8) - 1]
    train = df[df['Month'] <= cutoff].copy()
    test = df[df['Month'] > cutoff].copy()
    print(f"Train rows: {train.shape[0]}, Test rows: {test.shape[0]}")
    # Ensure all necessary columns are present and fill NaNs
# Reverse feature transformations (apply to both train and test)
    for df_split in [train, test]:
       df_split['Energy_All_rev'] = df_split['Energy_All'].max() - df_split['Energy_All']
       df_split['Digital Propensity Score_rev'] = df_split['Digital Propensity Score'].max() - df_split['Digital Propensity Score']


    for col in weights:
       if col not in train.columns:
        raise ValueError(f"Missing feature '{col}' needed for sample weight calculation.")

       train[col] = train[col].fillna(0)


    # Compute weighted sum across selected features
    sample_weights = train[list(weights.keys())].mul(pd.Series(weights), axis=1).sum(axis=1)
    sample_weights = sample_weights / sample_weights.mean()



    # Prepare features and targets
    exclude = ['LSOA code', 'Month', 'Burglary Count', 'Year', 'IMD Score',
               "Male", "Female", "Mean Male Age", "Mean Female Age", "Population",
            #    "b. Income Deprivation Domain", "c. Employment Deprivation Domain", "d. Education, Skills and Training Domain",
            #    "e. Health Deprivation and Disability Domain",
            "f. Crime Domain",
            # "g. Barriers to Housing and Services Domain",
            #    "h. Living Environment Deprivation Domain",
                'Digital Propensity Score',
                'Energy_All',
                #'Mean Age',
            #'Burglary Count_MA3', 'Burglary Count_MA6',
               'Custom_IMD_Score',
               'i. Income Deprivation Affecting Children Index (IDACI)', 'j. Income Deprivation Affecting Older People Index (IDAOPI)', 'Male/Female Ratio',
            #    'AvPTAI2015',
                'PTAL', 'Lag1', 'Lag2',
               'Custom Deprivation Score']
    features = [c for c in train.columns if c not in exclude]

    X_train, X_test = train[features], test[features]
    y_train_r, y_test_r = train['Burglary Count'], test['Burglary Count']
    y_train_c, y_test_c = (y_train_r > 0).astype(int), (y_test_r > 0).astype(int)
    print(f"Number of features: {len(features)}")

    if save_models:
        os.makedirs('models', exist_ok=True)

    # REGRESSION MODEL
    print("\n--- REGRESSION MODEL ---")
    reg = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_leaf=2,
        min_samples_split=5, max_features='sqrt',
        random_state=42
    )
    reg.fit(X_train, y_train_r, sample_weight=sample_weights)

    # Evaluate the regression model
    preds_r = reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test_r, preds_r))
    mae = mean_absolute_error(y_test_r, preds_r)
    r2 = r2_score(y_test_r, preds_r)

    print("\nRegression evaluation:")
    print(f" RMSE: {rmse:.2f}")
    print(f" MAE:  {mae:.2f}")
    print(f" R²:   {r2:.3f}")

    if save_models:
        from joblib import dump
        dump(reg, 'models/burglary_regressor.joblib')
        print("Saved improved regression model to models/burglary_regressor_improved.joblib")

    # CLASSIFICATION MODEL
    print("\n--- CLASSIFICATION MODEL ---")

    # Use all features for classification
    print("Using all features for classification...")
    selected_features = features
    X_train_selected = X_train
    X_test_selected = X_test
    print(f"Using {len(selected_features)} features: {', '.join(selected_features)}")

    # Train RandomForestClassifier
    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=200,  # Increased complexity
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_train_selected, y_train_c)

    # Evaluate classification model
    preds_c = clf.predict(X_test_selected)
    preds_proba = clf.predict_proba(X_test_selected)[:, 1]

    acc = accuracy_score(y_test_c, preds_c)
    auc = roc_auc_score(y_test_c, preds_proba)

    print("\nClassification evaluation:")
    print(f" Accuracy: {acc:.3f}")
    print(f" ROC AUC: {auc:.3f}")
    print(classification_report(y_test_c, preds_c))

    # Print confusion matrix values
    conf_matrix = confusion_matrix(y_test_c, preds_c)
    tn, fp, fn, tp = conf_matrix.ravel()
    print("Confusion Matrix:")
    print(f" True Negatives: {tn}, False Positives: {fp}")
    print(f" False Negatives: {fn}, True Positives: {tp}")

    if save_models:
        from joblib import dump
        dump(clf, 'models/burglary_classifier.joblib')
        dump(selected_features, 'models/selected_features.joblib')
        print("Saved improved classification model and feature list")

    # Return models and evaluation metrics for visualizations
    return reg, clf, features, selected_features, X_test, y_test_r, y_test_c, preds_r, preds_c, preds_proba


def visualize_results(reg, clf, features, selected_features, X_test, y_test_r, y_test_c,
                      preds_r, preds_c, preds_proba):
    """Create improved visualizations for model evaluation."""

    # Create directory for visualizations
    os.makedirs('visualizations', exist_ok=True)

    # 1. Feature importance for regression model
    print("\nCalculating feature importance...")
    imp = pd.Series(reg.feature_importances_, index=features).sort_values(ascending=True)
    top_imp = imp.tail(16)

    plt.figure(figsize=(12, 8))
    ax = top_imp.plot.barh()
    ax.set_title('Feature Importances - Regression Model', fontsize=14)
    ax.set_xlabel('Importance Score', fontsize=12)
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance_regression.png', dpi=300)

    # 2. Classification features importance (all features)
    plt.figure(figsize=(12, 8))
    clf_imp = pd.Series(clf.feature_importances_, index=selected_features).sort_values(ascending=True)
    top_clf_imp = clf_imp.tail(min(16, len(clf_imp)))

    ax = top_clf_imp.plot.barh()
    ax.set_title('Feature Importance - Random Forest Classifier', fontsize=14)
    ax.set_xlabel('Importance Score', fontsize=12)
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance_classification.png', dpi=300)

    # 3. Improved residuals plot
    resid = y_test_r - preds_r
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Histogram of residuals
    sns.histplot(resid, kde=True, ax=axes[0])
    axes[0].set_title('Residuals Distribution', fontsize=14)
    axes[0].set_xlabel('Residual Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].axvline(x=0, color='red', linestyle='--')

    # Predicted vs actual plot
    sns.scatterplot(x=preds_r, y=y_test_r, alpha=0.5, ax=axes[1])

    # Add perfect prediction line
    max_val = max(preds_r.max(), y_test_r.max())
    axes[1].plot([0, max_val], [0, max_val], color='red', linestyle='--')

    axes[1].set_title('Predicted vs Actual', fontsize=14)
    axes[1].set_xlabel('Predicted Burglary Count', fontsize=12)
    axes[1].set_ylabel('Actual Burglary Count', fontsize=12)

    plt.tight_layout()
    plt.savefig('visualizations/regression_diagnostics.png', dpi=300)

    # 4. ROC curve for classification
    from sklearn.metrics import roc_curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test_c, preds_proba)
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve for Burglary Classification', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/roc_curve.png', dpi=300)

    # 5. Confusion matrix as heatmap
    plt.figure(figsize=(8, 6))
    conf_matrix = confusion_matrix(y_test_c, preds_c)
    sns.heatmap(conf_matrix, annot=True, fmt='d',
                xticklabels=['No Burglary', 'Burglary'],
                yticklabels=['No Burglary', 'Burglary'])
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrix.png', dpi=300)

    print("\nAll visualizations saved to 'visualizations' directory")

    tree.plot_tree(clf.estimators_[0], feature_names=features,
                   class_names=['NoBurglary', 'Burglary'], max_depth=4,
                   filled=True, fontsize=8, ax=ax)
    plt.title('Decision Tree (depth=3)');
    plt.show()


def main():
    SAVE_MODELS = True

    print(" Starting main()")
    print("Preprocessing data (IMD, population, energy)")

    BURGLARY_CSV = 'data/metropolitan_police_data.csv'
    IMD_CSV = 'data/imd2019lsoa.csv'
    POP_CSV = 'data/population_summary.csv'
    ENERGY_XLSX = 'data/medianenergyefficiencyscoreenglandandwales.xlsx'
    MAPPING_CSV = 'data/PCD_OA21_LSOA21_MSOA21_LAD_NOV24_UK_LU.csv'
    PTAL_CSV = 'data/LSOA2011 AvPTAI2015.csv'
    LONDON_LSOA = 'data/london_lsoa_codes.csv'
    DIGITAL_XSLX = 'data/digitalpropensityindexlsoas.xlsx'

    try:
        df = preprocess_data(
            BURGLARY_CSV,
            IMD_CSV,
            POP_CSV,
            ENERGY_XLSX,
            MAPPING_CSV,
            PTAL_CSV,
            LONDON_LSOA,
            DIGITAL_XSLX,
        )
    except Exception as e:
        print(f"[ERROR] Preprocessing failed at preprocess_data: {e}")
        sys.exit(1)
    print(df.sample(5).to_string())
    reg, clf, features, selected_features, X_test, y_test_r, y_test_c, preds_r, preds_c, preds_proba = \
        train_eval_models(df, save_models=SAVE_MODELS)
    print(df.head())

    try:
       visualize_results(
           reg, clf, features, selected_features, X_test, y_test_r, y_test_c,
           preds_r, preds_c, preds_proba
       )
    except Exception as e:
       print(f"WARNING: Error during visualization: {e}")

    print("Analysis complete")




if __name__ == '__main__':
    main()
