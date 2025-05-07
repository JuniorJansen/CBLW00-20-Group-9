import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report
)
from sklearn import tree
from preprocessing import preprocess_data
def main():
    IMPORTANCE_METHOD = 'permutation'

    print(" Starting main()")
    print("Preprocessing data (IMD, population, energy)")

    BURGLARY_CSV = 'data/metropolitan_police_data.csv'
    IMD_CSV = 'data/imd2019lsoa.csv'
    POP_XLSX = 'data/sapelsoasyoa20192022.xlsx'
    ENERGY_XLSX = 'data/medianenergyefficiencyscoreenglandandwales.xlsx'
    MAPPING_CSV = 'data/PCD_OA21_LSOA21_MSOA21_LAD_NOV24_UK_LU.csv'
    PTAL_CSV = 'data/LSOA2011 AvPTAI2015.csv'
    LONDON_LSOA = 'data/london_lsoa_codes.csv'
    try:
        df = preprocess_data(
            BURGLARY_CSV,
            IMD_CSV,
            POP_XLSX,
            ENERGY_XLSX,
            MAPPING_CSV,
            PTAL_CSV,
            LONDON_LSOA
        )
    except Exception as e:
        print(f"[ERROR] Preprocessing failed at preprocess_data: {e}")
        sys.exit(1)

    pd.set_option('display.max_columns', None)
    print("Full model_df preview:")
    first_rows = (
        df.drop_duplicates(subset='LSOA code')
        .head(5)
    )
    print(first_rows)
    print("Columns in model_df:")
    print(df.columns.tolist())

    # 4. Split data
    print("Performing 70/30 chronological split")
    months = df['Month'].sort_values().unique()
    cutoff = months[int(len(months)*0.7)-1]
    train = df[df['Month'] <= cutoff].copy()
    test = df[df['Month'] > cutoff].copy()
    print(f"Train rows: {train.shape[0]}, Test rows: {test.shape[0]}")

    # 5. Prepare features and targets
    exclude = ['LSOA code','Month','Burglary Count','Year']
    features = [c for c in df.columns if c not in exclude]
    X_train, X_test = train[features], test[features]
    y_train_r, y_test_r = train['Burglary Count'], test['Burglary Count']
    y_train_c, y_test_c = (y_train_r>0).astype(int), (y_test_r>0).astype(int)
    print(f"Number of features: {len(features)}")

    # 6. Train & evaluate regression
    print("Training RandomForestRegressor")
    reg = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        random_state=42, n_jobs=-1, verbose=0
    )
    reg.fit(X_train, y_train_r)
    preds_r = reg.predict(X_test)
    print("Regression evaluation:")
    print(f" RMSE: {np.sqrt(mean_squared_error(y_test_r,preds_r)):.2f}")
    print(f" MAE:  {mean_absolute_error(y_test_r,preds_r):.2f}")
    print(f" R²:   {r2_score(y_test_r,preds_r):.3f}")

    # 7. Train & evaluate classification
    print("Training RandomForestClassifier")
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train_c)
    preds_c = clf.predict(X_test)
    print("Classification evaluation:")
    print(f" Accuracy: {accuracy_score(y_test_c,preds_c):.3f}")
    print(classification_report(y_test_c,preds_c))
    if IMPORTANCE_METHOD == 'gini':
        imp = pd.Series(reg.feature_importances_, index=features).sort_values(ascending=True)
        title = "Gini (MDI) Importances"
    else:
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(reg, X_test, y_test_r, n_repeats=10, random_state=42, n_jobs=-1)
        imp = pd.Series(perm.importances_mean, index=features).sort_values(ascending=True)
        title = "Permutation Importances"

    plt.figure(figsize=(8, 6))
    imp.plot.barh()
    plt.title(title)
    plt.tight_layout()
    plt.show()
    # 8. Diagnostics
    print("Plotting feature importances and diagnostics")
    imp = pd.Series(reg.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(8,6)); imp.plot.barh(); plt.gca().invert_yaxis(); plt.title('Feature Importances');
    plt.tight_layout()
    plt.show()
    resid = y_test_r - preds_r
    plt.figure(figsize=(6,4)); plt.hist(resid, bins=30, edgecolor='black'); plt.title('Residuals'); plt.show()
    fig, ax = plt.subplots(figsize=(15,7))
    tree.plot_tree(clf.estimators_[0], feature_names=features,
                   class_names=['NoBurglary','Burglary'], max_depth=3,
                   filled=True, fontsize=8, ax=ax)
    plt.title('Decision Tree (depth=3)'); plt.show()

if __name__ == '__main__':
    main()

