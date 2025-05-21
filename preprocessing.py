import pandas as pd
import numpy as np
import re

weights = {
    'b. Income Deprivation Domain': 22.5,
    'c. Employment Deprivation Domain': 22.5,
    'e. Health Deprivation and Disability Domain': 6.75,
    'd. Education, Skills and Training Domain': 6.75,
    'f. Crime Domain': 5.58,
    'g. Barriers to Housing and Services Domain': 5.58,
    'h. Living Environment Deprivation Domain': 5.58,
    'Digital Propensity Score': 6.75,
    'Energy_All': 5.58,
    'PTAL': 5.58,
    'Mean Age': 6.75
}


def load_london_lsoas(path: str) -> pd.Series:
    """
    Load London LSOA codes from CSV with column 'lsoa_code'.
    Returns a Series of LSOA codes.
    """
    df = pd.read_csv(path, usecols=['lsoa_code'], encoding='utf-8')
    return df['lsoa_code']


def load_burglary_data(path: str) -> pd.DataFrame:
    """
    Load burglary data, parse dates, and filter to burglary incidents.
    """
    try:
        df = pd.read_csv(path, parse_dates=['Month'], encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, parse_dates=['Month'], encoding='latin-1')
    if 'Crime type' in df.columns:
        df = df[df['Crime type'] == 'Burglary']
    return df


def build_full_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a complete grid of every LSOA × Month combination.
    """
    print("Building grid")
    lsoas = df['LSOA code'].unique()
    months = df['Month'].dt.to_period('M').unique().to_timestamp()
    grid = pd.MultiIndex.from_product([lsoas, months], names=['LSOA code', 'Month'])
    return grid.to_frame(index=False)


def aggregate_burglary_counts(grid: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Count burglaries per LSOA per month and merge onto the full grid.
    """
    observed = (
        df.assign(Month=df['Month'].dt.to_period('M').dt.to_timestamp())
        .groupby(['LSOA code', 'Month'])
        .size()
        .reset_index(name='Burglary Count')
    )
    base = grid.merge(observed, on=['LSOA code', 'Month'], how='left')
    base['Burglary Count'] = base['Burglary Count'].fillna(0)
    return base


def load_imd_scores(path: str) -> pd.DataFrame:
    """
    Load IMD domain scores, pivot to wide, and rename for merging.
    """
    print("Adding IMD")
    imd = pd.read_csv(
        path,
        usecols=['FeatureCode', 'Indices of Deprivation', 'Measurement', 'Value'],
        encoding='utf-8',
        low_memory=False
    )
    scores = imd[imd['Measurement'] == 'Score']
    wide = scores.pivot(
        index='FeatureCode',
        columns='Indices of Deprivation',
        values='Value'
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={
            'FeatureCode': 'LSOA code',
            'a. Index of Multiple Deprivation (IMD)': 'IMD Score'
        }
    )
    return wide


def load_population_estimates(path: str) -> pd.DataFrame:
    """
    Load LSOA population estimates from CSV with demographic data.
    Returns ['LSOA code','Year','Population','Female','Male','Mean Age'] and other demographics.

    The CSV is expected to already contain expanded years data from 2010-2025.
    """
    print("Loading population estimates")

    # Read the CSV with demographic data
    df = pd.read_csv(path, encoding='utf-8', low_memory=False)

    # Make column names consistent by removing any whitespace
    df.columns = [col.strip() for col in df.columns]

    # Rename columns to match expected format in the pipeline
    column_mapping = {
        'LSOA Code': 'LSOA code',
        'Total Population': 'Population',
        'Total Female': 'Female',
        'Total Male': 'Male',
        'Mean Age': 'Mean Age'
    }

    # Apply column renaming where columns exist
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})

    # Ensure numeric types for demographic columns
    numeric_cols = ['Population', 'Female', 'Male', 'Mean Age',
                    'Mean Female Age', 'Mean Male Age', 'Male/Female Ratio']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure we have required columns for merging in the pipeline
    required_cols = ['LSOA code', 'Year', 'Population']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"Warning: Missing required columns in population data: {missing_cols}")

        # If we're missing the Population column but have Male and Female, compute it
        if 'Population' in missing_cols and 'Female' in df.columns and 'Male' in df.columns:
            df['Population'] = df['Female'] + df['Male']
            missing_cols.remove('Population')

    # Select relevant columns
    output_cols = ['LSOA code', 'Year', 'Population']

    # Add demographic columns if they exist
    additional_cols = ['Female', 'Male', 'Mean Age', 'Mean Female Age', 'Mean Male Age', 'Male/Female Ratio']
    for col in additional_cols:
        if col in df.columns:
            output_cols.append(col)

    # Make sure all requested columns exist in the dataframe
    final_cols = [col for col in output_cols if col in df.columns]

    print(f"Loaded population data: {len(df)} rows covering years {df['Year'].min()}-{df['Year'].max()}")

    return df[final_cols]


def load_lsoa_msoa_mapping(path: str) -> pd.DataFrame:
    """
    Load LSOA->MSOA mapping from CSV columns 'lsoa21cd' & 'msoa21cd'.
    """
    df = pd.read_csv(path, encoding='latin-1', low_memory=False)
    required = {'lsoa21cd', 'msoa21cd'}
    if not required.issubset(df.columns):
        raise ValueError(f"Mapping CSV missing columns: {required - set(df.columns)}")
    mapping = (
        df[['lsoa21cd', 'msoa21cd']]
        .rename(columns={'lsoa21cd': 'LSOA code', 'msoa21cd': 'MSOA code'})
        .drop_duplicates()
    )
    return mapping


def load_ptal(path: str) -> pd.DataFrame:
    print("Adding Public Transport")
    df = pd.read_csv(path)
    df = df.rename(columns={'LSOA2011': 'LSOA code'})
    df['AvPTAI2015'] = pd.to_numeric(df.get('AvPTAI2015', None), errors='coerce')

    def parse_ptal(x):
        try:
            return float(x)
        except:
            s = str(x).strip().lower()
            if s.endswith('b') and s[:-1].isdigit():
                return float(s[:-1]) + 0.5
            return np.nan

    if 'PTAL' in df.columns:
        df['PTAL'] = df['PTAL'].apply(parse_ptal)
    else:
        df['PTAL'] = np.nan

    return df[['LSOA code', 'AvPTAI2015', 'PTAL']]


def load_online_share(path: str) -> pd.DataFrame:
    """
    Load Digital Propensity Score with extremely robust error handling.
    """
    print("Adding digital propensity")
    try:
        # Read the Excel file
        xls = pd.ExcelFile(path, engine='openpyxl')

        # Always use 'Online_share_LSOA' sheet
        sheet_name = 'Online_share_LSOA'

        # Read the entire sheet, allowing for complex header
        df = pd.read_excel(
            path,
            sheet_name=sheet_name,
            header=None,  # Read all rows as data
            engine='openpyxl'
        )

        # Print initial debug information
        print(f"Total rows in original dataframe: {len(df)}")
        print("Original dataframe columns:", list(df.columns))
        lsoa_row_index = None
        for i, row in df.iterrows():
            if any('LSOA code' in str(val) for val in row):
                lsoa_row_index = i
                break
        if lsoa_row_index is None:
            raise ValueError("Could not find LSOA code row")
        df = pd.read_excel(
            path,
            sheet_name=sheet_name,
            header=lsoa_row_index,
            engine='openpyxl'
        )

        df.columns = [str(col).strip() for col in df.columns]

        # Find LSOA code and Digital Propensity Score columns
        lsoa_cols = [col for col in df.columns if 'LSOA' in col or 'E0' in col]
        score_cols = [col for col in df.columns if 'Digital' in col or 'Propensity' in col or 'Score' in col]

        if not lsoa_cols or not score_cols:
            print("Available columns:", df.columns)
            raise ValueError("Could not find LSOA code or Digital Propensity Score columns")

        lsoa_col = lsoa_cols[0]
        score_col = score_cols[0]

        result_df = pd.DataFrame({
            'LSOA code': df[lsoa_col].astype(str).str.strip(),
            'Digital Propensity Score': pd.to_numeric(df[score_col], errors='coerce')
        })
        result_df = result_df.dropna(subset=['LSOA code', 'Digital Propensity Score'])
        result_df = result_df[
            ~result_df['LSOA code'].isin(['LSOA code', 'Not applicable']) &
            ~result_df['Digital Propensity Score'].isna()
            ]
        result_df = result_df.drop_duplicates(subset='LSOA code')

        print(f"Processed {len(result_df)} rows of digital propensity data")
        return result_df

    except Exception as e:
        print(f"Error loading digital propensity data: {e}")
        return pd.DataFrame(columns=['LSOA code', 'Digital Propensity Score'])

def load_energy_all_dwellings(path: str) -> pd.DataFrame:
    print("Adding energy all dwellings")
    xls = pd.ExcelFile(path)
    frames = []

    for sheet in xls.sheet_names:
        if re.match(r'^[1-4]a$', sheet, flags=re.IGNORECASE):
            df_head = pd.read_excel(path, sheet_name=sheet, header=3, nrows=0)
            cols = df_head.columns.tolist()

            # MSOA code column
            msoa_cols = [c for c in cols if 'msoa' in c.lower() and 'code' in c.lower()]
            if not msoa_cols:
                continue
            code_col = msoa_cols[0]

            # All dwellings column
            dwell_cols = [c for c in cols if 'all dwellings' in c.lower()]
            if not dwell_cols:
                continue
            dwell_col = dwell_cols[0]

            temp = pd.read_excel(path, sheet_name=sheet, header=3, usecols=[code_col, dwell_col])
            temp = temp.rename(columns={code_col: 'MSOA code', dwell_col: f'ED_{sheet}'})
            frames.append(temp)

    if not frames:
        raise ValueError('No energy sheets 1a–4a with MSOA code & All dwellings found')

    energy = frames[0]
    for df_e in frames[1:]:
        energy = energy.merge(df_e, on='MSOA code', how='outer')

    # Average per-sheet columns
    ed_cols = [c for c in energy.columns if c.startswith('ED_')]
    for c in ed_cols:
        energy[c] = pd.to_numeric(energy[c], errors='coerce')
    energy['Energy_All'] = energy[ed_cols].mean(axis=1)

    # Keep only MSOA code + aggregated feature
    energy = energy[['MSOA code', 'Energy_All']]
    energy['Energy_All'] = energy['Energy_All'].fillna(energy['Energy_All'].median())

    return energy


def compute_lag_features(df: pd.DataFrame, lags=(1, 2)) -> pd.DataFrame:
    print("Computing lags")
    df = df.sort_values(['LSOA code', 'Month'])
    for lag in lags:
        df[f'Lag{lag}'] = df.groupby('LSOA code')['Burglary Count'].shift(lag).fillna(0)
    return df

def compute_custom_deprivation_score(df: pd.DataFrame) -> pd.DataFrame:
    print("Computing custom deprivation score")

    # Define mapping: column -> weight
    weights = {
        'b. Income Deprivation Domain': 22.5,
        'c. Employment Deprivation Domain': 22.5,
        'e. Health Deprivation and Disability Domain': 6.75,
        'd. Education, Skills and Training Domain': 6.75,
        'f. Crime Domain': 5.58,
        'g. Barriers to Housing and Services Domain': 5.58,
        'h. Living Environment Deprivation Domain': 5.58,
        'Digital Propensity Score': 6.75,
        'Energy_All': 5.58,
        'PTAL': 5.58,
        'Mean Age': 6.75
    }

    reverse_features = {'Digital Propensity Score', 'Energy_All', 'PTAL'}

    total_weight = sum(weights.values())
    normalized_features = pd.DataFrame()

    for feature, weight in weights.items():
        if feature not in df.columns:
            raise ValueError(f"Missing required feature: {feature}")
        
        col = pd.to_numeric(df[feature], errors='coerce').fillna(df[feature].median())
        col_norm = (col - col.min()) / (col.max() - col.min())
        
        # Reverse if higher value = less deprived
        if feature in reverse_features:
            col_norm = 1 - col_norm

        normalized_features[feature] = col_norm * (weight / total_weight)

    df['Custom Deprivation Score'] = normalized_features.sum(axis=1)
    return df
def add_moving_averages(df, target_col='Burglary Count', windows=[3, 6]):
    """
    Adds moving averages over time for a given target column.
    """
    print("Adding moving average features...")
    
    df = df.sort_values(by=['LSOA code', 'Month']).copy()

    for window in windows:
        col_name = f'{target_col}_MA{window}'
        df[col_name] = df.groupby('LSOA code')[target_col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    
    return df

def compute_custom_score(df, weights):
    weighted_score = np.zeros(len(df))
    for col, w in weights.items():
        if col in df.columns:
            weighted_score += df[col] * w
    return weighted_score / sum(weights.values())


def preprocess_data(
        burglary_path: str,
        imd_path: str,
        pop_path: str,
        energy_path: str,
        mapping_path: str,
        ptal_path: str,
        london_lsoa_path: str,
        digital_propensity_path: str
) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      - filter to London LSOAs
      - burglary counts to grid
      - merge IMD, population, PTAL, energy
      - compute lag features
    """
    # Load London LSOAs and filter burglaries
    london_lsoas = load_london_lsoas(london_lsoa_path)
    df = load_burglary_data(burglary_path)
    # put code here :)
    df = df[df['LSOA code'].isin(london_lsoas)]

    # Build the full LSOA×Month grid
    grid = build_full_grid(df)
    model_df = aggregate_burglary_counts(grid, df)

    # IMD
    imd = load_imd_scores(imd_path)
    model_df = model_df.merge(imd, on='LSOA code', how='left')

    # Population
    pop_df = load_population_estimates(pop_path)
    model_df['Year'] = model_df['Month'].dt.year
    model_df = model_df.merge(pop_df, on=['LSOA code', 'Year'], how='left')

    # PTAL
    ptal_df = load_ptal(ptal_path)
    model_df = model_df.merge(ptal_df, on='LSOA code', how='left')
    # Digital propensity score
    dp_df = load_online_share(digital_propensity_path)
    model_df = model_df.merge(dp_df, on='LSOA code', how='left')
    # Energy
    mapping = load_lsoa_msoa_mapping(mapping_path)
    energy_ms = load_energy_all_dwellings(energy_path)
    mapping = mapping[mapping['LSOA code'].isin(model_df['LSOA code'])]
    energy_lsoa = mapping.merge(energy_ms, on='MSOA code', how='left').drop(columns=['MSOA code'])
    model_df = model_df.merge(energy_lsoa, on='LSOA code', how='left')

    # Lag features
    model_df = compute_lag_features(model_df)
    model_df = compute_custom_deprivation_score(model_df)
    model_df = add_moving_averages(model_df, target_col='Burglary Count', windows=[3, 6])

    
    return model_df