import pandas as pd
import numpy as np
import re

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
    grid = pd.MultiIndex.from_product([lsoas, months], names=['LSOA code','Month'])
    return grid.to_frame(index=False)


def aggregate_burglary_counts(grid: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Count burglaries per LSOA per month and merge onto the full grid.
    """
    observed = (
        df.assign(Month=df['Month'].dt.to_period('M').dt.to_timestamp())
          .groupby(['LSOA code','Month'])
          .size()
          .reset_index(name='Burglary Count')
    )
    base = grid.merge(observed, on=['LSOA code','Month'], how='left')
    base['Burglary Count'] = base['Burglary Count'].fillna(0)
    return base


def load_imd_scores(path: str) -> pd.DataFrame:
    """
    Load IMD domain scores, pivot to wide, and rename for merging.
    """
    print("Adding IMD")
    imd = pd.read_csv(
        path,
        usecols=['FeatureCode','Indices of Deprivation','Measurement','Value'],
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
    Load LSOA population estimates from sheets named like
    'Mid-2019 LSOA 2021', etc., returning ['LSOA code','Year','Population'].
    """
    print("Adding population estimates")
    xls = pd.ExcelFile(path)
    pop_list = []

    for sheet in xls.sheet_names:
        if sheet.startswith('Mid-') and 'LSOA' in sheet:
            df = pd.read_excel(path, sheet_name=sheet, header=3)
            # Identify LSOA code and Total columns
            code_col = next(c for c in df.columns if 'LSOA' in c and 'Code' in c)
            total_col = 'Total'
            sub = df[[code_col, total_col]].copy()
            sub.columns = ['LSOA code', 'Population']

            # Extract year
            m = re.search(r'(\d{4})', sheet)
            if not m:
                raise ValueError(f"Cannot parse year from sheet name '{sheet}'")
            sub['Year'] = int(m.group(1))
            pop_list.append(sub)

    pop_df = pd.concat(pop_list, ignore_index=True)
    return pop_df


def load_lsoa_msoa_mapping(path: str) -> pd.DataFrame:
    """
    Load LSOA->MSOA mapping from CSV columns 'lsoa21cd' & 'msoa21cd'.
    """
    df = pd.read_csv(path, encoding='latin-1', low_memory=False)
    required = {'lsoa21cd','msoa21cd'}
    if not required.issubset(df.columns):
        raise ValueError(f"Mapping CSV missing columns: {required - set(df.columns)}")
    mapping = (
        df[['lsoa21cd','msoa21cd']]
          .rename(columns={'lsoa21cd':'LSOA code','msoa21cd':'MSOA code'})
          .drop_duplicates()
    )
    return mapping


def load_ptal(path: str) -> pd.DataFrame:
    print("Adding Public Transport")
    df = pd.read_csv(path)
    df = df.rename(columns={'LSOA2011':'LSOA code'})
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

    return df[['LSOA code','AvPTAI2015','PTAL']]


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
            temp = temp.rename(columns={code_col:'MSOA code', dwell_col:f'ED_{sheet}'})
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
    energy = energy[['MSOA code','Energy_All']]
    energy['Energy_All'] = energy['Energy_All'].fillna(energy['Energy_All'].median())

    return energy


def compute_lag_features(df: pd.DataFrame, lags=(1,2)) -> pd.DataFrame:
    print("Computing lags")
    df = df.sort_values(['LSOA code','Month'])
    for lag in lags:
        df[f'Lag{lag}'] = df.groupby('LSOA code')['Burglary Count'].shift(lag).fillna(0)
    return df


def preprocess_data(
    burglary_path: str,
    imd_path: str,
    pop_path: str,
    energy_path: str,
    mapping_path: str,
    ptal_path: str,
    london_lsoa_path: str
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
    model_df = model_df.merge(pop_df, on=['LSOA code','Year'], how='left')

    # PTAL
    ptal_df = load_ptal(ptal_path)
    model_df = model_df.merge(ptal_df, on='LSOA code', how='left')

    # Energy
    mapping = load_lsoa_msoa_mapping(mapping_path)
    energy_ms = load_energy_all_dwellings(energy_path)
    mapping = mapping[mapping['LSOA code'].isin(model_df['LSOA code'])]
    energy_lsoa = mapping.merge(energy_ms, on='MSOA code', how='left').drop(columns=['MSOA code'])
    model_df = model_df.merge(energy_lsoa, on='LSOA code', how='left')

    # Lag features
    model_df = compute_lag_features(model_df)
    return model_df

