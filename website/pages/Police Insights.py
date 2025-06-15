import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import numpy as np
from shapely.geometry import Point
import json
import os
from difflib import get_close_matches
import plotly.express as px
import plotly.graph_objects as go
from libpysal.weights import Queen

st.set_page_config(page_title="Police Insights", layout="centered")

st.markdown("""
<style>
/* Global background */
.stApp {
    background-color: #eafafa !important;
    color: #000000 !important;
}

/* Sidebar text color fix */
section[data-testid="stSidebar"] .css-1d391kg,
section[data-testid="stSidebar"] .css-qrbaxs {
    color: white !important;
}

/* Hide theme switcher */
[data-testid="theme-toggle"] {
    display: none !important;
}

/* Fix input and textarea fields */
input, textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #999 !important;
}
input::placeholder, textarea::placeholder {
    color: #444 !important;
}

/* Fix button styling */
button {
    background-color: #007acc !important;
    color: white !important;
    border: none !important;
}
button:hover {
    background-color: #005fa3 !important;
}

/* Markdown/text elements */
.css-qrbaxs, .css-1d391kg, .css-10trblm, .css-1v0mbdj {
    color: #000000 !important;
}

/* Expanders & container borders */
.st-expander, .st-c1, .st-c2 {
    border: 1px solid #bbb !important;
    background-color: #f0fdfa !important;
    color: #000 !important;
}

/* Improve light-on-light visibility inside colored boxes */
div[style*="background-color: rgb(200, 245, 200)"],  /* Light green */
div[style*="background-color: rgb(255, 220, 220)"],  /* Light red */
div[style*="background-color: rgb(220, 220, 255)"] { /* Light blue */
    color: #000 !important;
}

/* Fix high-risk dark red alert box text visibility */
div[style*="background-color: #200000"] {
    color: #fff !important;
}

/* Fix folium map container height */
.folium-container,
.folium-container > div,
.folium-container iframe,
.stFolium {
    height: auto !important;
    min-height: 0 !important;
    padding-bottom: 0 !important;
    margin-bottom: 0 !important;
}
</style>
""", unsafe_allow_html=True)



def get_explanation(col, value):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = None

    explanations = {
        'Burglary_Probability': (
            f"{'High risk' if numeric_value is not None and numeric_value > 0.8 else 'Medium risk' if numeric_value is not None and numeric_value > 0.25 else 'Low risk'} - "
            "This score represents the estimated probability that at least one burglary will occur in this LSOA in the upcoming time window. "
            "A higher probability (closer to 1) indicates increased risk based on recent trends and local factors."
        ),
        'Predicted_Count': (
            f"{'High expected count' if numeric_value is not None and numeric_value > 3 else 'Moderate expected count' if numeric_value is not None and numeric_value > 1 else 'Low expected count'} - "
            "This is the model's predicted number of burglaries for this area. While not a guaranteed number, it reflects expected crime activity based on historical and spatial patterns."
        ),
        'Risk_Level': (
            f"{value} risk area - "
            "This qualitative label (e.g., 'Low', 'Medium', 'High') summarizes the overall burglary risk based on both predicted probability and contextual risk features."
        ),
        'b. Income Deprivation Domain': (
            f"{'High income deprivation' if numeric_value is not None and numeric_value > 0 else 'Low deprivation'} - "
            "This indicator captures the proportion of the population experiencing low income. High values suggest economic vulnerability, which may be correlated with certain types of crime."
        ),
        'c. Employment Deprivation Domain': (
            f"{'High employment deprivation' if numeric_value is not None and numeric_value > 0 else 'Low deprivation'} - "
            "Reflects the percentage of the working-age population excluded from the labor market. High unemployment may contribute to social instability or reduced surveillance in the area."
        ),
        'd. Education, Skills and Training Domain': (
            f"{'Educational deprivation' if numeric_value is not None and numeric_value > 0 else 'Low deprivation'} - "
            "Measures lack of educational attainment and skills among children and adults. Areas with poor education may face long-term socioeconomic challenges."
        ),
        'e. Health Deprivation and Disability Domain': (
            f"{'Health-deprived area' if numeric_value is not None and numeric_value > 0 else 'Healthier area'} - "
            "Indicates the risk of premature death and reduced quality of life due to poor health. Health deprivation may impact social cohesion and vulnerability."
        ),
        'f. Crime Domain': (
            f"{'High crime index' if numeric_value is not None and numeric_value > 0 else 'Low crime index'} - "
            "This score reflects the risk of personal and property crimes based on past records. High values suggest more historical crime activity in the area."
        ),
        'g. Barriers to Housing and Services Domain': (
            f"{'Significant barriers' if numeric_value is not None and numeric_value > 10 else 'Moderate barriers' if numeric_value is not None and numeric_value > 5 else 'Fewer barriers'} - "
            "Measures accessibility to essential services like housing, schools, and GPs. Higher values indicate physical or financial obstacles to housing and public services."
        ),
        'h. Living Environment Deprivation Domain': (
            f"{'Degraded living environment' if numeric_value is not None and numeric_value > 20 else 'Better living conditions'} - "
            "Assesses quality of housing, air pollution, and road safety. Poorer living environments may reduce informal surveillance and increase crime risk."
        ),
        'Mean Age': (
            f"{'Older population' if numeric_value is not None and numeric_value > 40 else 'Younger' if numeric_value is not None and numeric_value < 30 else 'Mixed-age population'} - "
            "The average age of residents. Age structure can influence neighborhood dynamics, with younger populations sometimes associated with higher mobility and less residential stability."
        ),
        'AvPTAI2015': (
            f"{'Good Public Transport' if numeric_value is not None and numeric_value > 70 else 'Moderate Public Transport' if numeric_value is not None and numeric_value > 40 else 'Lower Public Transport'} - "
            "This index represents with 100 being an amazing access to public transport in the area."
        ),
        'Digital Propensity Score': (
            f"{'Digitally engaged' if numeric_value is not None and numeric_value > 0.7 else 'Moderate' if numeric_value is not None and numeric_value > 0.4 else 'Low engagement'} - "
            "Estimates how likely residents are to use digital tools and services. High scores indicate better digital accessibility, which may influence access to services and communication."
        ),
        'Energy_All': (
            f"{'High energy use' if numeric_value is not None and numeric_value > 50 else 'Low energy use'} - "
            "Represents average energy consumption in the area. May indirectly reflect household size, property size, or affluence."
        ),
        'Burglary Count_MA3': (
            f"{'Increasing trend' if numeric_value is not None and numeric_value > 1 else 'Stable or declining'} - "
            "This is a 3-month moving average of burglary counts, capturing short-term changes and spikes in recent incidents."
        ),
        'Burglary Count_MA6': (
            f"{'Upward trend' if numeric_value is not None and numeric_value > 1 else 'Stable trend'} - "
            "This 6-month moving average provides a longer view of burglary activity trends in the area, useful for identifying sustained changes."
        ),
        'Burglary Count_SpatialLag1': (
            f"{'High risk from neighboring areas' if numeric_value is not None and numeric_value > 0.5 else 'Lower spillover risk'} - "
            "Shows how much crime is occurring in adjacent LSOAs. A higher value suggests that the surrounding area is also experiencing elevated burglary activity."
        ),
        'Energy_All_rev': (
            "This is a reversed version of the energy score, where lower values now represent higher energy use, to align with risk modeling conventions."
        ),
        'Digital Propensity Score_rev': (
            "This reversed score flips the original digital score. Higher values now represent lower digital engagement, useful when modeling risk factors inversely."
        ),
        'IMD_Custom_Rank': (
            "This shows the rank of the chosen LSOA by using our custom weights and ranking them with 1 being the least deprived LSOA."
        )

    }
    return explanations.get(col, "No detailed explanation available for this attribute.")



PASSWORD = "police123"

# Initialize session state for authentication
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

# If not authenticated, show login form
if not st.session_state.auth_ok:
    st.subheader("🔒 Enter password to access Police Insights")
    with st.form("password_form"):
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Submit")

        if submit:
            if password == PASSWORD:
                st.session_state.auth_ok = True
                st.success("✅ Access granted! You may now view police insights.")
                st.rerun()
            else:
                st.error("❌ Incorrect password.")
    st.stop()
else:
    st.title("🚓 Police Insights")
    st.write("Welcome! Here are your restricted insights:")

#Load predictions CSV
@st.cache_data
def load_predictions():
    try:
        df = pd.read_csv("../data/may_2025_predictions_lsoa.csv")

        if 'Year' in df.columns:
            try:
                if df['Year'].dtype == 'object' and df['Year'].astype(str).str.contains('-', na=False).any():
                    df['Year'] = pd.to_datetime(df['Year'], errors='coerce').dt.year
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int16')
            except:
                pass

        if 'Month' in df.columns:
            try:
                if df['Month'].dtype == 'object' and df['Month'].astype(str).str.contains('-', na=False).any():
                    df['Month'] = pd.to_datetime(df['Month'], errors='coerce').dt.month
                df['Month'] = pd.to_numeric(df['Month'], errors='coerce').astype('Int8')
            except:
                pass

        if 'Risk_Level' not in df.columns and 'Burglary_Probability' in df.columns:
            df['Risk_Level'] = pd.cut(
                df['Burglary_Probability'],
                bins=[0, 0.33, 0.66, 1.0],
                labels=['Low', 'Medium', 'High'],
                include_lowest=True
            ).astype(str)

        if 'LSOA code' in df.columns:
            df['LSOA code'] = df['LSOA code'].astype(str).str.strip()

        if 'Risk_Level' in df.columns:
            df['Risk_Level'] = df['Risk_Level'].astype('category')

        return df
    except Exception as e:
        st.error(f"Error loading predictions CSV: {e}")
        return pd.DataFrame()

df_pred = load_predictions()

if df_pred.empty:
    st.error("Failed to load predictions data")
    st.stop()

required_cols = ['LSOA code', 'Year', 'Month']
missing_cols = [col for col in required_cols if col not in df_pred.columns]
if missing_cols:
    st.error(f"CSV missing required columns: {missing_cols}")
    st.stop()

mode = st.sidebar.radio("View by", ["LSOA", "Ward"], index=0)

try:
    years = sorted([y for y in df_pred['Year'].unique() if pd.notna(y)])
    months = sorted([m for m in df_pred['Month'].unique() if pd.notna(m)])

    if not years or not months:
        st.error("No valid year/month data found")
        st.stop()

    selected_year = years[0]
    selected_month = months[0]

    df_filt = df_pred[
        (df_pred['Year'] == selected_year) &
        (df_pred['Month'] == selected_month)
    ].copy()

except Exception as e:
    st.error(f"Error in filtering: {e}")
    st.stop()

@st.cache_data
def load_lsoa_boundaries():
    """Load and preprocess LSOA boundaries from GeoJSON"""
    try:
        geojson_file = "../boundaries/london_lsoa.geojson"

        if not os.path.exists(geojson_file):
            st.error(f"GeoJSON file not found: {geojson_file}")
            return None

        try:
            gdf = gpd.read_file(geojson_file)
        except Exception as e:
            st.error(f"Error reading GeoJSON with geopandas: {e}")
            try:
                with open(geojson_file, 'r') as f:
                    geojson_data = json.load(f)
                gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
            except Exception as e2:
                st.error(f"Error reading GeoJSON as JSON: {e2}")
                return None

        
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        else:
            gdf = gdf.to_crs(epsg=4326)

        
        gdf['geometry'] = gdf.geometry.simplify(tolerance=0.0001, preserve_topology=True)

        rename_dict = {}
        lsoa_code_candidates = [
            'LSOA21CD', 'lsoa21cd', 'LSOA21_CD', 'lsoa21_cd',
            'LSOA11CD', 'lsoa11cd', 'LSOA11_CD', 'lsoa11_cd',
            'LSOA_CODE', 'lsoa_code', 'code', 'Code'
        ]
        lsoa_name_candidates = [
            'LSOA21NM', 'lsoa21nm', 'LSOA21_NM', 'lsoa21_nm',
            'LSOA11NM', 'lsoa11nm', 'LSOA11_NM', 'lsoa11_nm',
            'LSOA_NAME', 'lsoa_name', 'name', 'Name'
        ]

        for col in gdf.columns:
            if col in lsoa_code_candidates or col.upper() in [c.upper() for c in lsoa_code_candidates]:
                rename_dict[col] = 'LSOA code'
                break
        for col in gdf.columns:
            if col in lsoa_name_candidates or col.upper() in [c.upper() for c in lsoa_name_candidates]:
                rename_dict[col] = 'LSOA name'
                break

        if rename_dict:
            gdf = gdf.rename(columns=rename_dict)

        if 'LSOA code' in gdf.columns:
            gdf['LSOA code'] = gdf['LSOA code'].astype(str).str.strip()

        return gdf

    except Exception as e:
        st.error(f"Error reading LSOA GeoJSON file: {e}")
        return None

@st.cache_data
def load_ward_boundaries():
    """Load and preprocess Ward boundaries"""
    try:
        shp_ward = "../boundaries/London_Wards_2024.shp"

        if not os.path.exists(shp_ward):
            st.error(f"Ward shapefile not found: {shp_ward}")
            return None

        gdf = gpd.read_file(shp_ward)
        gdf['geometry'] = gdf.geometry.simplify(tolerance=0.001, preserve_topology=True)
        gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception as e:
        st.error(f"Error reading ward shapefile: {e}")
        return None

@st.cache_data
def load_lsoa_ward_mapping():
    """Load LSOA to Ward mapping with improved error handling"""
    try:
        map_file = "../data/LSOA_(2021)_to_Electoral_Ward_(2024)_to_LAD_(2024)_Best_Fit_Lookup_in_EW.csv"

        if not os.path.exists(map_file):
            st.error(f"Mapping file not found: {map_file}")
            return None

        df_map = pd.read_csv(map_file, dtype=str)


        column_mappings = {}
        lsoa_col_candidates = [
            'LSOA21CD', 'lsoa21cd', 'LSOA21_CD', 'lsoa21_cd',
            'LSOA11CD', 'lsoa11cd', 'LSOA11_CD', 'lsoa11_cd',
            'LSOA_CODE', 'lsoa_code'
        ]
        ward_code_candidates = [
            'WD24CD', 'wd24cd', 'WD24_CD', 'wd24_cd',
            'WARD_CODE', 'ward_code', 'Ward_Code'
        ]
        ward_name_candidates = [
            'WD24NM', 'wd24nm', 'WD24_NM', 'wd24_nm',
            'WARD_NAME', 'ward_name', 'Ward_Name'
        ]

        for col in df_map.columns:
            if col in lsoa_col_candidates or any(c.upper() == col.upper() for c in lsoa_col_candidates):
                column_mappings[col] = 'LSOA code'
                break
        for col in df_map.columns:
            if col in ward_code_candidates or any(c.upper() == col.upper() for c in ward_code_candidates):
                column_mappings[col] = 'Ward code'
                break
        for col in df_map.columns:
            if col in ward_name_candidates or any(c.upper() == col.upper() for c in ward_name_candidates):
                column_mappings[col] = 'Ward name'
                break

        df_map = df_map.rename(columns=column_mappings)
        required_cols = ['LSOA code', 'Ward code', 'Ward name']
        df_map = df_map[required_cols].dropna()
        for col in required_cols:
            df_map[col] = df_map[col].astype(str).str.strip()
        df_map = df_map[(df_map['LSOA code'] != '') & (df_map['Ward code'] != '') & (df_map['Ward name'] != '')]
        return df_map

    except Exception as e:
        st.error(f"Error reading mapping file: {e}")
        return None

if 'clicked_ward' not in st.session_state:
    st.session_state.clicked_ward = None
if 'clicked_lsoa' not in st.session_state:
    st.session_state.clicked_lsoa = None

try:
    if mode == "LSOA":
        gdf_base = load_lsoa_boundaries()
        if gdf_base is None:
            st.stop()
        if 'LSOA code' not in gdf_base.columns:
            st.error("LSOA GeoJSON missing LSOA code field")
            st.stop()

        df_filt['LSOA code'] = df_filt['LSOA code'].astype(str).str.strip()

        gdf = gdf_base.merge(df_filt, on='LSOA code', how='left')

        try:
            if 'Burglary_Probability' in gdf.columns and gdf['Burglary_Probability'].notna().sum() > 1:
                values = gdf['Burglary_Probability'].fillna(0)
                w = Queen.from_dataframe(gdf)

                bins = pd.cut(
                    values,
                    bins=[-np.inf, 0.33, 0.66, np.inf],
                    labels=['Low', 'Medium', 'High']
                )

                def get_neighbor_bin(i):
                    neighbors = w.neighbors[i]
                    if not neighbors:
                        return 'No neighbors'
                    neighbor_vals = values.iloc[neighbors]
                    avg = neighbor_vals.mean()
                    if avg <= 0.33:
                        return 'Low'
                    elif avg <= 0.66:
                        return 'Medium'
                    else:
                        return 'High'
                gdf['Own_Risk'] = bins.astype(str)
                gdf['Neighbor_Risk'] = [get_neighbor_bin(i) for i in range(len(gdf))]
                gdf['Cluster_Combo'] = gdf['Own_Risk'] + '-' + gdf['Neighbor_Risk']
            else:
                gdf['Own_Risk'] = np.nan
                gdf['Neighbor_Risk'] = np.nan
                gdf['Cluster_Combo'] = np.nan
        except Exception as e:
            st.warning(f"Could not compute spatial cluster labels: {e}")
            gdf['Own_Risk'] = np.nan
            gdf['Neighbor_Risk'] = np.nan
            gdf['Cluster_Combo'] = np.nan

        # Use Risk_Level for coloring
        color_field = 'Own_Risk'

        tooltip_field = 'Burglary_Probability'
        legend_label = 'Risk Level'
        name_field = 'LSOA name' if 'LSOA name' in gdf.columns else 'LSOA code'
        code_field = 'LSOA code'

        lsoa_names = gdf_base[['LSOA code', 'LSOA name']].drop_duplicates()
        df_display = df_filt.merge(lsoa_names, on='LSOA code', how='left')
        desired_cols = ['LSOA code', 'LSOA name']
        for c in ['Predicted_Count', 'Burglary_Probability',  'Own_Risk', 'Neighbor_Risk', 'Cluster_Combo','IMD_Custom_Rank']:
            if c in gdf.columns:
                desired_cols.append(c)
        df_display = gdf[desired_cols].reset_index(drop=True)

    else:  
        df_map = load_lsoa_ward_mapping()
        gdf_base = load_ward_boundaries()
        if df_map is None or gdf_base is None:
            st.error("Cannot load ward mapping or boundaries")
            st.stop()

        def find_best_column(gdf, candidates):
            for candidate in candidates:
                if candidate in gdf.columns:
                    return candidate
                for col in gdf.columns:
                    if col.upper() == candidate.upper():
                        return col
            return None

        ward_code_candidates = ['GSS_CODE', 'WD24CD', 'WARD_CODE', 'CODE', 'gss_code', 'ward_code']
        ward_name_candidates = ['NAME', 'WD24NM', 'WARD_NAME', 'name', 'ward_name']

        code_field_orig = find_best_column(gdf_base, ward_code_candidates)
        name_field_orig = find_best_column(gdf_base, ward_name_candidates)

        if code_field_orig is None or name_field_orig is None:
            text_cols = [col for col in gdf_base.columns if gdf_base[col].dtype == 'object']
            if len(text_cols) >= 2:
                col_stats = []
                for col in text_cols:
                    avg_len = gdf_base[col].astype(str).str.len().mean()
                    unique_ratio = gdf_base[col].nunique() / len(gdf_base)
                    col_stats.append((col, avg_len, unique_ratio))
                col_stats.sort(key=lambda x: x[1])
                if code_field_orig is None:
                    code_field_orig = col_stats[0][0]
                if name_field_orig is None:
                    name_field_orig = col_stats[1][0] if len(col_stats) > 1 else col_stats[0][0]

        if code_field_orig is None or name_field_orig is None:
            st.error("Cannot identify ward code/name columns in ward shapefile")
            st.stop()

        df_join = pd.merge(df_filt, df_map, on='LSOA code', how='inner')
        if df_join.empty:
            st.error("No matching data found between predictions and ward mapping")
            st.stop()

        agg_col = None
        if 'Burglary_Probability' in df_join.columns:
            agg_col = 'Burglary_Probability'
        elif 'Predicted_Count' in df_join.columns:
            agg_col = 'Predicted_Count'
        else:
            st.error("No suitable column for aggregation")
            st.stop()

        ward_agg = df_join.groupby(['Ward code', 'Ward name'], as_index=False).agg({
            'Burglary_Probability': 'mean',
            'Predicted_Count': 'sum'
        })

        legend_label = 'Average Burglary Probability'

        df_ward_display = ward_agg[['Ward name', agg_col]].copy()
        df_ward_display = df_ward_display.rename(columns={agg_col: 'Average_Burglary_Probability'})

        df_ward_display = ward_agg[['Ward name', agg_col]].copy()
        df_ward_display = df_ward_display.rename(columns={agg_col: 'Average_Burglary_Probability'})

        gdf = gdf_base.rename(columns={code_field_orig: 'Ward code', name_field_orig: 'Ward name'})
        def normalize_text(text):
            return "" if pd.isna(text) else str(text).strip().upper()

        gdf['Ward code'] = gdf['Ward code'].astype(str).str.strip()
        gdf['Ward name'] = gdf['Ward name'].astype(str).str.strip()
        gdf['Ward name normalized'] = gdf['Ward name'].apply(normalize_text)

        ward_agg['Ward code'] = ward_agg['Ward code'].astype(str).str.strip()
        ward_agg['Ward name'] = ward_agg['Ward name'].astype(str).str.strip()
        ward_agg['Ward name normalized'] = ward_agg['Ward name'].apply(normalize_text)

        merge_success = False


        gdf_temp = gdf.merge(ward_agg[['Ward code', 'Ward name', agg_col]], on=['Ward code', 'Ward name'], how='left')
        if gdf_temp[agg_col].notna().sum() > 0:
            gdf = gdf_temp; merge_success = True

        if not merge_success:
            gdf_temp = gdf.merge(ward_agg[['Ward code', agg_col]], on='Ward code', how='left')
            if gdf_temp[agg_col].notna().sum() > 0:
                gdf = gdf_temp; merge_success = True

        if not merge_success:
            gdf_temp = gdf.merge(ward_agg[['Ward name normalized', agg_col]], on='Ward name normalized', how='left')
            if gdf_temp[agg_col].notna().sum() > 0:
                gdf = gdf_temp; merge_success = True

        if not merge_success:
            boundary_names = gdf['Ward name normalized'].unique()
            mapping_names = ward_agg['Ward name normalized'].unique()
            name_mapping = {}
            for mapping_name in mapping_names:
                if mapping_name.strip():
                    matches = get_close_matches(mapping_name, boundary_names, n=1, cutoff=0.8)
                    if matches:
                        name_mapping[mapping_name] = matches[0]
            if name_mapping:
                ward_agg_fuzzy = ward_agg.copy()
                ward_agg_fuzzy['Ward name fuzzy'] = ward_agg_fuzzy['Ward name normalized'].map(name_mapping)
                ward_agg_fuzzy = ward_agg_fuzzy.dropna(subset=['Ward name fuzzy'])
                if not ward_agg_fuzzy.empty:
                    gdf_temp = gdf.merge(
                        ward_agg_fuzzy[['Ward name fuzzy', agg_col]].rename(columns={'Ward name fuzzy': 'Ward name normalized'}),
                        on='Ward name normalized', how='left'
                    )
                    if gdf_temp[agg_col].notna().sum() > 0:
                        gdf = gdf_temp; merge_success = True

        if not merge_success:
            st.error("Unable to merge ward data with any strategy.")
            st.stop()

        color_field = agg_col
        tooltip_field = agg_col
        legend_label = 'Expected Burglary Count' if agg_col == 'Predicted_Count' else 'Average Burglary Probability'
        code_field = 'Ward code'
        name_field = 'Ward name'

except Exception as e:
    st.error(f"Error in data preparation: {e}")
    import traceback
    st.error(f"Full error: {traceback.format_exc()}")
    st.stop()

# Remove null geometries
gdf = gdf[gdf.geometry.notnull()]

st.header(f"{mode} Map: {selected_month}/{selected_year}")

if gdf.empty:
    st.error("No geometries available for this view and filters.")
    st.stop()

# Create the map
try:
    bounds = gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='CartoDB dark_matter'
    )

    def style_function(feature):
        """Style function: LSOA colored by Risk_Level; Ward by gradient if numeric."""
        props = feature['properties']
        if mode == "LSOA":
            # Color by Risk_Level
            risk_val = props.get('Risk_Level')
            risk_colors = {'Low': '#00ff00', 'Medium': '#ffaa00', 'High': '#ff0000'}
            return {
                'fillColor': risk_colors.get(str(risk_val), '#888888'),
                'color': 'white',
                'weight': 0.5,
                'fillOpacity': 0.7
            }
        else:
            # Ward mode: numeric gradient
            val = props.get(color_field)
            if pd.isna(val) or val is None:
                color = '#888888'
            else:
                try:
                    num = float(val)
                    if not hasattr(style_function, '_min_max'):
                        valid_vals = gdf[color_field].dropna().astype(float)
                        if len(valid_vals) > 0:
                            style_function._min_max = (valid_vals.min(), valid_vals.max())
                        else:
                            style_function._min_max = (0, 1)
                    mn, mx = style_function._min_max
                    pct = (num - mn) / (mx - mn) if mx > mn else 0
                    pct = max(0, min(1, pct))
                    red = int(255 * pct)
                    green = int(255 * (1 - pct))
                    color = f'#{red:02x}{green:02x}64'
                except:
                    color = '#888888'
            return {
                'fillColor': color,
                'color': 'white',
                'weight': 0.5,
                'fillOpacity': 0.7
            }

    # Tooltip fields
    tooltip_fields = [name_field, tooltip_field]
    tooltip_aliases = [mode, 'Burglary_Probability']
    # Add cluster labels
    if 'Cluster_Combo' in gdf.columns:
        tooltip_fields += ['Own_Risk', 'Neighbor_Risk', 'Cluster_Combo']
        tooltip_aliases += ['Own Risk', 'Neighbor Risk', 'Cluster']

    folium.GeoJson(
        gdf,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            sticky=False,
            labels=True
        )
    ).add_to(m)

    if mode == "Ward":
        # Calculate cluster labels 
        ward_values = gdf[color_field].fillna(0)
        w_ward = Queen.from_dataframe(gdf)
        
        # Calculate risk thresholds 
        risk_thresholds = ward_values.quantile([0.2, 0.4, 0.6, 0.8])

        # Add Own_Risk to gdf 
        gdf['Own_Risk'] = pd.cut(
            ward_values,
            bins=[-float('inf'),
                  risk_thresholds[0.2],
                  risk_thresholds[0.4],
                  risk_thresholds[0.6],
                  risk_thresholds[0.8],
                  float('inf')],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        ).astype(str)
        
        # Calculate neighbor risk 
        neighbor_risks = []
        for i in range(len(gdf)):
            neighbors = w_ward.neighbors[i]
            if not neighbors:
                neighbor_risks.append('No neighbors')
            else:
                neighbor_vals = ward_values.iloc[neighbors].mean()
                if neighbor_vals <= risk_thresholds[0.2]:
                    neighbor_risks.append('Vert Low')
                elif neighbor_vals <= risk_thresholds[0.4]:
                    neighbor_risks.append('Low')
                elif neighbor_vals <= risk_thresholds[0.6]:
                    neighbor_risks.append('Medium')
                elif neighbor_vals <= risk_thresholds[0.8]:
                    neighbor_risks.append('High')
                else:
                    neighbor_risks.append('Very High')

        # Add risk pattern information to GeoDataFrame
        gdf['Neighbor_Risk'] = neighbor_risks
        gdf['Risk_Pattern'] = gdf['Own_Risk'] + '-' + gdf['Neighbor_Risk']
        
        # Update GeoJSON layer with new fields and labels
        folium.GeoJson(
            gdf,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['Ward name', 'Burglary_Probability', 'Own_Risk', 'Neighbor_Risk', 'Risk_Pattern'],
                aliases=['Ward', 'Probability', 'Own Risk', 'Neighbor Risk', 'Risk Pattern'],
                localize=True,
                sticky=False,
                labels=True
            )
        ).add_to(m)

    map_data = st_folium(m, width=700, returned_objects=["last_clicked"])

    clicked = map_data.get('last_clicked')
    if clicked:
        try:
            click_point = Point(clicked['lng'], clicked['lat'])
            selected_area = gdf[gdf.geometry.contains(click_point)]
            if not selected_area.empty:
                row = selected_area.iloc[0]
                area_name = row.get(name_field, 'Unknown')
                area_code = row.get(code_field, 'Unknown')

                if mode == "LSOA":
                    st.session_state.clicked_lsoa = {
                        'code': area_code,
                        'name': area_name,
                        'data': row
                    }
                else:  # Ward
                    st.session_state.clicked_ward = {
                        'code': area_code,
                        'name': area_name
                    }

                st.subheader(f"Selected {mode}: {area_name}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Code", area_code)
                with col2:
                    if tooltip_field in row.index and pd.notna(row[tooltip_field]):
                        val = row[tooltip_field]
                        if mode == "LSOA":
                            st.metric("Burglary_Probability", f"{float(val):.3f}")
                        else:
                            st.metric(
                                legend_label,
                                f"{float(val):.2f}" if isinstance(val, (int, float, np.number)) else str(val)
                            )
        except Exception as e:
            st.error(f"Error processing map click: {e}")



    if mode == "LSOA":
        if st.session_state.clicked_lsoa is not None:
            clicked_lsoa = st.session_state.clicked_lsoa
            st.subheader(f"📊 Detailed Analysis: {clicked_lsoa['name']}")

            attribute_columns = [
                'Predicted_Count',
                'Burglary_Probability',
                'Risk_Level',
                'b. Income Deprivation Domain',
                'c. Employment Deprivation Domain',
                'd. Education, Skills and Training Domain',
                'e. Health Deprivation and Disability Domain',
                'g. Barriers to Housing and Services Domain',
                'h. Living Environment Deprivation Domain',
                'Mean Age',
                'AvPTAI2015',
                'Digital Propensity Score',
                'Energy_All',
                'IMD_Custom_Rank'
            ]

            col1, col2 = st.columns(2)
            row_data = clicked_lsoa['data']

            for i, col_name in enumerate(attribute_columns):
                if col_name in row_data.index:
                    value = row_data[col_name]
                    if pd.isna(value):
                        continue
                    target_col = col1 if i % 2 == 0 else col2
                    with target_col:
                        explanation = get_explanation(col_name, value)
                        if isinstance(value, (int, float, np.number)):
                            if col_name == 'Burglary_Probability':
                                formatted_value = f"{value:.3f}"
                            elif col_name in ['Predicted_Count', 'Burglary Count_MA3', 'Burglary Count_MA6']:
                                formatted_value = f"{value:.1f}"
                            else:
                                formatted_value = f"{value:.2f}"
                        else:
                            formatted_value = str(value)

                        st.markdown(f"""
                        <div class="attribute-card">
                            <strong>{col_name}</strong><br>
                            <span class="attribute-value">Value: {formatted_value}</span>
                            <div class="attribute-explanation">{explanation}</div>
                        </div>
                        """, unsafe_allow_html=True)

            if st.button("Clear Selection"):
                st.session_state.clicked_lsoa = None
                st.rerun()
        else:
            st.write("Click on an LSOA in the map above to see its detailed attributes.")
    else:  # Ward
        if st.session_state.clicked_ward is not None:
            clicked_ward = st.session_state.clicked_ward
            st.write(f"**Top 5 Highest Risk LSOAs in {clicked_ward['name']}**")

            df_map = load_lsoa_ward_mapping()
            if df_map is not None:
                ward_lsoas = df_map[
                    (df_map['Ward code'].str.strip() == clicked_ward['code']) |
                    (df_map['Ward name'].str.strip() == clicked_ward['name'])
                ]['LSOA code'].tolist()

                if ward_lsoas:
                    ward_data = df_filt[df_filt['LSOA code'].isin(ward_lsoas)].copy()

                    if not ward_data.empty:
                        sort_col = 'Burglary_Probability' if 'Burglary_Probability' in ward_data.columns else 'Predicted_Count'
                        if sort_col in ward_data.columns:
                            top_5 = ward_data.nlargest(5, sort_col)

                            lsoa_gdf = load_lsoa_boundaries()
                            if lsoa_gdf is not None and 'LSOA code' in lsoa_gdf.columns and 'LSOA name' in lsoa_gdf.columns:
                                lsoa_names = lsoa_gdf[['LSOA code', 'LSOA name']].drop_duplicates()
                                top_5 = top_5.merge(lsoa_names, on='LSOA code', how='left')

                            display_cols = ['LSOA code', 'LSOA name']
                            if 'Burglary_Probability' in top_5.columns:
                                display_cols.append('Burglary_Probability')
                            if 'Predicted_Count' in top_5.columns:
                                display_cols.append('Predicted_Count')
                            if 'Risk_Level' in top_5.columns:
                                display_cols.append('Risk_Level')

                            display_cols = [col for col in display_cols if col in top_5.columns]

                            st.dataframe(
                                top_5[display_cols].reset_index(drop=True),
                                use_container_width=True
                            )
                        else:
                            st.write("No risk data available for this ward’s LSOAs.")
                    else:
                        st.write("No LSOA data found for this ward.")
                else:
                    st.write("No LSOAs found for this ward.")
            else:
                st.write("Ward mapping not available.")
        else:
            st.write("Click on a Ward in the map above to see its top-5 LSOAs.")


    if mode == "LSOA":
        st.markdown("## 📋 All LSOAs – Name, Predicted Count, Probability, Risk")
        st.dataframe(df_display, use_container_width=True)
    else:  
        st.markdown("## 📋 All Wards – Name, Risk Pattern and Aggregated Values")
        
        # Calculate risk patterns for ward_agg
        ward_values = ward_agg[agg_col].fillna(0)
        risk_thresholds = ward_values.quantile([0.2, 0.4, 0.6, 0.8])
        
        # Add Own_Risk
        ward_agg['Own_Risk'] = pd.cut(
            ward_values,
            bins=[-float('inf'), 
                  risk_thresholds[0.2], 
                  risk_thresholds[0.4],
                  risk_thresholds[0.6],
                  risk_thresholds[0.8], 
                  float('inf')],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        ).astype(str)
        
        # Create a spatial weights matrix using ward geometries
        ward_gdf = gdf[['Ward name', 'geometry']].drop_duplicates(subset=['Ward name'])
        ward_gdf = ward_gdf.merge(ward_agg[['Ward name', agg_col]], on='Ward name', how='right')
        w_ward = Queen.from_dataframe(ward_gdf)
        
        # Calculate neighbor risk
        neighbor_risks = []
        for _, row in ward_agg.iterrows():
            ward_name = row['Ward name']
            ward_idx = ward_gdf[ward_gdf['Ward name'] == ward_name].index
            
            if len(ward_idx) == 0:
                neighbor_risks.append('No neighbors')
                continue
                
            ward_idx = ward_idx[0]
            neighbors = w_ward.neighbors[ward_idx]
            
            if not neighbors:
                neighbor_risks.append('No neighbors')
            else:
                neighbor_vals = ward_gdf.iloc[neighbors][agg_col].mean()
                if neighbor_vals <= risk_thresholds[0.2]:
                    neighbor_risks.append('Very Low')
                elif neighbor_vals <= risk_thresholds[0.4]:
                    neighbor_risks.append('Low')
                elif neighbor_vals <= risk_thresholds[0.6]:
                    neighbor_risks.append('Medium')
                elif neighbor_vals <= risk_thresholds[0.8]:
                    neighbor_risks.append('High')
                else:
                    neighbor_risks.append('Very High')
        
        ward_agg['Neighbor_Risk'] = neighbor_risks
        ward_agg['Risk_Pattern'] = ward_agg['Own_Risk'] + '-' + ward_agg['Neighbor_Risk']
        
        # Create display DataFrame
        df_ward_display = ward_agg[['Ward name', agg_col, 'Predicted_Count', 'Own_Risk', 'Neighbor_Risk', 'Risk_Pattern']].copy()
        df_ward_display = df_ward_display.rename(columns={
            agg_col: 'Burglary Probability',
            'Predicted_Count': 'Predicted Burglaries'
        })
        
        st.dataframe(
            df_ward_display.sort_values('Burglary Probability', ascending=False).reset_index(drop=True),
            use_container_width=True
        )


    if mode == "LSOA":
        csv_data = df_display.to_csv(index=False)
        st.download_button(
            label="Download LSOA Data",
            data=csv_data,
            file_name=f"lsoa_data_{selected_month}_{selected_year}.csv",
            mime="text/csv"
        )
    else:
        ward_csv = ward_agg.to_csv(index=False)
        st.download_button(
            label="Download Ward Data",
            data=ward_csv,
            file_name=f"ward_data_{selected_month}_{selected_year}.csv",
            mime="text/csv"
        )

except Exception as e:
    st.error(f"Error displaying data table: {e}")



if mode == "Ward":
    st.markdown("---")
    st.header("🚓 Risk-Based Allocation of Police Officers")

    ward_allocation_data = ward_agg.copy()
    ward_allocation_data = ward_allocation_data.dropna(subset=['Ward code']).drop_duplicates(subset=['Ward code'])
    
    # Determine which column to use for risk assessment
    allocation_agg_col = 'Predicted_Count'
    
    # Sort wards by risk metric and assign risk categories
    ward_allocation_data = ward_allocation_data.sort_values(by=allocation_agg_col, ascending=False).reset_index(drop=True)
    total_wards = len(ward_allocation_data)

    # Define cutoffs for 5 risk tiers
    high_cutoff = int(total_wards * 0.10)  
    med_high_cutoff = high_cutoff + int(total_wards * 0.05)  
    medium_cutoff = med_high_cutoff + int(total_wards * 0.35)  
    med_low_cutoff = medium_cutoff + int(total_wards * 0.05) 

    # Assign risk categories and officers
    def assign_risk_and_officers(index):
        if index < high_cutoff:
            return 'High', 100
        elif index < med_high_cutoff:
            return 'Medium-High', 75
        elif index < medium_cutoff:
            return 'Medium', 60
        elif index < med_low_cutoff:
            return 'Medium-Low', 45
        else:
            return 'Low', 30

    # Apply risk and officer assignments
    risk_officers = [assign_risk_and_officers(i) for i in range(len(ward_allocation_data))]
    ward_allocation_data['Risk_Category'] = [x[0] for x in risk_officers]
    ward_allocation_data['Officers_Allocated'] = [x[1] for x in risk_officers]

    # Now create display columns dictionary
    display_cols = {
        'Ward name': 'Ward Name',
        'Risk_Category': 'Risk Level',
        'Officers_Allocated': 'Officers',
        allocation_agg_col: 'Risk Score'
    }

    # Metrics
    total_officers = ward_allocation_data['Officers_Allocated'].sum()
    counts = ward_allocation_data['Risk_Category'].value_counts()
    high_risk_wards = counts.get('High', 0)
    med_high_risk_wards = counts.get('Medium-High', 0)
    medium_risk_wards = counts.get('Medium', 0)
    med_low_risk_wards = counts.get('Medium-Low', 0)
    low_risk_wards = counts.get('Low', 0)

    st.subheader("📊 Allocation Summary")


    row1_col1, row1_col2, row1_col3 = st.columns([1, 1, 1])
    with row1_col1:
        st.metric("Total Wards", f"{total_wards}")
    with row1_col2:
        st.metric("Total Officers", f"{total_officers:,}")
    with row1_col3:
        st.metric("Allocation Model", "5-Tier Split")

    st.markdown("")


    row2_col1, row2_col2, row2_col3 = st.columns([1, 1, 1])
    with row2_col1:
        st.metric("🔴 High Risk", f"{high_risk_wards} wards", f"{(high_risk_wards/total_wards)*100:.1f}%")
    with row2_col2:
        st.metric("🟠 Medium-High", f"{med_high_risk_wards} wards", f"{(med_high_risk_wards/total_wards)*100:.1f}%")
    with row2_col3:
        st.metric("🟡 Medium", f"{medium_risk_wards} wards", f"{(medium_risk_wards/total_wards)*100:.1f}%")

    row3_col1, row3_col2 = st.columns([1, 1])
    with row3_col1:
        st.metric("🟢 Medium-Low", f"{med_low_risk_wards} wards", f"{(med_low_risk_wards/total_wards)*100:.1f}%")
    with row3_col2:
        st.metric("🟩 Low", f"{low_risk_wards} wards", f"{(low_risk_wards/total_wards)*100:.1f}%")



    st.subheader("Highest Allocation Wards")

    top_wards = ward_allocation_data.nlargest(15, 'Officers_Allocated')[
        ['Ward name', 'Risk_Category', 'Officers_Allocated', allocation_agg_col]
    ].reset_index(drop=True)

    display_cols = {
        'Ward name': 'Ward Name',
        'Risk_Category': 'Risk Level',
        'Officers_Allocated': 'Officers',
        allocation_agg_col: 'Predicted Burglaries'
    }

    top_wards_display = top_wards.rename(columns=display_cols)

    st.dataframe(top_wards_display, use_container_width=True, hide_index=True)



    st.subheader("🔍 Search Ward Allocation")

    ward_names = ward_allocation_data['Ward name'].sort_values().unique()
    selected_ward = st.selectbox("Select a Ward", ward_names)

    ward_info = ward_allocation_data[ward_allocation_data['Ward name'] == selected_ward]

    if not ward_info.empty:
        ward_details = ward_info.iloc[0]
        st.markdown(f"""
        **Ward Name:** {ward_details['Ward name']}  
        **Risk Level:** {ward_details['Risk_Category']}  
        **Officers Allocated:** {ward_details['Officers_Allocated']}  
        **Risk Score:** {ward_details[allocation_agg_col]:.2f}
        """)
    else:
        st.warning("Selected ward not found in the data.")



    # Download allocation data
    st.subheader("💾 Download Allocation Data")

    download_data = ward_allocation_data[['Ward name', 'Ward code', 'Risk_Category', 'Officers_Allocated', allocation_agg_col]].copy()
    download_data = download_data.rename(columns={
        'Ward name': 'Ward_Name',
        'Ward code': 'Ward_Code',
        'Risk_Category': 'Risk_Level',
        'Officers_Allocated': 'Officers_Allocated',
        allocation_agg_col: 'Risk_Score'
    })

    csv_allocation = download_data.to_csv(index=False)

    st.download_button(
        label="📥 Download Ward Allocation Data (CSV)",
        data=csv_allocation,
        file_name=f"police_allocation_baseline_{selected_month}_{selected_year}.csv",
        mime="text/csv"
    )

    # Implementation notes
    with st.expander("ℹ️ Implementation and Justification"):
        st.markdown("""
                        
        ### Police Deployment Strategy: **5-Tier Allocation Model**

        - We use a five-tier allocation strategy to assign officers based on the predicted burglary numbers that would occur at the ward level in a month. 
        - This strategy covers both practical policing approaches and insights from crime concentration research.
        - We maintain a base presence even in the safest communities to uphold trust and fairness.
        - Allocation is scalable: Can be adjusted if more granular crime data or risk tiers are added.
        - Simple and intuitive.
            
        ---

        ### Allocation Logic

        Wards are classified using predicted burglary counts and sorted into:

        - 🟥 **High Risk** (Top 10%) → **100 officers**
        - 🟠 **Medium-High Risk** (Next 5%) → **75 officers**
        - 🟡 **Medium Risk** (Next 35%) → **60 officers**
        - 🟢 **Medium-Low Risk** (Next 5%) → **45 officers**
        - 🟩 **Low Risk** (Remaining 45%) → **30 officers**

        We originally used a **3-tier model** (*Low*, *Medium*, *High*), but found that the discrete jumps between allocation levels (from 30 → 60 → 100 officers) were too abrupt — especially for wards falling near the boundaries between tiers.

        To improve the fairness and realism of deployment, we introduced two **intermediate tiers** — **Medium-High** and **Medium-Low** — which help ease the transition in officer numbers between risk levels.

        This results in a more radual, proportional, and data-informed gradient of policing effort that better reflects the continuous nature of crime risk across London’s wards, while staying within the total available police force size (~34,000 officers across ~680 wards).

        ---

        ### Academic & Policy Support

        - **Crime concentration theory**: 5–10% of areas account for the majority of urban crimes.
        - **Problem-oriented policing**: Focusing on high-need areas produces higher crime reduction per officer deployed (College of Policing, 2022)
        - **Operational parallels**: NYPD's *Operation Impact*, Met's *Operation Bumblebee*, and West Midlands' *Impact Zones* followed similar focused-distribution patterns.
                    
        ---

        ### Allocation times and Future Work

        """)

    with st.expander("🕒 Future Work: Patrol Time Strategy & Officer Presence"):
        st.markdown("""
        
        - **Burglary is not random** — it’s concentrated in hotspots and follows predictable times based on our research.
        - Research shows that **focused, visible patrols in high-risk zones** can cut burglary by up to **20%** (Braga et al., 2014).
        - The **“Koper Curve”** indicates that **10–15 minutes** of police presence in a hotspot has the greatest residual deterrent effect.

        ---
        
        ### ⏳ Officer Availability Assumptions

        - Officers are on duty **daily from 06:00 to 22:00** (16 hours).
        - Officers are available for burglary-focused patrols for 2 hours on 4 days per week.
        - Total available patrol window per ward:  
        **16 hours × 7 days = 112 hours per week**

        ---
        
        ### 🧮 How Officer Presence is Calculated

        We align officer availability with our **5-tier risk model**. Based on how many hours are allocated to each ward, we compute the **average number of officers present per hour**, ensuring sufficient coverage for deterrence without over-policing.

        | Risk Tier         | Weekly Patrol Hours | Avg. Officers Active (per hour) |
        |-------------------|---------------------|-------------------------------|
        | 🟥 High Risk       | 800 hrs             | ~7 officers                   |
        | 🟠 Medium-High     | 600 hrs             | ~5 officers                   |
        | 🟡 Medium          | 480 hrs             | ~4 officers                   |
        | 🟢 Medium-Low      | 360 hrs             | ~3 officers                   |
        | 🟩 Low Risk        | 240 hrs             | ~2 officers                   |

        ✅ Even **low-risk wards** maintain a **base presence of 2 officers** per hour — critical for visibility, public confidence, and response capability.

        ---
        
        ### 📆 How Patrols Are Conducted

        - Patrols are broken into short, 15-minute bursts.
        - Officers rotate through **micro-hotspots** within wards.
        - Patrol **times vary across days**, to avoid predictability by potential offenders.
        - **Peak patrol times** include:
        - Weekday late morning & early afternoon (when homes are empty e.g., 10:00–14:00)
        - Evenings after sunset (when visibility is low and burglary risk rises e.g., 17:00–21:00)

        ---
        
        By combining our model with real-world scheduling principles, we aim to research and deliver a safer, fairer, and smarter policing system across London.
        """)



