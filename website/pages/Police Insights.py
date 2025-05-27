import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import numpy as np
from shapely.geometry import Point

# ✅ MUST be first Streamlit command
st.set_page_config(page_title="Police Insights", layout="centered")


# Password that police has access to
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

# --- Load predictions CSV ---
@st.cache_data
def load_predictions():
    try:
        df = pd.read_csv("../data/march_2025_predictions_lsoa.csv")

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
            df['LSOA code'] = df['LSOA code'].astype('category')
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

st.sidebar.header("Settings")

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
    """Load and preprocess LSOA boundaries"""
    try:
        shp_lsoa = "../boundaries/LSOA_2011_London_gen_MHW.shp"
        gdf = gpd.read_file(shp_lsoa)
        gdf['geometry'] = gdf.geometry.simplify(tolerance=0.005, preserve_topology=True)
        gdf = gdf.to_crs(epsg=4326)
        rename_dict = {}
        if 'LSOA11CD' in gdf.columns:
            rename_dict['LSOA11CD'] = 'LSOA code'
        elif 'lsoa11cd' in gdf.columns:
            rename_dict['lsoa11cd'] = 'LSOA code'

        if 'LSOA11NM' in gdf.columns:
            rename_dict['LSOA11NM'] = 'LSOA name'
        elif 'lsoa11nm' in gdf.columns:
            rename_dict['lsoa11nm'] = 'LSOA name'

        if rename_dict:
            gdf = gdf.rename(columns=rename_dict)

        return gdf
    except Exception as e:
        st.error(f"Error reading LSOA shapefile: {e}")
        return None


@st.cache_data
def load_ward_boundaries():
    """Load and preprocess Ward boundaries"""
    try:
        shp_ward = "../boundaries/London_Ward_CityMerged.shp"
        gdf = gpd.read_file(shp_ward)
        gdf['geometry'] = gdf.geometry.simplify(tolerance=0.01, preserve_topology=True)
        gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception as e:
        st.error(f"Error reading ward shapefile: {e}")
        return None


@st.cache_data
def load_lsoa_ward_mapping():
    """Load LSOA to Ward mapping"""
    try:
        map_excel = "../boundaries/LSOA11_WD21_LAD21_EW_LU_V2.xlsx"
        df_map = pd.read_excel(map_excel, dtype=str)

        lsoa_col = None
        for col in ['LSOA11CD', 'lsoa11cd', 'LSOA_CODE']:
            if col in df_map.columns:
                lsoa_col = col
                break

        if lsoa_col is None:
            st.error("Cannot find LSOA code column in mapping file")
            return None

        rename_dict = {lsoa_col: 'LSOA code'}

        ward_code_cols = ['WD21CD', 'wd21cd', 'WARD_CODE']
        ward_name_cols = ['WD21NM', 'wd21nm', 'WARD_NAME']

        for col in ward_code_cols:
            if col in df_map.columns:
                rename_dict[col] = 'Ward code'
                break

        for col in ward_name_cols:
            if col in df_map.columns:
                rename_dict[col] = 'Ward name'
                break

        df_map = df_map.rename(columns=rename_dict)

        required_cols = ['LSOA code', 'Ward code', 'Ward name']
        available_cols = [col for col in required_cols if col in df_map.columns]

        if len(available_cols) != len(required_cols):
            st.error(f"Ward mapping missing columns. Available: {available_cols}")
            return None

        return df_map[required_cols].dropna()

    except Exception as e:
        st.error(f"Error reading mapping: {e}")
        return None

if 'clicked_ward' not in st.session_state:
    st.session_state.clicked_ward = None

try:
    if mode == "LSOA":
        gdf_base = load_lsoa_boundaries()
        if gdf_base is None:
            st.stop()

        if 'LSOA code' not in gdf_base.columns:
            st.error("LSOA shapefile missing LSOA code field")
            st.stop()

        gdf = gdf_base.merge(df_filt, on='LSOA code', how='left')

        if 'Burglary_Probability' not in df_filt.columns:
            st.error("Burglary_Probability column not found in data")
            st.stop()

        display_field = 'Burglary_Probability'
        legend_label = 'Burglary Probability'
        name_field = 'LSOA name' if 'LSOA name' in gdf.columns else 'LSOA code'
        code_field = 'LSOA code'

    else:

        df_map = load_lsoa_ward_mapping()
        gdf_base = load_ward_boundaries()

        if df_map is None or gdf_base is None:
            st.error("Cannot load ward mapping or boundaries")
            st.stop()

        code_options = [c for c in gdf_base.columns if gdf_base[c].dtype == 'object']

        default_code_idx = 0
        for i, col in enumerate(code_options):
            if any(keyword in col.upper() for keyword in ['GSS', 'CODE', 'CD']):
                default_code_idx = i
                break

        code_field_orig = code_options[default_code_idx]

        name_options = [c for c in gdf_base.columns if gdf_base[c].dtype == 'object' and c != code_field_orig]
        default_name_idx = 0
        for i, col in enumerate(name_options):
            if any(keyword in col.upper() for keyword in ['NAME', 'NM']):
                default_name_idx = i
                break

        name_field_orig = name_options[default_name_idx]

        df_join = pd.merge(df_filt, df_map, on='LSOA code', how='inner')

        if df_join.empty:
            st.error("No matching data found between predictions and ward mapping")
            st.stop()
        if 'Predicted_Count' not in df_join.columns:
            st.error("Predicted_Count column not found for ward aggregation")
            st.stop()

        ward_agg = df_join.groupby(['Ward code', 'Ward name'], as_index=False)['Predicted_Count'].sum()

        gdf = gdf_base.rename(columns={
            code_field_orig: 'Ward code',
            name_field_orig: 'Ward name'
        })
        gdf['Ward code'] = gdf['Ward code'].astype(str).str.strip()
        gdf['Ward name'] = gdf['Ward name'].astype(str).str.strip()
        ward_agg['Ward code'] = ward_agg['Ward code'].astype(str).str.strip()
        ward_agg['Ward name'] = ward_agg['Ward name'].astype(str).str.strip()
        gdf = gdf.merge(ward_agg, on='Ward code', how='left')
        if gdf['Predicted_Count'].isna().all():
            gdf = gdf.drop(columns=['Predicted_Count'], errors='ignore')
            ward_agg_code_only = df_join.groupby('Ward code', as_index=False)['Predicted_Count'].sum()
            ward_agg_code_only['Ward code'] = ward_agg_code_only['Ward code'].astype(str).str.strip()
            gdf = gdf.merge(ward_agg_code_only, on='Ward code', how='left')
        display_field = 'Predicted_Count'
        legend_label = 'Expected Burglary Count'
        code_field = 'Ward code'
        name_field = 'Ward name'

except Exception as e:
    st.error(f"Error in data preparation: {e}")
    st.stop()

gdf = gdf[gdf.geometry.notnull()]
st.header(f"{mode} Map: {selected_month}/{selected_year}")

if gdf.empty:
    st.error("No geometries available for this view and filters.")
    st.stop()

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
        """Optimized style function"""
        try:
            value = feature['properties'].get(display_field)

            if display_field == 'Risk_Level':
                risk_colors = {'Low': '#00ff00', 'Medium': '#ffaa00', 'High': '#ff0000'}
                color = risk_colors.get(str(value), '#888888')
            else:
                if pd.isna(value) or value is None:
                    color = '#888888'
                else:
                    try:
                        value = float(value)
                        if not hasattr(style_function, '_min_max_computed'):
                            valid_vals = gdf[display_field].dropna()
                            if len(valid_vals) > 0:
                                style_function._min_val = valid_vals.min()
                                style_function._max_val = valid_vals.max()
                            else:
                                style_function._min_val = 0
                                style_function._max_val = 1
                            style_function._min_max_computed = True

                        if style_function._max_val > style_function._min_val:
                            pct = (value - style_function._min_val) / (
                                    style_function._max_val - style_function._min_val)
                            pct = max(0, min(1, pct))
                        else:
                            pct = 0

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
        except:
            return {'fillColor': '#888888', 'color': 'white', 'weight': 0.5, 'fillOpacity': 0.7}
    if name_field in gdf.columns:
        tooltip_fields = [name_field, display_field]
        tooltip_aliases = [mode, legend_label]
    else:
        tooltip_fields = [display_field]
        tooltip_aliases = [legend_label]

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
    map_data = st_folium(m, width=700, height=500, returned_objects=["last_clicked"])
    clicked = map_data.get('last_clicked')
    if clicked:
        try:
            click_point = Point(clicked['lng'], clicked['lat'])
            selected_area = gdf[gdf.geometry.contains(click_point)]

            if not selected_area.empty:
                row = selected_area.iloc[0]
                area_name = row.get(name_field, 'Unknown')
                area_code = row.get(code_field, 'Unknown')
                if mode == "Ward":
                    st.session_state.clicked_ward = {
                        'code': area_code,
                        'name': area_name
                    }

                st.subheader(f"Selected {mode}: {area_name}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Code", area_code)
                with col2:
                    if display_field in row.index and pd.notna(row[display_field]):
                        st.metric(legend_label,
                                  f"{row[display_field]:.2f}" if isinstance(row[display_field], (int, float)) else str(
                                      row[display_field]))

        except Exception as e:
            st.error(f"Error processing map click: {e}")

except Exception as e:
    st.error(f"Error rendering map: {e}")
st.subheader("Data Table")

try:
    if mode == "LSOA" or st.session_state.clicked_ward is None:
        display_cols = [col for col in [code_field, name_field, display_field] if col in gdf.columns]
        df_display = gdf[display_cols].copy()

        if mode == "LSOA" and display_field == 'Risk_Level':
            risk_counts = df_display[display_field].value_counts()
            st.write("**Risk Level Summary:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Low Risk", risk_counts.get('Low', 0))
            with col2:
                st.metric("Medium Risk", risk_counts.get('Medium', 0))
            with col3:
                st.metric("High Risk", risk_counts.get('High', 0))

        st.dataframe(df_display, use_container_width=True)

    else:
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
                        display_cols = ['LSOA code']
                        if 'LSOA name' in gdf.columns:
                            lsoa_names = gdf[['LSOA code', 'LSOA name']].drop_duplicates()
                            top_5 = top_5.merge(lsoa_names, on='LSOA code', how='left')
                            display_cols.append('LSOA name')

                        if 'Burglary_Probability' in top_5.columns:
                            display_cols.append('Burglary_Probability')
                        if 'Predicted_Count' in top_5.columns:
                            display_cols.append('Predicted_Count')
                        if 'Risk_Level' in top_5.columns:
                            display_cols.append('Risk_Level')
                        display_cols = [col for col in display_cols if col in top_5.columns]

                        st.dataframe(top_5[display_cols].reset_index(drop=True), use_container_width=True)
                    else:
                        st.write("No risk data available for this ward's LSOAs")
                else:
                    st.write("No LSOA data found for this ward")
            else:
                st.write("No LSOAs found for this ward")
        else:
            st.write("Ward mapping not available")
    if 'df_display' in locals():
        csv_data = df_display.to_csv(index=False)
        st.download_button(
            label=f"Download {mode} Data",
            data=csv_data,
            file_name=f"{mode.lower()}_data_{selected_month}_{selected_year}.csv",
            mime="text/csv"
        )

except Exception as e:
    st.error(f"Error displaying data table: {e}")