import pandas as pd
import os
import numpy as np
import traceback


def compute_population_stats(excel_path, output_csv, expanded_years=range(2010, 2026)):
    """
    Compute population statistics from Excel files containing LSOA population data by age and gender.
    Now includes expanded years functionality to generate data for a range of years during CSV creation.

    Args:
        excel_path: Path to the Excel file with population data
        output_csv: Path where the CSV output should be saved
        expanded_years: Range of years to expand the data to (default: 2010-2025)
    """
    xl = pd.ExcelFile(excel_path)
    result_rows = []

    print(f"Found sheets: {xl.sheet_names}")
    processed_sheets = 0

    # Dictionary to track years in the source data
    year_data = {}

    # We'll use a more specific approach based on the debugged structure
    for sheet_name in xl.sheet_names:
        if not sheet_name.startswith("Mid-"):  # skip metadata/notes
            print(f"Skipping sheet: {sheet_name}")
            continue

        print(f"\nProcessing sheet: {sheet_name}")
        processed_sheets += 1

        # Extract year from sheet name (e.g., "Mid-2019 LSOA 2021" -> 2019)
        try:
            year = int(sheet_name.split("-")[1].split(" ")[0])
            print(f"Detected year: {year}")
        except (IndexError, ValueError):
            print(f"Could not extract year from sheet name: {sheet_name}, using sheet name as is")
            year = None

        try:
            # Skip the first 3 rows (0, 1, 2) to get to the data
            df = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=3)
            print(f"DataFrame shape: {df.shape}")

            # Print the first few column names to check structure
            print(f"First few columns: {df.columns[:10].tolist()}")

            # Identify LSOA code column - typically the first or second column
            lsoa_col = None
            # Check first few columns for LSOA codes (starting with E01 or W01)
            for i in range(min(5, len(df.columns))):
                col = df.columns[i]
                sample_vals = df[col].dropna().astype(str).head(10)
                if any(x.startswith(('E01', 'W01')) for x in sample_vals):
                    lsoa_col = col
                    print(f"Found LSOA column at index {i}: {lsoa_col}")
                    break

            if lsoa_col is None:
                print("Could not identify LSOA column, trying column 0 as default")
                lsoa_col = df.columns[0]

            # Column indices for gender-specific data (skip the first few columns which have metadata)
            start_col_idx = 3  # Skip LSOA code and any other metadata columns

            # Calculate midpoint for splitting female/male columns
            total_data_cols = len(df.columns) - start_col_idx
            midpoint = total_data_cols // 2

            # Define female and male column ranges
            female_col_indices = range(start_col_idx, start_col_idx + midpoint)
            male_col_indices = range(start_col_idx + midpoint, len(df.columns))

            # Get the actual columns
            female_cols = [df.columns[i] for i in female_col_indices]
            male_cols = [df.columns[i] for i in male_col_indices]

            # Convert non-numeric columns to numeric
            for col in df.columns[start_col_idx:]:
                if not pd.api.types.is_numeric_dtype(df[col].dtype):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass

            # Define age arrays (0 to N-1) for both genders
            female_ages = list(range(len(female_cols)))
            male_ages = list(range(len(male_cols)))

            # Process data rows
            valid_rows = 0
            sheet_lsoa_data = {}  # Store data for this sheet

            for idx, row in df.iterrows():
                try:
                    lsoa_code = row[lsoa_col]

                    # Skip rows with missing or invalid LSOA code
                    if pd.isna(lsoa_code):
                        continue

                    # Convert to string if it's not already
                    if not isinstance(lsoa_code, str):
                        lsoa_code = str(lsoa_code)

                    # Skip non-LSOA code rows (they typically start with E01 or W01)
                    if not (lsoa_code.startswith('E01') or lsoa_code.startswith('W01')):
                        continue

                    # Female population data
                    female_pops = []
                    for col in female_cols:
                        try:
                            val = pd.to_numeric(row[col], errors='coerce')
                            if pd.notna(val) and val >= 0:
                                female_pops.append(val)
                            else:
                                female_pops.append(0)
                        except:
                            female_pops.append(0)

                    female_total = sum(female_pops)

                    # Female mean age calculation
                    female_weighted_sum = sum(age * pop for age, pop in zip(female_ages, female_pops))
                    female_mean_age = female_weighted_sum / female_total if female_total > 0 else 0

                    # Male population data
                    male_pops = []
                    for col in male_cols:
                        try:
                            val = pd.to_numeric(row[col], errors='coerce')
                            if pd.notna(val) and val >= 0:
                                male_pops.append(val)
                            else:
                                male_pops.append(0)
                        except:
                            male_pops.append(0)

                    male_total = sum(male_pops)

                    # Male mean age calculation
                    male_weighted_sum = sum(age * pop for age, pop in zip(male_ages, male_pops))
                    male_mean_age = male_weighted_sum / male_total if male_total > 0 else 0

                    # Calculate male to female ratio
                    ratio = male_total / female_total if female_total > 0 else None

                    # Calculate overall mean age
                    total_pop = female_total + male_total
                    mean_age = (female_weighted_sum + male_weighted_sum) / total_pop if total_pop > 0 else 0

                    row_data = {
                        "LSOA Code": lsoa_code,
                        "Sheet Name": sheet_name,
                        "Year": year,  # Add the extracted year
                        "Total Female": int(female_total),
                        "Total Male": int(male_total),
                        "Total Population": int(female_total + male_total),
                        "Mean Female Age": round(female_mean_age, 2),
                        "Mean Male Age": round(male_mean_age, 2),
                        "Mean Age": round(mean_age, 2),
                        "Male/Female Ratio": round(ratio, 2) if ratio is not None else None
                    }

                    # Add to results
                    result_rows.append(row_data)

                    # Store data for expansion later
                    if year is not None:
                        if lsoa_code not in year_data:
                            year_data[lsoa_code] = {}
                        year_data[lsoa_code][year] = row_data

                    valid_rows += 1

                    # Report progress occasionally
                    if valid_rows % 500 == 0:
                        print(f"Processed {valid_rows} valid rows so far...")

                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
                    continue

            print(f"Processed {valid_rows} valid rows in sheet {sheet_name}")

        except Exception as e:
            print(f"Error processing sheet {sheet_name}: {e}")
            traceback.print_exc()
            continue

    if not result_rows:
        print("Warning: No data was processed. Please check the Excel file format.")
        return

    # Convert to DataFrame and prepare for expansion
    result_df = pd.DataFrame(result_rows)
    print(f"Raw data rows: {len(result_df)}")

    # Extract unique years from the data
    all_data_years = sorted(result_df['Year'].dropna().unique()) if 'Year' in result_df.columns else []

    if all_data_years:
        print(f"Data contains years: {all_data_years}")

        # Create expanded years data
        expanded_rows = []
        min_year = min(all_data_years) if all_data_years else None
        max_year = max(all_data_years) if all_data_years else None

        if min_year is not None and max_year is not None:
            # Get unique LSOA codes
            unique_lsoa_codes = result_df['LSOA Code'].unique()
            print(
                f"Expanding data for {len(unique_lsoa_codes)} LSOA codes across years {min(expanded_years)}-{max(expanded_years)}")

            # Track data for expansion
            for lsoa_code in unique_lsoa_codes:
                # Get data for this LSOA across all years
                lsoa_df = result_df[result_df['LSOA Code'] == lsoa_code]

                # For each expanded year, find or interpolate data
                for year in expanded_years:
                    if year in all_data_years:
                        # Check if we already have data for this exact year
                        exact_year_data = lsoa_df[lsoa_df['Year'] == year]
                        if not exact_year_data.empty:
                            # Already have this data, will be included in result_df
                            continue

                    # Need to create data for this year
                    if year < min_year:
                        # Use earliest year data
                        source_year = min_year
                        source_data = lsoa_df[lsoa_df['Year'] == source_year]
                    elif year > max_year:
                        # Use latest year data
                        source_year = max_year
                        source_data = lsoa_df[lsoa_df['Year'] == source_year]
                    else:
                        # Interpolate between available years
                        # Find the closest years before and after
                        lower_years = [y for y in all_data_years if y <= year]
                        higher_years = [y for y in all_data_years if y >= year]

                        if lower_years and higher_years:
                            lower_year = max(lower_years)
                            higher_year = min(higher_years)

                            if lower_year == higher_year:
                                # Exact match, use that data
                                source_data = lsoa_df[lsoa_df['Year'] == lower_year]
                            else:
                                # Need to interpolate
                                lower_data = lsoa_df[lsoa_df['Year'] == lower_year].iloc[0].to_dict()
                                higher_data = lsoa_df[lsoa_df['Year'] == higher_year].iloc[0].to_dict()

                                # Linear interpolation factor
                                factor = (year - lower_year) / (higher_year - lower_year)

                                # Create interpolated row
                                new_row = lower_data.copy()
                                new_row['Year'] = year

                                # Interpolate numeric fields
                                for field in ['Total Female', 'Total Male', 'Total Population',
                                              'Mean Female Age', 'Mean Male Age', 'Mean Age', 'Male/Female Ratio']:
                                    if field in lower_data and field in higher_data:
                                        lower_val = lower_data[field]
                                        higher_val = higher_data[field]
                                        if pd.notna(lower_val) and pd.notna(higher_val):
                                            # Linear interpolation
                                            new_row[field] = lower_val + factor * (higher_val - lower_val)
                                            # Round to appropriate precision
                                            if field.startswith('Total'):
                                                new_row[field] = int(round(new_row[field]))
                                            else:
                                                new_row[field] = round(new_row[field], 2)

                                expanded_rows.append(new_row)
                                continue

                        elif lower_years:
                            source_year = max(lower_years)
                            source_data = lsoa_df[lsoa_df['Year'] == source_year]
                        elif higher_years:
                            source_year = min(higher_years)
                            source_data = lsoa_df[lsoa_df['Year'] == source_year]
                        else:
                            # Shouldn't happen, but handle it
                            continue

                    # Only proceed if we found source data
                    if not source_data.empty:
                        # Create new row based on source data
                        new_row = source_data.iloc[0].to_dict()
                        new_row['Year'] = year

                        # Add to expanded rows
                        expanded_rows.append(new_row)

            # Add expanded rows to result
            if expanded_rows:
                expanded_df = pd.DataFrame(expanded_rows)
                print(f"Created {len(expanded_df)} expanded year records")
                result_df = pd.concat([result_df, expanded_df], ignore_index=True)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_csv)
    if output_dir:  # Only try to create dir if there's a path
        os.makedirs(output_dir, exist_ok=True)

    # Sort the dataframe for consistency
    if 'LSOA Code' in result_df.columns and 'Year' in result_df.columns:
        result_df = result_df.sort_values(['LSOA Code', 'Year'])

    result_df.to_csv(output_csv, index=False)
    print(f"\nSaved population stats to: {output_csv}")
    print(f"Total records in output: {len(result_df)}")
    print(f"Processed {processed_sheets} data sheets")

    # Print some summary statistics
    if not result_df.empty:
        print("\nSummary statistics:")
        # Group by year if available, otherwise by sheet name
        if 'Year' in result_df.columns:
            print(f"Total populations by year:")
            year_totals = result_df.groupby('Year')['Total Population'].sum()
            for year, total in year_totals.items():
                print(f"  {year}: {int(total):,}")
        else:
            print(f"Total populations by sheet:")
            sheet_totals = result_df.groupby('Sheet Name')['Total Population'].sum()
            for sheet, total in sheet_totals.items():
                print(f"  {sheet}: {int(total):,}")

        if 'Male/Female Ratio' in result_df.columns:
            print(f"\nAverage male/female ratio: {result_df['Male/Female Ratio'].dropna().mean():.2f}")

        if 'Mean Age' in result_df.columns:
            print(f"Average mean age: {result_df['Mean Age'].dropna().mean():.2f}")

    # Return a preview of the data
    return result_df.head()


# Example usage
if __name__ == "__main__":
    try:
        # Wrap in try/except to get full traceback
        try:
            result = compute_population_stats("data/sapelsoasyoa20192022.xlsx", "data/population_summary.csv")
            if result is not None and not result.empty:
                print("\nData preview:")
                print(result)
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
    except Exception as e:
        print(f"Critical error: {e}")