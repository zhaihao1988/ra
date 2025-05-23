import pandas as pd
import numpy as np
from scipy.stats import trim_mean  # Used if trimming strategy involves it


def extract_triangle_from_df(df, triangle_type_keyword, specific_triangle_keyword):
    """
    Extracts a specific runoff triangle from the loaded DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame loaded from the CSV or Excel sheet.
        triangle_type_keyword (str): Keyword like "再保前" or "再保后".
        specific_triangle_keyword (str): Keyword for the specific triangle, e.g., "累计已决三角形".

    Returns:
        pd.DataFrame: The extracted runoff triangle with accident years as index
                      and development periods as columns. Returns None if not found.
    """
    try:
        # Find the row index for the triangle type (e.g., "再保前")
        # Search for the specific triangle keyword starting from the type_row_index
        # We are looking for the first block of data usually.

        # Locate "累计已决三角形"
        paid_keyword_rows = df[
            df.apply(lambda r: r.astype(str).str.contains(specific_triangle_keyword).any(), axis=1)].index
        if not paid_keyword_rows.any():
            print(f"Keyword '{specific_triangle_keyword}' not found.")
            return None

        accident_month_loc = None
        # Try to find the "事故月份" header, usually below "累计已决三角形"
        # Search for "事故月份" which serves as a marker for the start of the data table
        # We are interested in the first block, so look at early columns mainly for these headers
        for r_idx in paid_keyword_rows:  # Iterate through rows where specific_triangle_keyword was found
            # Check if triangle_type_keyword is nearby (e.g., in the same column, a few rows above, or in a merged cell context)
            # This part is tricky with merged cells. We rely on finding "事故月份" below it.
            potential_accident_month_row = r_idx + 1
            if potential_accident_month_row < df.shape[0]:
                # Search for "事故月份" in the row below specific_triangle_keyword
                for c_idx in range(min(5, df.shape[1])):  # Search in first few columns
                    if isinstance(df.iloc[potential_accident_month_row, c_idx], str) and "事故月份" in df.iloc[
                        potential_accident_month_row, c_idx]:
                        # Heuristic: Check if "再保前" is above this structure in the same column block
                        # This is a simplified check. A more robust check would trace back merged cells.
                        is_pre_reinsurance_block = False
                        for prev_r_idx in range(r_idx - 3, r_idx + 1):  # Check few rows above
                            if prev_r_idx >= 0 and isinstance(df.iloc[prev_r_idx, c_idx],
                                                              str) and triangle_type_keyword in df.iloc[
                                prev_r_idx, c_idx]:
                                is_pre_reinsurance_block = True
                                break
                        if is_pre_reinsurance_block:
                            accident_month_loc = (potential_accident_month_row, c_idx)
                            break
            if accident_month_loc:
                break

        if not accident_month_loc:  # Fallback to more direct search if above fails
            # This part is similar to the original CSV search logic
            for r_idx_main in range(df.shape[0]):
                for c_idx_main in range(min(5, df.shape[1])):
                    if isinstance(df.iloc[r_idx_main, c_idx_main], str) and "事故月份" in df.iloc[
                        r_idx_main, c_idx_main]:
                        if r_idx_main > 0 and isinstance(df.iloc[r_idx_main - 1, c_idx_main],
                                                         str) and specific_triangle_keyword in df.iloc[
                            r_idx_main - 1, c_idx_main]:
                            if r_idx_main > 1 and isinstance(df.iloc[r_idx_main - 2, c_idx_main],
                                                             str) and triangle_type_keyword in df.iloc[
                                r_idx_main - 2, c_idx_main]:
                                accident_month_loc = (r_idx_main, c_idx_main)
                                break
                if accident_month_loc:
                    break

        if not accident_month_loc:
            print(
                f"Could not robustly locate '事故月份' under '{triangle_type_keyword}' and '{specific_triangle_keyword}'. Please check sheet structure.")
            return None

        dev_period_header_row = accident_month_loc[0] + 1
        data_start_row = accident_month_loc[0] + 2
        header_col_idx = accident_month_loc[1]  # Column of "事故月份"

        # Extract development period headers
        dev_periods = []
        if dev_period_header_row < df.shape[0]:
            for c in range(header_col_idx + 1, df.shape[1]):
                val = df.iloc[dev_period_header_row, c]
                try:
                    float_val = float(val)
                    if float_val.is_integer():
                        dev_periods.append(int(float_val))
                    else:
                        dev_periods.append(int(float_val))
                except (ValueError, TypeError):
                    break

        num_dev_periods = len(dev_periods)
        if num_dev_periods == 0:
            print("No development periods found under '事故月份'.")
            return None

        # Extract data rows
        triangle_data = []
        accident_years = []
        if data_start_row < df.shape[0]:
            for r in range(data_start_row, df.shape[0]):
                acc_year_val = df.iloc[r, header_col_idx]
                if pd.isna(acc_year_val) or str(acc_year_val).strip() == "":
                    # Check for end of block based on "合计" or similar
                    potential_sum_label_col = header_col_idx - 1 if header_col_idx > 0 else header_col_idx
                    if isinstance(df.iloc[r, potential_sum_label_col], str) and (
                            "合计" in df.iloc[r, potential_sum_label_col] or "Total" in df.iloc[
                        r, potential_sum_label_col]):
                        break  # End of triangle data
                    # If it's just an empty line within data, might indicate end
                    if df.iloc[r, header_col_idx: header_col_idx + 1 + num_dev_periods].isnull().all():
                        if len(accident_years) > 0:  # If we already have data, an all-NaN row likely ends the block
                            break
                        else:  # If no data yet, skip this empty row
                            continue

                try:
                    str_acc_year = str(acc_year_val).strip()
                    # A simple check for a valid accident period (e.g., "201606", "2016", or even just numeric)
                    # If the cell to its right is not numeric, it's likely not a data row for the triangle
                    first_data_cell_val = df.iloc[r, header_col_idx + 1]
                    is_numeric_check = str(first_data_cell_val).replace('.', '', 1).isdigit() or pd.isna(
                        first_data_cell_val)

                    if not (str_acc_year and (
                            str_acc_year[0].isdigit() or str_acc_year.startswith('AY')) and is_numeric_check):
                        if len(accident_years) > 0:
                            break
                        else:
                            continue

                    accident_years.append(str_acc_year)
                    row_data = df.iloc[r, header_col_idx + 1: header_col_idx + 1 + num_dev_periods].tolist()
                    numeric_row_data = [pd.to_numeric(val, errors='coerce') for val in row_data]
                    triangle_data.append(numeric_row_data)
                except Exception as e:
                    if len(accident_years) > 0 and str(accident_years[-1]) == str_acc_year:
                        accident_years.pop()
                    break

        if not triangle_data:
            print("No data extracted for the triangle.")
            return None

        extracted_df = pd.DataFrame(triangle_data, index=pd.Index(accident_years, name="AccidentYear"),
                                    columns=dev_periods)
        extracted_df = extracted_df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        return extracted_df

    except Exception as e:
        print(f"An error occurred during triangle extraction: {e}")
        import traceback
        traceback.print_exc()
        return None


def cumulative_to_incremental(cumulative_triangle):
    """Converts a cumulative triangle to an incremental triangle."""
    incremental_triangle = cumulative_triangle.copy()
    # First column is already incremental (it's the first payment)
    for col_idx in range(1, incremental_triangle.shape[1]):
        incremental_triangle.iloc[:, col_idx] = cumulative_triangle.iloc[:, col_idx] - cumulative_triangle.iloc[:,
                                                                                       col_idx - 1]
    return incremental_triangle


def calculate_weighted_ldfs(cumulative_triangle, trim_extremes=True):
    """
    Calculates volume-weighted LDFs, optionally trimming max/min individual LDFs.
    """
    n_rows, n_cols = cumulative_triangle.shape
    ldfs_final_selection = []

    for j in range(n_cols - 1):
        col1_vals_all = cumulative_triangle.iloc[:, j]
        col2_vals_all = cumulative_triangle.iloc[:, j + 1]

        current_period_ldfs_data = []  # Store (ldf, weight) tuples

        for i in range(n_rows - (j + 1)):
            val1 = col1_vals_all.iloc[i]
            val2 = col2_vals_all.iloc[i]

            if pd.notna(val1) and pd.notna(val2) and val1 != 0:
                current_period_ldfs_data.append({'ldf': val2 / val1, 'weight': val1})

        if not current_period_ldfs_data:
            ldfs_final_selection.append(1.0)  # Default if no data
            continue

        ldf_values = [d['ldf'] for d in current_period_ldfs_data]
        weights = [d['weight'] for d in current_period_ldfs_data]

        if trim_extremes and len(ldf_values) >= 5:  # Need at least 5 points to trim 2 and average remaining 3
            # Sort by LDF value to identify extremes
            sorted_indices = np.argsort(ldf_values)
            # Exclude min and max LDF by using indices of sorted values
            indices_to_keep = sorted_indices[1:-1]

            trimmed_ldf_values = [ldf_values[i] for i in indices_to_keep]
            trimmed_weights = [weights[i] for i in indices_to_keep]

            if not trimmed_ldf_values:  # Should not happen if len >=5
                avg_ldf = 1.0
            else:
                sum_trimmed_weights = sum(trimmed_weights)
                if sum_trimmed_weights == 0:
                    avg_ldf = np.mean(trimmed_ldf_values) if trimmed_ldf_values else 1.0
                else:
                    # Weighted average of (LDF * weight) / sum(weight) which is sum(LDF*weight)/sum(weight)
                    # = sum( (val2/val1) * val1 ) / sum(val1) = sum(val2_trimmed) / sum(val1_trimmed)
                    # Here val1_trimmed are the weights of the trimmed LDFs
                    numerator = sum(ldf * w for ldf, w in zip(trimmed_ldf_values, trimmed_weights))
                    avg_ldf = numerator / sum_trimmed_weights
        else:  # Not enough to trim or trimming disabled, use all data points for standard volume weighted
            total_weight = sum(weights)
            if total_weight == 0:
                avg_ldf = np.mean(ldf_values) if ldf_values else 1.0
            else:
                # sum ( (val2/val1) * val1 ) / sum(val1) = sum(val2) / sum(val1)
                numerator = sum(ldf * w for ldf, w in zip(ldf_values, weights))  # sum of val2 equivalents
                avg_ldf = numerator / total_weight

        ldfs_final_selection.append(avg_ldf if pd.notna(avg_ldf) else 1.0)

    # Tail factor
    valid_ldfs_for_tail = [ldf for ldf in ldfs_final_selection[-3:] if
                           pd.notna(ldf) and ldf > 0.1]  # Avoid using tiny LDFs from sparse data
    if len(valid_ldfs_for_tail) > 0:
        tail_factor = np.mean(valid_ldfs_for_tail)
    else:
        tail_factor = 1.0

    tail_factor = max(1.0, tail_factor)  # Ensure tail is at least 1.0
    ldfs_final_selection.append(tail_factor)

    return np.array(ldfs_final_selection)


def project_triangle(cumulative_triangle, ldfs):
    """Projects a cumulative triangle to ultimate using LDFs."""
    projected_triangle_df = cumulative_triangle.copy().astype(float)
    n_rows, n_cols_orig = projected_triangle_df.shape

    # The number of development periods LDFs cover (excluding tail)
    num_dev_ldfs = len(ldfs) - 1

    # Add an 'Ultimate' column if it doesn't exist
    if 'Ultimate' not in projected_triangle_df.columns:
        projected_triangle_df['Ultimate'] = np.nan

    for i in range(n_rows):
        current_cumulative = np.nan
        last_observed_col_idx = -1

        # Find the last observed cumulative payment for this accident year
        for j in range(n_cols_orig):
            if pd.notna(projected_triangle_df.iloc[i, j]):
                current_cumulative = projected_triangle_df.iloc[i, j]
                last_observed_col_idx = j
            else:  # First NaN means this is where projection starts for this row
                # If no data at all for the row, fill with 0 and its ultimate is 0
                if last_observed_col_idx == -1 and j == 0:
                    projected_triangle_df.iloc[i, :n_cols_orig] = 0
                    current_cumulative = 0
                    last_observed_col_idx = n_cols_orig - 1  # Treat as fully developed at 0
                break

        if pd.isna(current_cumulative):  # Row was all NaNs or became NaN
            projected_triangle_df.iloc[i, :] = 0  # Fill entire row (incl ultimate) with 0
            continue

        # Project from last_observed_col_idx + 1 onwards using LDFs
        # k is the index of the LDF to apply (0-indexed)
        # LDF[k] takes you from development period k to k+1
        # So, if last_observed_col_idx is j, the next value (at j+1) uses LDF[j]

        temp_cumulative_val = current_cumulative
        for k_ldf_idx in range(last_observed_col_idx + 1, num_dev_ldfs + 1):  # +1 to include the tail LDF
            ldf_to_apply = ldfs[k_ldf_idx] if k_ldf_idx < len(ldfs) else 1.0  # Safety for ldf index
            ldf_to_apply = max(1.0, ldf_to_apply) if pd.notna(ldf_to_apply) else 1.0  # Ensure LDF is >=1 and not NaN

            temp_cumulative_val *= ldf_to_apply

            # If k_ldf_idx corresponds to a column within the original triangle's width
            if k_ldf_idx < n_cols_orig:
                projected_triangle_df.iloc[i, k_ldf_idx] = temp_cumulative_val
            # If k_ldf_idx is for the tail factor application (num_dev_ldfs is the index of tail factor)
            # The LDF for col (n_cols_orig-1) to Ultimate is ldfs[n_cols_orig-1], which is the tail if n_cols_orig = num_dev_ldfs
            # This loop structure will apply all LDFs including the tail. The final temp_cumulative_val is the ultimate.

        projected_triangle_df.loc[projected_triangle_df.index[i], 'Ultimate'] = temp_cumulative_val

    return projected_triangle_df


def calculate_ra(triangle_df, n_simulations=1000, ra_percentile=0.75):
    """
    Calculates Risk Adjustment (RA) using chain ladder and bootstrapping.
    """
    if triangle_df is None or triangle_df.empty:
        print("Input triangle is empty or None. Cannot calculate RA.")
        return None, None, None, None

    cumulative_actual_triangle = triangle_df.apply(pd.to_numeric, errors='coerce').astype(float)
    if cumulative_actual_triangle.isnull().all().all():  # Check if all values became NaN
        print("Error: Triangle is all NaN after numeric conversion. Check data.")
        return None, None, None, None

    selected_ldfs = calculate_weighted_ldfs(cumulative_actual_triangle, trim_extremes=True)
    print(f"Selected LDFs (incl. tail): {selected_ldfs}")
    if np.isnan(selected_ldfs).all() or len(selected_ldfs) == 0:
        print("Error: All LDFs are NaN or LDF array is empty. Check data quality or LDF calculation.")
        return None, None, None, None
    selected_ldfs = np.nan_to_num(np.array(selected_ldfs), nan=1.0)
    selected_ldfs[selected_ldfs < 1.0] = 1.0

    fitted_cumulative_triangle_full = project_triangle(cumulative_actual_triangle.copy(), selected_ldfs)
    fitted_cumulative_triangle_upper = fitted_cumulative_triangle_full.iloc[:, :cumulative_actual_triangle.shape[1]]

    incremental_actual_triangle = cumulative_to_incremental(cumulative_actual_triangle)
    incremental_fitted_triangle = cumulative_to_incremental(fitted_cumulative_triangle_upper)

    residuals_inc = incremental_actual_triangle - incremental_fitted_triangle
    sqrt_fitted_inc_clipped = np.sqrt(
        np.maximum(incremental_fitted_triangle, 1e-9))  # Increased clipping minimum slightly

    standardized_residuals_matrix = residuals_inc / sqrt_fitted_inc_clipped

    pool_of_residuals = []
    for r in range(cumulative_actual_triangle.shape[0]):
        for c in range(cumulative_actual_triangle.shape[1]):
            if pd.notna(cumulative_actual_triangle.iloc[r, c]):  # Only from observed part
                # For incremental, if c=0, it's the first payment. If c>0, actual payment was made.
                # We need a corresponding fitted incremental value.
                if pd.notna(incremental_fitted_triangle.iloc[r, c]) and pd.notna(
                        standardized_residuals_matrix.iloc[r, c]):
                    pool_of_residuals.append(standardized_residuals_matrix.iloc[r, c])

    if not pool_of_residuals:
        print("Error: No residuals could be calculated for the pool. Check data or LDFs.")
        return None, None, None, None
    pool_of_residuals = np.array(pool_of_residuals)
    pool_of_residuals = pool_of_residuals[np.isfinite(pool_of_residuals)]  # Clean out any NaN/inf
    if len(pool_of_residuals) == 0:
        print("Error: Residual pool is empty after removing non-finite values.")
        return None, None, None, None

    current_latest_paid = 0
    for r_idx in range(cumulative_actual_triangle.shape[0]):
        row_data = cumulative_actual_triangle.iloc[r_idx, :].dropna()
        if not row_data.empty:
            current_latest_paid += row_data.iloc[-1]
    if pd.isna(current_latest_paid): current_latest_paid = 0  # Safety

    simulated_ibnrs = []
    for sim_num in range(n_simulations):
        sim_incremental_triangle = incremental_fitted_triangle.copy()

        # Create mask for where original data existed to apply residuals
        observed_mask = cumulative_actual_triangle.notna()

        # Sample residuals only for the cells that were originally observed
        num_residuals_needed = observed_mask.sum().sum()
        if num_residuals_needed > 0:
            sampled_std_residuals_flat = np.random.choice(pool_of_residuals, size=num_residuals_needed, replace=True)
        else:  # No observed data to bootstrap on (highly unlikely for a valid triangle)
            sampled_std_residuals_flat = np.array([])

        k = 0
        for r in range(sim_incremental_triangle.shape[0]):
            for c in range(sim_incremental_triangle.shape[1]):
                if observed_mask.iloc[r, c]:  # If this cell was observed in original triangle
                    if k < len(sampled_std_residuals_flat):
                        sampled_std_residual = sampled_std_residuals_flat[k]
                        k += 1

                        fitted_inc_val = incremental_fitted_triangle.iloc[r, c]
                        if pd.isna(fitted_inc_val): fitted_inc_val = 0  # if fitted is NaN use 0

                        # De-standardize carefully, ensuring sqrt_val is from positive fitted_inc_val
                        sqrt_val_for_destd = np.sqrt(max(fitted_inc_val, 1e-9))
                        sim_payment_inc = fitted_inc_val + sampled_std_residual * sqrt_val_for_destd
                        sim_incremental_triangle.iloc[r, c] = max(0, sim_payment_inc)
                    # else: # Not enough residuals sampled (should not happen if size matches)
                    # else: # Not observed, keep the fitted incremental (which might be NaN if outside upper triangle)
                    # These will be filled by chain ladder projection after converting to cumulative
                    pass

        sim_cumulative_triangle = sim_incremental_triangle.cumsum(axis=1)
        # Fill NaNs that might result from cumsum if initial values were NaN (e.g. future accident years)
        # This is important if sim_incremental_triangle has NaNs for unobserved parts.
        # The project_triangle function should handle rows that are all NaN or start with NaN.
        # However, let's ensure the input to project_triangle is clean for non-bootstrapped cells.
        # The chain ladder `project_triangle` will fill in NaNs in the lower right.

        sim_ultimate_triangle_full = project_triangle(sim_cumulative_triangle.copy(), selected_ldfs)

        sim_ultimate_claims_sum = sim_ultimate_triangle_full['Ultimate'].sum()
        if pd.isna(sim_ultimate_claims_sum): sim_ultimate_claims_sum = 0  # Safety

        ibnr = sim_ultimate_claims_sum - current_latest_paid
        simulated_ibnrs.append(max(0, ibnr))

    simulated_ibnrs.sort()
    ra_index = int(n_simulations * ra_percentile) - 1
    ra_index = min(max(0, ra_index), n_simulations - 1)

    ra_value = simulated_ibnrs[ra_index] if simulated_ibnrs else np.nan
    mean_ibnr = np.mean(simulated_ibnrs) if simulated_ibnrs else np.nan

    return mean_ibnr, ra_value, simulated_ibnrs, fitted_cumulative_triangle_full


# Main execution part (MODIFIED FOR EXCEL INPUT)
if __name__ == '__main__':
    EXCEL_FILE_PATH = "202412三角形.xlsx"
    SHEET_NAME = "商业三者"  # The specific sheet name

    try:
        # Read the specific sheet from the Excel file
        df_sheet_data = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME, header=None, dtype=str)
        print(f"Successfully loaded sheet '{SHEET_NAME}' from '{EXCEL_FILE_PATH}'")

        commercial_pre_re_paid_triangle = extract_triangle_from_df(df_sheet_data,
                                                                   "再保前",
                                                                   "累计已决三角形")

        if commercial_pre_re_paid_triangle is not None and not commercial_pre_re_paid_triangle.empty:
            print("\nSuccessfully extracted '商业三者再保前已决' triangle from the sheet:")
            # print(commercial_pre_re_paid_triangle.head().to_string()) # Print more for debug if needed

            # Convert to numeric AFTER extraction is complete.
            # The extractor should return a DataFrame of strings that look like numbers.
            for col in commercial_pre_re_paid_triangle.columns:
                commercial_pre_re_paid_triangle[col] = pd.to_numeric(commercial_pre_re_paid_triangle[col],
                                                                     errors='coerce')

            commercial_pre_re_paid_triangle = commercial_pre_re_paid_triangle.dropna(axis=0, how='all').dropna(axis=1,
                                                                                                               how='all')

            if commercial_pre_re_paid_triangle.empty:
                print("Triangle became empty after numeric conversion and NaN drop. Check data within the sheet.")
            else:
                print("\nCleaned '商业三者再保前已决' triangle for processing (first 5 rows):")
                print(commercial_pre_re_paid_triangle.head().to_string())
                print(f"Triangle shape: {commercial_pre_re_paid_triangle.shape}")

                print("\nStarting RA calculation...")
                mean_ibnr, ra_at_75, all_ibnrs, fitted_triangle = calculate_ra(commercial_pre_re_paid_triangle,
                                                                               n_simulations=1000,  # Default is 1000
                                                                               ra_percentile=0.75)
                if mean_ibnr is not None and ra_at_75 is not None:
                    print(f"\n--- Results for '商业三者再保前已决' ---")
                    print(f"Estimated Mean IBNR: {mean_ibnr:,.2f}")
                    print(f"Risk Adjustment (RA) at 75% percentile: {ra_at_75:,.2f}")

                    print("\nFitted Cumulative Triangle (including Ultimate) (first 5 rows):")
                    if fitted_triangle is not None:
                        print(fitted_triangle.head().to_string(
                            formatters={c: '{:,.0f}'.format for c in fitted_triangle.columns if
                                        fitted_triangle[c].dtype in [np.int64, np.float64]}))
                    else:
                        print("Fitted triangle is None.")
                else:
                    print("\nRA calculation failed or returned None.")
        else:
            print(
                f"\nCould not extract the '商业三者再保前已决' triangle from sheet '{SHEET_NAME}'. Please check the sheet content and keywords.")

    except FileNotFoundError:
        print(
            f"Error: Excel file '{EXCEL_FILE_PATH}' not found. Ensure it's in the same directory as the script, or provide the full path.")
    except ValueError as ve:
        if "Worksheet" in str(ve) and SHEET_NAME in str(ve):
            print(
                f"Error: Sheet named '{SHEET_NAME}' not found in '{EXCEL_FILE_PATH}'. Please check the sheet name (it's case-sensitive).")
        else:
            print(f"A ValueError occurred: {ve}")
            import traceback

            traceback.print_exc()
    except Exception as e:
        # If it's an import error for openpyxl, guide the user.
        if "openpyxl" in str(e).lower():
            print(f"An error occurred: {e}")
            print(
                "This might be due to the 'openpyxl' library not being installed. Please install it by running: pip install openpyxl")
        else:
            print(f"An unexpected error occurred: {e}")
            import traceback

            traceback.print_exc()