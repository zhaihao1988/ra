import pandas as pd
import numpy as np
from scipy.stats import trim_mean  # Used if trimming strategy involves it
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For a nicer KDE plot


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
            potential_accident_month_row = r_idx + 1
            if potential_accident_month_row < df.shape[0]:
                # Search for "事故月份" in the row below specific_triangle_keyword
                for c_idx in range(min(5, df.shape[1])):  # Search in first few columns
                    if isinstance(df.iloc[potential_accident_month_row, c_idx], str) and "事故月份" in df.iloc[
                        potential_accident_month_row, c_idx]:
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

        dev_periods = []
        if dev_period_header_row < df.shape[0]:
            for c in range(header_col_idx + 1, df.shape[1]):
                val = df.iloc[dev_period_header_row, c]
                try:
                    float_val = float(val)
                    if float_val.is_integer():
                        dev_periods.append(int(float_val))
                    else:
                        dev_periods.append(int(float_val))  # Keep as int if possible
                except (ValueError, TypeError):
                    break

        num_dev_periods = len(dev_periods)
        if num_dev_periods == 0:
            print("No development periods found under '事故月份'.")
            return None

        triangle_data = []
        accident_years = []
        if data_start_row < df.shape[0]:
            for r in range(data_start_row, df.shape[0]):
                acc_year_val = df.iloc[r, header_col_idx]
                if pd.isna(acc_year_val) or str(acc_year_val).strip() == "":
                    potential_sum_label_col = header_col_idx - 1 if header_col_idx > 0 else header_col_idx
                    if isinstance(df.iloc[r, potential_sum_label_col], str) and (
                            "合计" in df.iloc[r, potential_sum_label_col] or "Total" in df.iloc[
                        r, potential_sum_label_col]):
                        break
                    if df.iloc[r, header_col_idx: header_col_idx + 1 + num_dev_periods].isnull().all():
                        if len(accident_years) > 0:  # Break if we have collected some years and hit a fully blank data row
                            break
                        else:  # Skip if it's a blank row before any data
                            continue

                try:
                    str_acc_year = str(acc_year_val).strip()
                    first_data_cell_val = df.iloc[r, header_col_idx + 1]
                    # Check if first data cell looks like a number or is NaN (which is acceptable for later AYs)
                    try:
                        pd.to_numeric(first_data_cell_val)
                        is_numeric_check = True
                    except (ValueError, TypeError):
                        is_numeric_check = pd.isna(first_data_cell_val)

                    if not (str_acc_year and (
                            str_acc_year[0].isdigit() or str_acc_year.lower().startswith('ay')) and is_numeric_check):
                        if len(accident_years) > 0:  # If we already started collecting, and this row is not valid, break
                            break
                        else:  # Otherwise, skip this invalid header/ οδηγός row
                            continue

                    accident_years.append(str_acc_year)
                    row_data = df.iloc[r, header_col_idx + 1: header_col_idx + 1 + num_dev_periods].tolist()
                    numeric_row_data = [pd.to_numeric(val, errors='coerce') for val in row_data]
                    triangle_data.append(numeric_row_data)
                except Exception:  # Broad exception to catch any issue during row processing
                    # If an error occurs after some AYs have been added, it might be the end of the valid data block.
                    if len(accident_years) > 0 and str(
                            accident_years[-1]) == str_acc_year:  # Remove last added AY if it was problematic
                        accident_years.pop()
                    break  # Stop processing further rows

        if not triangle_data:
            print("No data extracted for the triangle.")
            return None

        extracted_df = pd.DataFrame(triangle_data, index=pd.Index(accident_years, name="AccidentYear"),
                                    columns=dev_periods)
        extracted_df = extracted_df.dropna(axis=1, how='all').dropna(axis=0, how='all')  # Drop fully NaN rows/cols
        return extracted_df

    except Exception as e:
        print(f"An error occurred during triangle extraction: {e}")
        import traceback
        traceback.print_exc()
        return None


def cumulative_to_incremental(cumulative_triangle):
    """Converts a cumulative triangle to an incremental one."""
    incremental_triangle = cumulative_triangle.copy()
    # Ensure columns are sorted if they represent development periods numerically
    sorted_columns = sorted(cumulative_triangle.columns)
    # First column of incremental is same as first column of cumulative
    # For subsequent columns, subtract the previous cumulative value
    for col_idx in range(1, len(sorted_columns)):
        current_col = sorted_columns[col_idx]
        prev_col = sorted_columns[col_idx - 1]
        incremental_triangle[current_col] = cumulative_triangle[current_col] - cumulative_triangle[prev_col]
    return incremental_triangle


def calculate_weighted_ldfs(cumulative_triangle, trim_extremes=True):
    """Calculates weighted LDFs and a tail factor."""
    n_rows, n_cols = cumulative_triangle.shape
    ldfs_final_selection = []

    triangle_columns = cumulative_triangle.columns
    for j in range(n_cols - 1):  # Iterate up to the second to last column
        col1_vals_all = cumulative_triangle[triangle_columns[j]]
        col2_vals_all = cumulative_triangle[triangle_columns[j + 1]]
        current_period_ldfs_data = []
        # LDFs are calculated for n_rows - (j + 1) accident years for development j to j+1
        for i in range(n_rows - (j + 1)):
            val1 = col1_vals_all.iloc[i]
            val2 = col2_vals_all.iloc[i]
            if pd.notna(val1) and pd.notna(val2) and val1 != 0:
                current_period_ldfs_data.append({'ldf': val2 / val1, 'weight': val1})
            elif pd.notna(val1) and pd.notna(val2) and val1 == 0 and val2 == 0:  # if 0 to 0, LDF is 1
                current_period_ldfs_data.append({'ldf': 1.0, 'weight': 1e-9})  # Small weight for 0/0 cases
            elif pd.notna(val1) and pd.notna(
                    val2) and val1 == 0 and val2 != 0:  # if 0 to X, LDF is large, handle as outlier or cap
                current_period_ldfs_data.append({'ldf': 999.0, 'weight': 1e-9})  # Large LDF, small weight

        if not current_period_ldfs_data:
            ldfs_final_selection.append(1.0)  # Default LDF if no data
            continue

        ldf_values = [d['ldf'] for d in current_period_ldfs_data]
        weights = [d['weight'] for d in current_period_ldfs_data]

        # Trimming logic (same as before)
        if trim_extremes and len(
                ldf_values) >= 5:  # Original code had >=3, let's stick to user's original logic for now if it was >=5
            sorted_indices = np.argsort(ldf_values)
            indices_to_keep = sorted_indices[1:-1]  # Remove 1 smallest and 1 largest
            trimmed_ldf_values = [ldf_values[i] for i in indices_to_keep]
            trimmed_weights = [weights[i] for i in indices_to_keep]
            if not trimmed_ldf_values:  # If trimming leaves nothing
                avg_ldf = 1.0
            else:
                sum_trimmed_weights = sum(trimmed_weights)
                if sum_trimmed_weights == 0:
                    avg_ldf = np.mean(trimmed_ldf_values) if trimmed_ldf_values else 1.0
                else:
                    numerator = sum(ldf * w for ldf, w in zip(trimmed_ldf_values, trimmed_weights))
                    avg_ldf = numerator / sum_trimmed_weights
        else:  # No trimming or not enough data points to trim
            total_weight = sum(weights)
            if total_weight == 0:
                avg_ldf = np.mean(ldf_values) if ldf_values else 1.0
            else:
                numerator = sum(ldf * w for ldf, w in zip(ldf_values, weights))
                avg_ldf = numerator / total_weight
        ldfs_final_selection.append(avg_ldf if pd.notna(avg_ldf) else 1.0)

    # Tail factor calculation (same as before)
    # Use last 3 LDFs from ldfs_final_selection, not from a sub-selection, ensure they are valid
    valid_ldfs_for_tail_calc = [ldf for ldf in ldfs_final_selection[-3:] if
                                pd.notna(ldf) and ldf > 0.1]  # Check last 3 *calculated* LDFs
    if len(valid_ldfs_for_tail_calc) > 0:
        tail_factor = np.mean(valid_ldfs_for_tail_calc)
    else:
        tail_factor = 1.0  # Default tail factor
    tail_factor = max(1.0, tail_factor)  # Ensure tail is not less than 1.0
    ldfs_final_selection.append(tail_factor)
    return np.array(ldfs_final_selection)


def project_triangle(cumulative_triangle, ldfs):
    """Projects a cumulative triangle to ultimate using LDFs."""
    projected_triangle_df = cumulative_triangle.copy().astype(float)
    n_rows, n_cols_orig = projected_triangle_df.shape

    # Ensure LDFs can cover all projections; num_dev_ldfs is max index for LDF array
    # ldfs array includes age-to-age LDFs and one tail factor at the end.
    # If n_cols_orig is N, there are N-1 age-to-age LDFs needed, plus tail.
    # So len(ldfs) should be at least n_cols_orig.
    num_ldfs_available = len(ldfs)

    if 'Ultimate' not in projected_triangle_df.columns:
        # Add columns for future development periods if LDFs suggest more periods than original triangle
        max_dev_periods_from_ldfs = num_ldfs_available  # Each LDF projects one period forward, tail is the last step
        for dev_col_idx in range(n_cols_orig, max_dev_periods_from_ldfs):
            # Use original column names if possible, or create new ones
            new_col_name = cumulative_triangle.columns[0] + dev_col_idx if isinstance(cumulative_triangle.columns[0],
                                                                                      int) else dev_col_idx + 1
            projected_triangle_df[new_col_name] = np.nan
        projected_triangle_df['Ultimate'] = np.nan

    actual_cols_for_projection = projected_triangle_df.columns.tolist()
    if 'Ultimate' in actual_cols_for_projection:
        actual_cols_for_projection.remove('Ultimate')

    num_projection_cols = len(actual_cols_for_projection)

    for i in range(n_rows):  # For each accident year
        current_cumulative_val = np.nan
        last_known_col_idx = -1

        # Find the last known cumulative payment for the current accident year
        for j in range(n_cols_orig):  # Iterate through original columns
            if pd.notna(projected_triangle_df.iloc[i, j]):
                current_cumulative_val = projected_triangle_df.iloc[i, j]
                last_known_col_idx = j
            else:  # Hit a NaN, this is where projection starts internally for this AY
                # or it's an AY that is all NaN
                if last_known_col_idx == -1 and j == 0:  # If first cell is NaN for this AY
                    projected_triangle_df.iloc[i, :n_cols_orig] = 0  # Set all original periods to 0
                    current_cumulative_val = 0
                    last_known_col_idx = n_cols_orig - 1  # effectively starts projection from a full row of 0s
                break  # Stop searching for last known value

        if pd.isna(current_cumulative_val):  # If row is all NaNs or becomes all NaNs
            projected_triangle_df.iloc[i, :] = 0  # Set entire row including Ultimate to 0
            continue

        # Project future payments
        # last_known_col_idx is the 0-indexed column of the last known payment
        # ldfs[k] is LDF from col k to col k+1

        # Fill in the triangle cells for future development periods
        temp_val_for_ultimate = current_cumulative_val
        for proj_col_idx in range(last_known_col_idx + 1, num_projection_cols):
            ldf_idx_to_use = proj_col_idx - 1  # LDF to get from previous col to proj_col_idx
            if last_known_col_idx < proj_col_idx - 1:  # if previous cell was also projected
                ldf_idx_to_use = proj_col_idx - 1
            else:  # if previous cell was the last known actual
                ldf_idx_to_use = last_known_col_idx + (proj_col_idx - (last_known_col_idx + 1))
                # simplified: ldf_idx_to_use = proj_col_idx-1 if using LDF to get *to* proj_col_idx

            # Correct LDF indexing: to project from col K to K+1, use LDFs[K]
            # We are filling column proj_col_idx. The value it's based on is in proj_col_idx-1 (or earlier if that was NaN)
            # The LDF to apply to the value at 'current_ldf_base_col_idx' to get to 'proj_col_idx'
            # is ldfs[current_ldf_base_col_idx]
            current_ldf_base_col_idx = last_known_col_idx + (proj_col_idx - (last_known_col_idx + 1))

            if current_ldf_base_col_idx < num_ldfs_available - 1:  # -1 because last LDF is tail
                ldf = ldfs[current_ldf_base_col_idx]
                ldf = max(1.0, ldf) if pd.notna(ldf) else 1.0
                current_cumulative_val *= ldf
                if proj_col_idx < n_cols_orig:  # check if proj_col_idx is within original bounds, or new extended columns
                    projected_triangle_df.iloc[i, proj_col_idx] = current_cumulative_val
                elif proj_col_idx < num_projection_cols:  # It's one of the newly added columns
                    projected_triangle_df.iloc[i, proj_col_idx] = current_cumulative_val

                temp_val_for_ultimate = current_cumulative_val  # update for ultimate calc
            else:  # Ran out of age-to-age LDFs before filling all projected_triangle_df columns
                # This implies these columns should not be filled further with age-to-age, or are covered by tail
                break  # Stop filling intermediate triangle cells

        # Apply tail factor to the last projected or known cumulative value
        # The tail factor is the last element in the ldfs array
        tail_ldf = ldfs[num_ldfs_available - 1]
        tail_ldf = max(1.0, tail_ldf) if pd.notna(tail_ldf) else 1.0
        ultimate_val = temp_val_for_ultimate * tail_ldf
        projected_triangle_df.loc[projected_triangle_df.index[i], 'Ultimate'] = ultimate_val

    return projected_triangle_df


def calculate_ra(triangle_df, n_simulations, ra_percentile):
    if triangle_df is None or triangle_df.empty:
        print("Input triangle is empty or None. Cannot calculate RA.")
        return None, None, None, None

    cumulative_actual_triangle = triangle_df.apply(pd.to_numeric, errors='coerce').astype(float)
    if cumulative_actual_triangle.isnull().all().all():
        print("Error: Triangle is all NaN after numeric conversion. Check data.")
        return None, None, None, None

    # Calculate LDFs (this part remains unchanged)
    selected_ldfs = calculate_weighted_ldfs(cumulative_actual_triangle.copy(), trim_extremes=True)
    print(f"Selected LDFs (incl. tail): {selected_ldfs}")
    if np.isnan(selected_ldfs).all() or len(selected_ldfs) == 0:
        print("Error: All LDFs are NaN or LDF array is empty. Check data quality or LDF calculation.")
        return None, None, None, None
    # Ensure LDFs are not less than 1.0 (as per original logic in project_triangle)
    # This step is now handled within project_triangle and LDF calculation implicitly or explicitly.
    # Keep selected_ldfs as they are from calculation, project_triangle will handle >=1 rule.

    # --- MODIFICATION START: Calculate "true" fitted values for residual calculation ---
    n_rows, n_cols = cumulative_actual_triangle.shape
    fitted_observed_cumulative_triangle = pd.DataFrame(np.nan, index=cumulative_actual_triangle.index,
                                                       columns=cumulative_actual_triangle.columns)

    for r in range(n_rows):
        if pd.notna(cumulative_actual_triangle.iloc[r, 0]):
            fitted_observed_cumulative_triangle.iloc[r, 0] = cumulative_actual_triangle.iloc[r, 0]
            current_actual_cum = cumulative_actual_triangle.iloc[r, 0]
            for c in range(1, n_cols):
                if pd.notna(current_actual_cum) and c - 1 < len(selected_ldfs) - 1:  # -1 because last ldf is tail
                    ldf_to_apply = selected_ldfs[c - 1]  # LDF from col c-1 to col c
                    ldf_to_apply = max(1.0, ldf_to_apply) if pd.notna(
                        ldf_to_apply) else 1.0  # Consistent with projection
                    fitted_value = current_actual_cum * ldf_to_apply
                    fitted_observed_cumulative_triangle.iloc[r, c] = fitted_value
                    # For the *next* step's fitting, we must use the *actual* as the base
                    # This is crucial for how residuals are typically defined in chain ladder bootstrapping.
                    # The fitted value is E[C_c | C_{c-1}=actual].
                    if pd.notna(cumulative_actual_triangle.iloc[r, c]):
                        current_actual_cum = cumulative_actual_triangle.iloc[
                            r, c]  # Reset for next LDF application based on actual path
                    else:
                        current_actual_cum = fitted_value  # If actual is NaN, continue with fitted for multi-step fitting (less common for residuals)
                        # For residuals, we only care where actuals exist.
                else:
                    break  # Stop if no base actual or no LDF

    # This is the base for calculating residuals and for simulations
    true_incremental_fitted_triangle = cumulative_to_incremental(fitted_observed_cumulative_triangle)
    incremental_actual_triangle = cumulative_to_incremental(cumulative_actual_triangle)

    # Calculate residuals based on these "true" fitted values
    # Only consider residuals where actual incremental payments are known
    residuals_inc = incremental_actual_triangle - true_incremental_fitted_triangle

    # Denominator for scaling: use the "true" fitted incremental values
    sqrt_fitted_inc_clipped = np.sqrt(np.maximum(true_incremental_fitted_triangle, 1e-9))

    standardized_residuals_matrix = residuals_inc / sqrt_fitted_inc_clipped
    # --- MODIFICATION END ---

    pool_of_residuals = []
    for r in range(n_rows):
        for c in range(n_cols):
            # Only pool residuals from cells where actual incremental data (and thus actual cumulative) existed to form it
            if pd.notna(cumulative_actual_triangle.iloc[r, c]) and \
                    (c == 0 or pd.notna(
                        cumulative_actual_triangle.iloc[r, c - 1])):  # Ensure the incremental value was observable
                if pd.notna(true_incremental_fitted_triangle.iloc[r, c]) and \
                        pd.notna(standardized_residuals_matrix.iloc[r, c]) and \
                        np.isfinite(standardized_residuals_matrix.iloc[r, c]):  # Check for finite residuals
                    pool_of_residuals.append(standardized_residuals_matrix.iloc[r, c])

    if not pool_of_residuals:
        print(
            "Error: No valid residuals could be calculated for the pool. All residuals might be NaN/inf or pool is empty.")
        # Add a fallback: use a dummy residual if none available, to see if rest of sim works
        # This is a diagnostic, in reality, this situation means model/data issues
        print("Using a dummy residual pool of [0] for diagnostic purposes.")
        pool_of_residuals = [0.0]  # This will lead to Std Dev 0 again if reached
        # return None, None, None, None # Original behavior

    pool_of_residuals = np.array(pool_of_residuals)
    # pool_of_residuals = pool_of_residuals[np.isfinite(pool_of_residuals)] # Already filtered above
    if len(pool_of_residuals) == 0:  # Should not happen if previous check passed and fallback not used
        print("Error: Residual pool is empty after removing non-finite values (should not happen).")
        return None, None, None, None

    # Deterministic projection for reference (this is what the original code's "fitted_triangle" was)
    # This is NOT used for residuals anymore, but for comparison or as one view of "best estimate ultimate"
    reference_projected_triangle_full = project_triangle(cumulative_actual_triangle.copy(), selected_ldfs)

    current_latest_paid = 0
    for r_idx in range(cumulative_actual_triangle.shape[0]):
        row_data = cumulative_actual_triangle.iloc[r_idx, :].dropna()
        if not row_data.empty:
            current_latest_paid += row_data.iloc[-1]
    if pd.isna(current_latest_paid): current_latest_paid = 0

    simulated_ibnrs = []
    for sim_num in range(n_simulations):
        # --- MODIFICATION START: Base of simulation is true_incremental_fitted_triangle ---
        sim_incremental_triangle = true_incremental_fitted_triangle.copy()
        # --- MODIFICATION END ---

        observed_mask_for_residuals = true_incremental_fitted_triangle.notna() & cumulative_actual_triangle.notna()
        # We only apply residuals where we could calculate a true_incremental_fitted_triangle value
        # and where the original cumulative_actual_triangle had data (implicitly handled by residual pool creation)

        num_residuals_to_apply = observed_mask_for_residuals.sum().sum()

        if num_residuals_to_apply > 0 and len(pool_of_residuals) > 0:
            sampled_std_residuals_flat = np.random.choice(pool_of_residuals, size=num_residuals_to_apply, replace=True)
        else:
            sampled_std_residuals_flat = np.array([])

        k_res_idx = 0
        for r in range(sim_incremental_triangle.shape[0]):
            for c in range(sim_incremental_triangle.shape[1]):
                # Apply residual if this cell was part of the observed history where residuals were calculated
                if observed_mask_for_residuals.iloc[r, c]:
                    if k_res_idx < len(sampled_std_residuals_flat):
                        sampled_std_residual = sampled_std_residuals_flat[k_res_idx]
                        k_res_idx += 1

                        # --- MODIFICATION START: Use true_incremental_fitted_triangle as base ---
                        fitted_inc_val = true_incremental_fitted_triangle.iloc[r, c]
                        # --- MODIFICATION END ---

                        if pd.isna(fitted_inc_val): fitted_inc_val = 0  # Should not happen if mask is correct

                        sqrt_val_for_destd = np.sqrt(max(fitted_inc_val, 1e-9))  # De-standardization factor

                        sim_payment_inc = fitted_inc_val + sampled_std_residual * sqrt_val_for_destd
                        sim_incremental_triangle.iloc[r, c] = max(0, sim_payment_inc)
                    # else:
                    # This cell in true_incremental_fitted_triangle was NaN, or no more residuals
                    # It remains as its original fitted value (or NaN if it was NaN)
                    # This part of logic might need refinement based on how NaNs in true_incremental_fitted propagate
                    # For now, sim_incremental_triangle starts as true_incremental_fitted_triangle,
                    # so cells not touched by residuals keep their "fitted" value.
                    # else:
                    # This cell is beyond the observed part for which we have fitted values for residuals,
                    # or was NaN in true_incremental_fitted_triangle.
                    # These will be handled by cumsum and projection.
                    # If true_incremental_fitted_triangle has NaNs in the lower right, cumsum will propagate them.
                    # Project_triangle will then take the latest available from this sim_cumulative_triangle.
                    pass

        sim_cumulative_triangle = sim_incremental_triangle.cumsum(axis=1)
        # Handle potential NaNs from cumsum if sim_incremental_triangle had leading NaNs (shouldn't if first col is actual)
        sim_cumulative_triangle = sim_cumulative_triangle.ffill(axis=1).fillna(0)

        sim_ultimate_triangle_full = project_triangle(sim_cumulative_triangle.copy(), selected_ldfs)
        sim_ultimate_claims_sum = sim_ultimate_triangle_full['Ultimate'].sum()
        if pd.isna(sim_ultimate_claims_sum): sim_ultimate_claims_sum = 0

        ibnr = sim_ultimate_claims_sum - current_latest_paid
        simulated_ibnrs.append(max(0, ibnr))

    simulated_ibnrs.sort()
    ra_index = int(n_simulations * ra_percentile) - 1
    ra_index = min(max(0, ra_index), n_simulations - 1)  # Ensure index is within bounds
    ra_value = simulated_ibnrs[ra_index] if simulated_ibnrs else np.nan
    mean_ibnr = np.mean(simulated_ibnrs) if simulated_ibnrs else np.nan

    # Return the reference_projected_triangle_full as the "fitted_triangle" for consistency in reporting a deterministic view
    return mean_ibnr, ra_value, simulated_ibnrs, reference_projected_triangle_full


if __name__ == '__main__':
    # Set Pandas display options
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    EXCEL_FILE_PATH = "202412三角形.xlsx"  # Replace with your actual file path
    SHEET_NAME = "商业三者"  # Replace with your actual sheet name

    try:
        df_sheet_data = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME, header=None, dtype=str)
        print(f"Successfully loaded sheet '{SHEET_NAME}' from '{EXCEL_FILE_PATH}'")

        commercial_pre_re_paid_triangle = extract_triangle_from_df(df_sheet_data,
                                                                   "再保前",
                                                                   "累计已决三角形")

        if commercial_pre_re_paid_triangle is not None and not commercial_pre_re_paid_triangle.empty:
            print("\nSuccessfully extracted '商业三者再保前已决' triangle from the sheet:")

            for col in commercial_pre_re_paid_triangle.columns:
                commercial_pre_re_paid_triangle[col] = pd.to_numeric(commercial_pre_re_paid_triangle[col],
                                                                     errors='coerce')

            commercial_pre_re_paid_triangle = commercial_pre_re_paid_triangle.dropna(axis=0, how='all').dropna(axis=1,
                                                                                                               how='all')

            if commercial_pre_re_paid_triangle.empty:
                print("Triangle became empty after numeric conversion and NaN drop. Check data within the sheet.")
            else:
                print("\nCleaned '商业三者再保前已决' triangle for processing (displaying all rows):")
                print(commercial_pre_re_paid_triangle.to_string())
                print(f"Triangle shape: {commercial_pre_re_paid_triangle.shape}")

                print("\nStarting RA calculation...")
                mean_ibnr, ra_at_75_val, all_ibnrs, fitted_triangle_for_display = calculate_ra(
                    commercial_pre_re_paid_triangle,
                    n_simulations=100,  # You can increase this for more stable results, e.g., 5000 or 10000
                    ra_percentile=0.75
                )
                # The RA itself is the difference between the percentile and the mean
                if mean_ibnr is not None and ra_at_75_val is not None:
                    risk_adjustment_final = ra_at_75_val - mean_ibnr
                else:
                    risk_adjustment_final = np.nan

                if mean_ibnr is not None and ra_at_75_val is not None and all_ibnrs is not None:
                    print(f"\n--- Results for '商业三者再保前已决' ---")
                    print(f"Estimated Mean IBNR: {mean_ibnr:,.2f}")
                    print(f"Value at 75th Percentile of IBNR: {ra_at_75_val:,.2f}")
                    print(f"Risk Adjustment (RA) (75th Pctl - Mean): {risk_adjustment_final:,.2f}")

                    # --- Statistics for Simulated IBNRs (for verification) ---
                    if all_ibnrs:  # Check if list is not empty
                        print(f"\n--- Statistics for Simulated IBNRs ---")
                        print(f"Number of simulations: {len(all_ibnrs)}")
                        print(f"Min IBNR: {np.min(all_ibnrs):,.2f}")
                        print(f"Max IBNR: {np.max(all_ibnrs):,.2f}")
                        # Mean is already printed
                        print(f"Std Dev IBNR: {np.std(all_ibnrs):,.2f}")
                        if len(all_ibnrs) >= 5:
                            print(f"First 5 IBNRs: {[f'{x:,.2f}' for x in all_ibnrs[:5]]}")
                            print(f"Last 5 IBNRs (sorted): {[f'{x:,.2f}' for x in all_ibnrs[-5:]]}")
                    # --- End Statistics ---

                    print("\nReference Projected Cumulative Triangle (including Ultimate) (displaying all rows):")
                    if fitted_triangle_for_display is not None:
                        formatted_fitted_triangle = fitted_triangle_for_display.copy()
                        for c_col in formatted_fitted_triangle.columns:
                            if formatted_fitted_triangle[c_col].dtype in [np.int64, np.float64, int, float]:
                                formatted_fitted_triangle[c_col] = formatted_fitted_triangle[c_col].map(
                                    '{:,.0f}'.format)
                        print(formatted_fitted_triangle.to_string())
                    else:
                        print("Reference projected triangle is None.")

                    if all_ibnrs and np.std(all_ibnrs) > 1e-6:  # Check for non-zero variance before plotting
                        plt.figure(figsize=(12, 6))
                        plt.subplot(1, 2, 1)
                        plt.hist(all_ibnrs, bins=50, color='skyblue', edgecolor='black')
                        plt.title('Distribution of Simulated IBNR (Histogram)')
                        plt.xlabel('Simulated IBNR Value')
                        plt.ylabel('Frequency')
                        plt.axvline(mean_ibnr, color='red', linestyle='dashed', linewidth=1,
                                    label=f'Mean IBNR: {mean_ibnr:,.0f}')
                        plt.axvline(ra_at_75_val, color='green', linestyle='dashed', linewidth=1,
                                    label=f'75th Pctl: {ra_at_75_val:,.0f}')
                        plt.legend()
                        plt.ticklabel_format(style='plain', axis='x')

                        plt.subplot(1, 2, 2)
                        sns.kdeplot(all_ibnrs, fill=True, color="olive", linewidth=2)
                        plt.title('Distribution of Simulated IBNR (KDE Plot)')
                        plt.xlabel('Simulated IBNR Value')
                        plt.ylabel('Density')
                        plt.axvline(mean_ibnr, color='red', linestyle='dashed', linewidth=1,
                                    label=f'Mean IBNR: {mean_ibnr:,.0f}')
                        plt.axvline(ra_at_75_val, color='green', linestyle='dashed', linewidth=1,
                                    label=f'75th Pctl: {ra_at_75_val:,.0f}')
                        plt.legend()
                        plt.ticklabel_format(style='plain', axis='x')

                        plt.tight_layout()
                        plot_filename = "ibnr_distribution_plot.png"
                        plt.savefig(plot_filename)
                        print(f"\nIBNR distribution plot saved as '{plot_filename}'")
                        # plt.show()
                    elif all_ibnrs:
                        print(
                            "\nIBNR distribution plot not generated due to zero or negligible variance in simulated IBNRs.")
                    else:
                        print("\nIBNR distribution plot not generated as no IBNR values were simulated.")
                else:
                    print("\nRA calculation failed or returned None values.")
        else:
            print(f"\nCould not extract the '商业三者再保前已决' triangle from sheet '{SHEET_NAME}'.")

    except FileNotFoundError:
        print(f"Error: Excel file '{EXCEL_FILE_PATH}' not found.")
    except ValueError as ve:
        if "Worksheet" in str(ve) and SHEET_NAME in str(ve):
            print(f"Error: Sheet named '{SHEET_NAME}' not found in '{EXCEL_FILE_PATH}'.")
        else:
            print(f"A ValueError occurred: {ve}")
            import traceback

            traceback.print_exc()
    except Exception as e:
        if "openpyxl" in str(e).lower():
            print(
                f"An error occurred: {e}. This might be due to 'openpyxl' not being installed (pip install openpyxl).")
        elif "matplotlib" in str(e).lower() or "seaborn" in str(e).lower():
            print(f"An error occurred: {e}. This might be due to plotting libraries not being installed.")
        else:
            print(f"An unexpected error occurred: {e}")
            import traceback

            traceback.print_exc()