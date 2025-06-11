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
        paid_keyword_rows = df[
            df.apply(lambda r: r.astype(str).str.contains(specific_triangle_keyword).any(), axis=1)].index
        if not paid_keyword_rows.any():
            print(f"Keyword '{specific_triangle_keyword}' not found.")
            return None

        accident_month_loc = None
        for r_idx in paid_keyword_rows:
            potential_accident_month_row = r_idx + 1
            if potential_accident_month_row < df.shape[0]:
                for c_idx in range(min(5, df.shape[1])):
                    if isinstance(df.iloc[potential_accident_month_row, c_idx], str) and "事故月份" in df.iloc[
                        potential_accident_month_row, c_idx]:
                        is_pre_reinsurance_block = False
                        for prev_r_idx in range(r_idx - 3, r_idx + 1):
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

        if not accident_month_loc:
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
        header_col_idx = accident_month_loc[1]

        dev_periods = []
        if dev_period_header_row < df.shape[0]:
            for c in range(header_col_idx + 1, df.shape[1]):
                val = df.iloc[dev_period_header_row, c]
                try:
                    float_val = float(val)
                    dev_periods.append(int(float_val))
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
                        if len(accident_years) > 0:
                            break
                        else:
                            continue
                try:
                    str_acc_year = str(acc_year_val).strip()
                    first_data_cell_val = df.iloc[r, header_col_idx + 1]
                    try:
                        pd.to_numeric(first_data_cell_val)
                        is_numeric_check = True
                    except (ValueError, TypeError):
                        is_numeric_check = pd.isna(first_data_cell_val)

                    if not (str_acc_year and (
                            str_acc_year[0].isdigit() or str_acc_year.lower().startswith('ay')) and is_numeric_check):
                        if len(accident_years) > 0:
                            break
                        else:
                            continue
                    accident_years.append(str_acc_year)
                    row_data = df.iloc[r, header_col_idx + 1: header_col_idx + 1 + num_dev_periods].tolist()
                    numeric_row_data = [pd.to_numeric(val, errors='coerce') for val in row_data]
                    triangle_data.append(numeric_row_data)
                except Exception:
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
    incremental_triangle = cumulative_triangle.copy()
    # Ensure columns are sorted if they represent development periods numerically
    # If columns are not purely numeric, this sorting might not be appropriate
    # For this specific use case, assuming columns are 1, 2, 3... or already in correct order.
    # sorted_columns = sorted(cumulative_triangle.columns) # This might fail if column names are not sortable

    # Assuming columns are already in the correct sequential order (e.g., 1, 2, ..., N)
    for col_idx in range(1, incremental_triangle.shape[1]):
        current_col_name = incremental_triangle.columns[col_idx]
        prev_col_name = incremental_triangle.columns[col_idx - 1]
        incremental_triangle[current_col_name] = cumulative_triangle[current_col_name] - cumulative_triangle[
            prev_col_name]
    return incremental_triangle


def calculate_weighted_ldfs(cumulative_triangle, trim_extremes=True):
    n_rows, n_cols = cumulative_triangle.shape
    ldfs_final_selection = []
    triangle_columns = cumulative_triangle.columns

    for j in range(n_cols - 1):
        col1_vals_all = cumulative_triangle[triangle_columns[j]]
        col2_vals_all = cumulative_triangle[triangle_columns[j + 1]]
        current_period_ldfs_data = []
        for i in range(n_rows - (j + 1)):
            val1 = col1_vals_all.iloc[i]
            val2 = col2_vals_all.iloc[i]
            if pd.notna(val1) and pd.notna(val2) and val1 != 0:
                current_period_ldfs_data.append({'ldf': val2 / val1, 'weight': val1})
            elif pd.notna(val1) and pd.notna(val2) and val1 == 0 and val2 == 0:
                current_period_ldfs_data.append({'ldf': 1.0, 'weight': 1e-9})
            elif pd.notna(val1) and pd.notna(val2) and val1 == 0 and val2 != 0:
                current_period_ldfs_data.append({'ldf': 999.0, 'weight': 1e-9})
        if not current_period_ldfs_data:
            ldfs_final_selection.append(1.0)
            continue
        ldf_values = [d['ldf'] for d in current_period_ldfs_data]
        weights = [d['weight'] for d in current_period_ldfs_data]
        if trim_extremes and len(ldf_values) >= 5:
            sorted_indices = np.argsort(ldf_values)
            indices_to_keep = sorted_indices[1:-1]
            trimmed_ldf_values = [ldf_values[i] for i in indices_to_keep]
            trimmed_weights = [weights[i] for i in indices_to_keep]
            if not trimmed_ldf_values:
                avg_ldf = 1.0
            else:
                sum_trimmed_weights = sum(trimmed_weights)
                if sum_trimmed_weights == 0:
                    avg_ldf = np.mean(trimmed_ldf_values) if trimmed_ldf_values else 1.0
                else:
                    numerator = sum(ldf * w for ldf, w in zip(trimmed_ldf_values, trimmed_weights))
                    avg_ldf = numerator / sum_trimmed_weights
        else:
            total_weight = sum(weights)
            if total_weight == 0:
                avg_ldf = np.mean(ldf_values) if ldf_values else 1.0
            else:
                numerator = sum(ldf * w for ldf, w in zip(ldf_values, weights))
                avg_ldf = numerator / total_weight
        ldfs_final_selection.append(avg_ldf if pd.notna(avg_ldf) else 1.0)

    valid_ldfs_for_tail_calc = [ldf for ldf in ldfs_final_selection[-3:] if pd.notna(ldf) and ldf > 0.1]
    if len(valid_ldfs_for_tail_calc) > 0:
        tail_factor = np.mean(valid_ldfs_for_tail_calc)
    else:
        tail_factor = 1.0
    tail_factor = max(1.0, tail_factor)
    ldfs_final_selection.append(tail_factor)
    return np.array(ldfs_final_selection)


def project_triangle(cumulative_triangle, ldfs):
    projected_triangle_df = cumulative_triangle.copy().astype(float)
    n_rows, n_cols_orig = projected_triangle_df.shape
    num_ldfs_available = len(ldfs)

    if 'Ultimate' not in projected_triangle_df.columns:
        max_dev_periods_from_ldfs = num_ldfs_available
        # Ensure new column names are consistent if original are integers
        original_cols_are_int = all(isinstance(c, int) for c in cumulative_triangle.columns)

        for dev_col_idx in range(n_cols_orig, max_dev_periods_from_ldfs):
            if original_cols_are_int and cumulative_triangle.columns.max() < dev_col_idx + 1:
                new_col_name = dev_col_idx + 1  # Assuming dev periods are 1-indexed for display
            else:  # Fallback or if original columns are not simple integers
                new_col_name = f"Dev_{dev_col_idx + 1}" if not original_cols_are_int else dev_col_idx + 1

            # Check if column already exists from a previous extension or by chance
            if new_col_name not in projected_triangle_df.columns:
                projected_triangle_df[new_col_name] = np.nan
        projected_triangle_df['Ultimate'] = np.nan

    # Determine the columns that represent development periods (excluding 'Ultimate')
    dev_period_cols = [col for col in projected_triangle_df.columns if col != 'Ultimate']
    num_dev_period_cols_total = len(dev_period_cols)

    for i in range(n_rows):
        current_cumulative_val = np.nan
        last_known_col_idx_in_orig = -1  # Tracks index relative to n_cols_orig

        for j in range(n_cols_orig):
            if pd.notna(projected_triangle_df.iloc[i, j]):
                current_cumulative_val = projected_triangle_df.iloc[i, j]
                last_known_col_idx_in_orig = j
            else:
                if last_known_col_idx_in_orig == -1 and j == 0:
                    projected_triangle_df.iloc[i, :n_cols_orig] = 0
                    current_cumulative_val = 0
                    last_known_col_idx_in_orig = n_cols_orig - 1
                break
        if pd.isna(current_cumulative_val):
            projected_triangle_df.iloc[i, :] = 0
            continue

        temp_val_for_ultimate = current_cumulative_val

        # Project through all development columns up to (but not including) 'Ultimate'
        # last_known_col_idx_in_orig is the 0-based index of the last column with data in the original part
        # LDFs are 0-indexed: ldfs[k] is from dev k to dev k+1 (column k to k+1)

        current_col_being_projected_from_idx = last_known_col_idx_in_orig

        for k_target_dev_col_idx in range(last_known_col_idx_in_orig + 1, num_dev_period_cols_total):
            if current_col_being_projected_from_idx < num_ldfs_available - 1:  # Ensure LDF exists (not tail yet)
                ldf_to_apply = ldfs[current_col_being_projected_from_idx]
                ldf_to_apply = max(1.0, ldf_to_apply) if pd.notna(ldf_to_apply) else 1.0

                temp_val_for_ultimate *= ldf_to_apply
                projected_triangle_df.iloc[i, k_target_dev_col_idx] = temp_val_for_ultimate
                current_col_being_projected_from_idx += 1  # Move to next LDF for next projection
            else:  # Ran out of specific age-to-age LDFs
                break

        # Apply tail factor (last LDF in the ldfs array)
        tail_ldf = ldfs[num_ldfs_available - 1]
        tail_ldf = max(1.0, tail_ldf) if pd.notna(tail_ldf) else 1.0
        ultimate_val = temp_val_for_ultimate * tail_ldf
        projected_triangle_df.loc[projected_triangle_df.index[i], 'Ultimate'] = ultimate_val
    return projected_triangle_df


def calculate_ra(triangle_df, n_simulations, ra_percentile):
    if triangle_df is None or triangle_df.empty:
        print("Input triangle is empty or None. Cannot calculate RA.")
        return None, None, None, None, None, None

    cumulative_actual_triangle = triangle_df.apply(pd.to_numeric, errors='coerce').astype(float)
    if cumulative_actual_triangle.isnull().all().all():
        print("Error: Triangle is all NaN after numeric conversion. Check data.")
        return None, None, None, None, None, None

    selected_ldfs = calculate_weighted_ldfs(cumulative_actual_triangle.copy(), trim_extremes=True)
    print(f"Selected LDFs (incl. tail): {selected_ldfs}")
    if selected_ldfs is None or len(selected_ldfs) == 0 or np.all(np.isnan(selected_ldfs)):
        print("Error: All LDFs are NaN or LDF array is empty/None. Check data quality or LDF calculation.")
        return None, None, None, None, None, None

    n_rows, n_cols = cumulative_actual_triangle.shape
    fitted_observed_cumulative_triangle = pd.DataFrame(np.nan, index=cumulative_actual_triangle.index,
                                                       columns=cumulative_actual_triangle.columns)

    for r in range(n_rows):
        if pd.notna(cumulative_actual_triangle.iloc[r, 0]):
            fitted_observed_cumulative_triangle.iloc[r, 0] = cumulative_actual_triangle.iloc[r, 0]
            current_actual_cum_for_fitting = cumulative_actual_triangle.iloc[r, 0]
            for c in range(1, n_cols):
                if pd.notna(current_actual_cum_for_fitting) and (c - 1) < (len(selected_ldfs) - 1):
                    ldf_to_apply = selected_ldfs[c - 1]
                    if pd.isna(ldf_to_apply):
                        ldf_to_apply = 1.0

                    fitted_value = current_actual_cum_for_fitting * ldf_to_apply
                    fitted_observed_cumulative_triangle.iloc[r, c] = fitted_value

                    if pd.notna(cumulative_actual_triangle.iloc[r, c]):
                        current_actual_cum_for_fitting = cumulative_actual_triangle.iloc[r, c]
                    else:
                        current_actual_cum_for_fitting = np.nan
                else:
                    break

    true_incremental_fitted_triangle = cumulative_to_incremental(fitted_observed_cumulative_triangle)

    # --- 新增代码块开始 ---
    print("\n--- True Incremental Fitted Triangle (Base for Residuals) ---")
    # true_incremental_fitted_triangle 继承了 fitted_observed_cumulative_triangle 的索引和列名
    # fitted_observed_cumulative_triangle 的索引和列名与 cumulative_actual_triangle 一致
    print(true_incremental_fitted_triangle.to_string(float_format="%.2f"))
    # --- 新增代码块结束 ---

    incremental_actual_triangle = cumulative_to_incremental(cumulative_actual_triangle)

    residuals_inc = incremental_actual_triangle - true_incremental_fitted_triangle
    # --- 修改标准化分母的逻辑 ---
    # 分母将是 sqrt(期初实际累积赔款额)
    # 对于第 c 列的增量赔款，其期初实际累积赔款额是第 c-1 列的实际累积赔款额

    # 创建一个用于存储标准化分母的DataFrame，与cumulative_actual_triangle形状相同
    # 初始化为1.0，以处理第一列（其残差通常为0，标准化的具体值不影响结果，但要避免除以0）
    scaling_denominator_sqrt = pd.DataFrame(1.0,
                                            index=cumulative_actual_triangle.index,
                                            columns=cumulative_actual_triangle.columns)

    if n_cols > 1:
        # 对于第2列及以后的列 (c_idx from 1 to n_cols-1)
        # 其对应的期初累积赔款在原始累积三角形的 c_idx-1 列
        actual_cumul_at_interval_start = cumulative_actual_triangle.iloc[:, :-1].values  # 取所有行，除了最后一列的所有列

        # np.maximum确保至少为1e-9，然后开方
        # 这个结果的列数会比 scaling_denominator_sqrt 少一列，所以我们赋值给 scaling_denominator_sqrt 的第2列及以后
        scaling_denominator_sqrt.iloc[:, 1:] = np.sqrt(np.maximum(actual_cumul_at_interval_start, 1e-9))

    standardized_residuals_matrix = residuals_inc / scaling_denominator_sqrt
    # --- 标准化分母逻辑修改结束 ---

    print("\n--- Raw Residuals (Actual Incremental - Fitted Incremental) ---")
    residuals_inc_df = pd.DataFrame(residuals_inc.values, index=cumulative_actual_triangle.index,
                                    columns=cumulative_actual_triangle.columns)
    print(residuals_inc_df.to_string(float_format="%.2f"))

    print("\n--- Standardized Residuals ---")
    standardized_residuals_df = pd.DataFrame(standardized_residuals_matrix.values,
                                             index=cumulative_actual_triangle.index,
                                             columns=cumulative_actual_triangle.columns)
    print(standardized_residuals_df.to_string(float_format="%.2f"))

    pool_of_residuals = []
    for r_idx in range(n_rows):
        for c_idx in range(n_cols):
            if pd.notna(cumulative_actual_triangle.iloc[r_idx, c_idx]) and \
                    (c_idx == 0 or pd.notna(cumulative_actual_triangle.iloc[r_idx, c_idx - 1])):
                if pd.notna(true_incremental_fitted_triangle.iloc[r_idx, c_idx]) and \
                        pd.notna(standardized_residuals_matrix.iloc[r_idx, c_idx]) and \
                        np.isfinite(standardized_residuals_matrix.iloc[r_idx, c_idx]):
                    pool_of_residuals.append(standardized_residuals_matrix.iloc[r_idx, c_idx])

    if not pool_of_residuals:
        print("Error: No valid residuals in pool. Using dummy pool [0]. Check data/fitting.")
        pool_of_residuals = [0.0]

    pool_of_residuals = np.array(pool_of_residuals)
    if len(pool_of_residuals) == 0:
        print("Error: Residual pool is critically empty.")
        return None, None, None, None, residuals_inc_df, standardized_residuals_df

    reference_projected_triangle_full = project_triangle(cumulative_actual_triangle.copy(), selected_ldfs)
    current_latest_paid = 0
    for r_idx in range(cumulative_actual_triangle.shape[0]):
        row_data = cumulative_actual_triangle.iloc[r_idx, :].dropna()
        if not row_data.empty:
            current_latest_paid += row_data.iloc[-1]
    if pd.isna(current_latest_paid): current_latest_paid = 0

    simulated_ibnrs = []
    for sim_num in range(n_simulations):
        sim_incremental_triangle = true_incremental_fitted_triangle.copy()
        cells_for_residual_application = []
        for r_sim in range(n_rows):
            for c_sim in range(n_cols):
                if pd.notna(true_incremental_fitted_triangle.iloc[r_sim, c_sim]) and \
                        pd.notna(standardized_residuals_matrix.iloc[r_sim, c_sim]) and \
                        np.isfinite(standardized_residuals_matrix.iloc[r_sim, c_sim]):
                    cells_for_residual_application.append((r_sim, c_sim))

        if cells_for_residual_application and len(pool_of_residuals) > 0:
            sampled_indices = np.random.choice(len(pool_of_residuals), size=len(cells_for_residual_application),
                                               replace=True)
            sampled_std_residuals_flat = pool_of_residuals[sampled_indices]
        else:
            sampled_std_residuals_flat = np.array([])

        res_idx = 0
        for r_sim, c_sim in cells_for_residual_application:
            if res_idx < len(sampled_std_residuals_flat):
                sampled_std_residual = sampled_std_residuals_flat[res_idx]
                res_idx += 1

                fitted_inc_val = true_incremental_fitted_triangle.iloc[r_sim, c_sim]
                # --- 修改去标准化因子的逻辑 ---
                # 去标准化的因子必须与标准化时使用的因子一致
                # scaling_denominator_sqrt 已经预先计算好了，包含了每个单元格(r_sim, c_sim)对应的 sqrt(C_actual_at_start_of_interval)
                sqrt_val_for_destd = scaling_denominator_sqrt.iloc[r_sim, c_sim]
                # --- 去标准化因子逻辑修改结束 ---

                sim_payment_inc = fitted_inc_val + sampled_std_residual * sqrt_val_for_destd
                sim_incremental_triangle.iloc[r_sim, c_sim] = max(0, sim_payment_inc)

        sim_cumulative_triangle = sim_incremental_triangle.cumsum(axis=1)
        sim_cumulative_triangle = sim_cumulative_triangle.ffill(axis=1).fillna(0)

        sim_ultimate_triangle_full = project_triangle(sim_cumulative_triangle.copy(), selected_ldfs)
        sim_ultimate_claims_sum = sim_ultimate_triangle_full['Ultimate'].sum()
        if pd.isna(sim_ultimate_claims_sum): sim_ultimate_claims_sum = 0

        ibnr = sim_ultimate_claims_sum - current_latest_paid
        simulated_ibnrs.append(max(0, ibnr))

    simulated_ibnrs.sort()
    ra_index = int(n_simulations * ra_percentile) - 1
    ra_index = min(max(0, ra_index), n_simulations - 1)
    ra_value = simulated_ibnrs[ra_index] if simulated_ibnrs else np.nan
    mean_ibnr = np.mean(simulated_ibnrs) if simulated_ibnrs else np.nan

    return mean_ibnr, ra_value, simulated_ibnrs, reference_projected_triangle_full, residuals_inc_df, standardized_residuals_df

if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    EXCEL_FILE_PATH = "202412三角形.xlsx"
    SHEET_NAME = "财产险"

    try:
        df_sheet_data = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME, header=None, dtype=str)
        print(f"Successfully loaded sheet '{SHEET_NAME}' from '{EXCEL_FILE_PATH}'")
        commercial_pre_re_paid_triangle = extract_triangle_from_df(df_sheet_data, "再保前", "累计已决三角形")

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
                mean_ibnr, ra_at_75_val, all_ibnrs, fitted_triangle_for_display, raw_residuals_df, std_residuals_df = calculate_ra(
                    commercial_pre_re_paid_triangle,
                    n_simulations=1000,  # For actual results, consider 5000 or 10000
                    ra_percentile=0.75
                )
                if mean_ibnr is not None and ra_at_75_val is not None:
                    risk_adjustment_final = ra_at_75_val - mean_ibnr
                else:
                    risk_adjustment_final = np.nan

                if mean_ibnr is not None and ra_at_75_val is not None and all_ibnrs is not None:
                    print(f"\n--- Results for '商业三者再保前已决' ---")
                    print(f"Estimated Mean IBNR: {mean_ibnr:,.2f}")
                    print(f"Value at 75th Percentile of IBNR: {ra_at_75_val:,.2f}")
                    print(f"Risk Adjustment (RA) (75th Pctl - Mean): {risk_adjustment_final:,.2f}")
                    if all_ibnrs:
                        print(f"\n--- Statistics for Simulated IBNRs ---")
                        print(f"Number of simulations: {len(all_ibnrs)}")
                        print(f"Min IBNR: {np.min(all_ibnrs):,.2f}")
                        print(f"Max IBNR: {np.max(all_ibnrs):,.2f}")
                        print(f"Std Dev IBNR: {np.std(all_ibnrs):,.2f}")
                        if len(all_ibnrs) >= 5:
                            print(f"First 5 IBNRs: {[f'{x:,.2f}' for x in sorted(all_ibnrs)[:5]]}")  # Show sorted
                            print(f"Last 5 IBNRs (sorted): {[f'{x:,.2f}' for x in sorted(all_ibnrs)[-5:]]}")
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
                    if all_ibnrs and np.std(all_ibnrs) > 1e-6:
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