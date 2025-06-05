import pandas as pd
import numpy as np
from scipy.stats import trim_mean  # Used if trimming strategy involves it (not directly in CoC part)
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For a nicer KDE plot


# Solution: Set a font that supports Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei']  # Or 'Microsoft YaHei', 'SimSun', etc.
plt.rcParams['axes.unicode_minus'] = False  # Fix for displaying the minus sign correctly

# --- 从您原始代码中复制的函数 ---
def extract_triangle_from_df(df, triangle_type_keyword, specific_triangle_keyword):
    """
    Extracts a specific runoff triangle from the loaded DataFrame.
    (代码与您之前提供的一致，此处省略以减少篇幅，假设其功能正确)
    """
    try:
        paid_keyword_rows = df[
            df.apply(lambda r: r.astype(str).str.contains(specific_triangle_keyword).any(), axis=1)].index
        if not paid_keyword_rows.any():
            print(f"Keyword '{specific_triangle_keyword}' not found.")
            return None

        accident_month_loc = None
        # Try to find "事故月份" under the specific_triangle_keyword first, then check triangle_type_keyword proximity
        for r_idx in paid_keyword_rows:
            potential_accident_month_row = r_idx + 1
            if potential_accident_month_row < df.shape[0]:
                # Check for "事故月份" in a few initial columns of the next row
                for c_idx in range(min(5, df.shape[1])):  # Check first 5 columns
                    if isinstance(df.iloc[potential_accident_month_row, c_idx], str) and "事故月份" in df.iloc[
                        potential_accident_month_row, c_idx]:
                        # Now check if the triangle_type_keyword is nearby (above)
                        is_correct_block = False
                        # Look in the rows from 3 above the specific_triangle_keyword up to the specific_triangle_keyword row
                        # for the triangle_type_keyword, in the same column or nearby.
                        # This proximity check helps disambiguate if multiple specific_triangle_keywords exist.
                        start_check_row = max(0, r_idx - 3)
                        for prev_r_idx in range(start_check_row, r_idx + 1):
                            # Check a few columns around c_idx for the type keyword
                            for check_c_idx in range(max(0, c_idx - 1), min(df.shape[1], c_idx + 2)):
                                if prev_r_idx < df.shape[0] and isinstance(df.iloc[prev_r_idx, check_c_idx], str) and \
                                        triangle_type_keyword in df.iloc[prev_r_idx, check_c_idx]:
                                    is_correct_block = True
                                    break
                            if is_correct_block:
                                break

                        if is_correct_block:
                            accident_month_loc = (potential_accident_month_row, c_idx)
                            break  # Found "事故月份" under the correct block
            if accident_month_loc:
                break  # Exit outer loop if found

        # Fallback: If the above didn't find it, try a more general search for "事故月份"
        # that is directly below specific_triangle_keyword and then check triangle_type_keyword above that.
        if not accident_month_loc:
            for r_idx_main in range(df.shape[0]):
                for c_idx_main in range(min(5, df.shape[1])):  # Check first 5 columns
                    if isinstance(df.iloc[r_idx_main, c_idx_main], str) and "事故月份" in df.iloc[
                        r_idx_main, c_idx_main]:
                        # Check if specific_triangle_keyword is directly above
                        if r_idx_main > 0 and isinstance(df.iloc[r_idx_main - 1, c_idx_main], str) and \
                                specific_triangle_keyword in df.iloc[r_idx_main - 1, c_idx_main]:
                            # Check if triangle_type_keyword is above that
                            if r_idx_main > 1 and isinstance(df.iloc[r_idx_main - 2, c_idx_main], str) and \
                                    triangle_type_keyword in df.iloc[r_idx_main - 2, c_idx_main]:
                                accident_month_loc = (r_idx_main, c_idx_main)
                                break
                            # Check one more row up for type keyword for slight variations
                            elif r_idx_main > 2 and isinstance(df.iloc[r_idx_main - 3, c_idx_main], str) and \
                                    triangle_type_keyword in df.iloc[r_idx_main - 3, c_idx_main]:
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
        header_col_idx = accident_month_loc[1]  # Column index of "事故月份"

        dev_periods = []
        if dev_period_header_row < df.shape[0]:
            for c in range(header_col_idx + 1, df.shape[1]):  # Start from column after "事故月份"
                val = df.iloc[dev_period_header_row, c]
                try:
                    # Attempt to convert to float first for robustness, then to int
                    float_val = float(val)
                    if float_val.is_integer():  # Ensure it's a whole number
                        dev_periods.append(int(float_val))
                    else:  # If it's a float but not an integer, it's likely not a dev period
                        break
                except (ValueError, TypeError):
                    # If conversion fails, it's the end of dev periods
                    break

        num_dev_periods = len(dev_periods)
        if num_dev_periods == 0:
            print("No development periods found or parsed correctly under '事故月份'. Check header row.")
            return None

        triangle_data = []
        accident_years = []  # Changed from "accident_months" to "accident_years" for clarity

        if data_start_row < df.shape[0]:
            for r in range(data_start_row, df.shape[0]):
                acc_year_val = df.iloc[r, header_col_idx]  # This is the "Accident Year/Month" column

                # Stop condition: if the AY cell is empty OR if it's a "合计" or "Total" row
                if pd.isna(acc_year_val) or str(acc_year_val).strip() == "":
                    # Check if this empty AY cell row is actually a "合计" row by looking at a preceding column
                    potential_sum_label_col = header_col_idx - 1 if header_col_idx > 0 else header_col_idx
                    if potential_sum_label_col >= 0 and isinstance(df.iloc[r, potential_sum_label_col], str) and \
                            ("合计" in df.iloc[r, potential_sum_label_col] or "Total" in df.iloc[
                                r, potential_sum_label_col]):
                        break  # End of data block
                    # If the entire data part of the row is also empty, then it's likely the end
                    if df.iloc[r, header_col_idx + 1: header_col_idx + 1 + num_dev_periods].isnull().all():
                        if len(accident_years) > 0:  # if we have already collected some data
                            break
                        else:  # if it's an empty row before any data, skip it
                            continue

                # If AY cell has "合计" or "Total", stop
                if isinstance(acc_year_val, str) and ("合计" in acc_year_val or "Total" in acc_year_val):
                    break

                try:
                    str_acc_year = str(acc_year_val).strip()
                    # Basic validation for an accident year/period label (starts with digit or common AY prefixes)
                    # And check if the first data cell is numeric-like or NaN (not another string header)
                    first_data_cell_val = df.iloc[r, header_col_idx + 1]
                    try:
                        pd.to_numeric(first_data_cell_val)
                        is_numeric_check = True
                    except (ValueError, TypeError):
                        is_numeric_check = pd.isna(first_data_cell_val)  # Allow NaN as a valid data point start

                    if not (str_acc_year and (str_acc_year[0].isdigit() or \
                                              str_acc_year.lower().startswith('ay') or \
                                              str_acc_year.lower().startswith('u')) and \
                            is_numeric_check):  # U for UMS
                        if len(accident_years) > 0:  # If we already have data, this different format row means end
                            break
                        else:  # Otherwise, skip this non-data row
                            continue

                    accident_years.append(str_acc_year)
                    row_data = df.iloc[r, header_col_idx + 1: header_col_idx + 1 + num_dev_periods].tolist()
                    numeric_row_data = [pd.to_numeric(val, errors='coerce') for val in row_data]
                    triangle_data.append(numeric_row_data)
                except Exception:  # Broad exception if row processing fails
                    # If an error occurs, and we've added an AY for this row, remove it as the data is incomplete/invalid
                    if len(accident_years) > 0 and str(accident_years[-1]) == str_acc_year:
                        accident_years.pop()
                    break  # Stop processing on error for this triangle

        if not triangle_data:
            print("No data extracted for the triangle.")
            return None

        # Create DataFrame
        extracted_df = pd.DataFrame(triangle_data, index=pd.Index(accident_years, name="AccidentYear"),
                                    columns=dev_periods)

        # Clean up: drop rows/columns that are entirely NaN (can happen if extraction grabs too far)
        extracted_df = extracted_df.dropna(axis=1, how='all').dropna(axis=0, how='all')

        return extracted_df
    except Exception as e:
        print(f"An error occurred during triangle extraction: {e}")
        import traceback
        traceback.print_exc()
        return None


def cumulative_to_incremental(cumulative_triangle):
    """
    Converts a cumulative triangle to an incremental one.
    (代码与您之前提供的一致，此处省略)
    """
    incremental_triangle = cumulative_triangle.copy()
    # Ensure columns are pandas Index for proper subtraction if they are not already
    # This function assumes columns are ordered by development period
    for col_idx in range(1, incremental_triangle.shape[1]):
        current_col_name = incremental_triangle.columns[col_idx]
        prev_col_name = incremental_triangle.columns[col_idx - 1]
        # Ensure subtraction is done on numeric types, NaNs should propagate
        incremental_triangle[current_col_name] = pd.to_numeric(cumulative_triangle[current_col_name], errors='coerce') - \
                                                 pd.to_numeric(cumulative_triangle[prev_col_name], errors='coerce')
    return incremental_triangle


def calculate_weighted_ldfs(cumulative_triangle, trim_extremes=True):
    """
    Calculates weighted LDFs.
    (代码与您之前提供的一致，此处省略)
    """
    n_rows, n_cols = cumulative_triangle.shape
    ldfs_final_selection = []
    triangle_columns = cumulative_triangle.columns  # Use actual column names

    for j in range(n_cols - 1):  # Iterate up to the second to last column
        col1_name = triangle_columns[j]
        col2_name = triangle_columns[j + 1]

        col1_vals_all = cumulative_triangle[col1_name]
        col2_vals_all = cumulative_triangle[col2_name]

        current_period_ldfs_data = []
        # LDFs are calculated for n_rows - (j+1) pairs for development period j to j+1
        for i in range(n_rows - (j + 1)):
            val1 = col1_vals_all.iloc[i]
            val2 = col2_vals_all.iloc[i]

            if pd.notna(val1) and pd.notna(val2) and val1 != 0:
                current_period_ldfs_data.append({'ldf': val2 / val1, 'weight': val1})
            elif pd.notna(val1) and pd.notna(val2) and val1 == 0 and val2 == 0:
                # Both zero, development is stable, LDF is 1.0. Low weight.
                current_period_ldfs_data.append({'ldf': 1.0, 'weight': 1e-9})
            elif pd.notna(val1) and pd.notna(val2) and val1 == 0 and val2 != 0:
                # Developed from zero to non-zero, high LDF. Low weight.
                current_period_ldfs_data.append({'ldf': 999.0, 'weight': 1e-9})
            # If val1 is non-zero and val2 is NaN, or val1 is NaN, the pair is ignored (no LDF)

        if not current_period_ldfs_data:
            ldfs_final_selection.append(1.0)  # Default LDF if no data
            continue

        ldf_values = [d['ldf'] for d in current_period_ldfs_data]
        weights = [d['weight'] for d in current_period_ldfs_data]

        if trim_extremes and len(ldf_values) >= 5:  # Trim only if enough data points
            # Sort by LDF value to find indices of min/max
            sorted_indices = np.argsort(ldf_values)
            # Keep all indices except the first (min LDF) and last (max LDF)
            indices_to_keep = sorted_indices[1:-1]

            trimmed_ldf_values = [ldf_values[i] for i in indices_to_keep]
            trimmed_weights = [weights[i] for i in indices_to_keep]

            if not trimmed_ldf_values:  # If trimming left no values (e.g., all LDFs were same)
                # Fallback to simple average of original or default to 1.0
                sum_original_weights = sum(weights)
                if sum_original_weights == 0:
                    avg_ldf = np.mean(ldf_values) if ldf_values else 1.0
                else:
                    numerator = sum(ldf * w for ldf, w in zip(ldf_values, weights))
                    avg_ldf = numerator / sum_original_weights
            else:
                sum_trimmed_weights = sum(trimmed_weights)
                if sum_trimmed_weights == 0:
                    avg_ldf = np.mean(trimmed_ldf_values) if trimmed_ldf_values else 1.0
                else:
                    numerator = sum(ldf * w for ldf, w in zip(trimmed_ldf_values, trimmed_weights))
                    avg_ldf = numerator / sum_trimmed_weights
        else:  # Not trimming
            total_weight = sum(weights)
            if total_weight == 0:  # Should ideally not happen if there's data
                avg_ldf = np.mean(ldf_values) if ldf_values else 1.0
            else:
                numerator = sum(ldf * w for ldf, w in zip(ldf_values, weights))
                avg_ldf = numerator / total_weight

        ldfs_final_selection.append(avg_ldf if pd.notna(avg_ldf) else 1.0)

    # Tail Factor Calculation (using last 3 LDFs if available)
    # Ensure LDFs used for tail are reasonable (e.g. > 0.1 to avoid distortion from near-zero LDFs)
    valid_ldfs_for_tail_calc = [ldf for ldf in ldfs_final_selection[-3:] if pd.notna(ldf) and ldf > 0.1]
    if len(valid_ldfs_for_tail_calc) > 0:
        tail_factor = np.mean(valid_ldfs_for_tail_calc)
    else:  # Fallback if no valid LDFs for tail (e.g. all are 1.0 or less, or NaN)
        tail_factor = 1.0

    tail_factor = max(1.0, tail_factor)  # Tail factor should not be less than 1.0
    ldfs_final_selection.append(tail_factor)

    return np.array(ldfs_final_selection)


def project_triangle(cumulative_triangle, ldfs):
    """
    Projects a cumulative triangle to ultimate using LDFs.
    (代码与您之前提供的一致，此处省略，但会确保其与CoC逻辑兼容)
    """
    projected_triangle_df = cumulative_triangle.copy().astype(float)
    n_rows, n_cols_orig = projected_triangle_df.shape
    num_ldfs_available = len(ldfs)  # This includes the tail factor as the last LDF

    # Determine maximum development periods needed based on LDFs
    # The number of age-to-age LDFs is num_ldfs_available - 1.
    # These project up to n_cols_orig + (num_ldfs_available - 1 - n_cols_orig) = num_ldfs_available -1 development periods
    # So, the ultimate column would effectively be the (num_ldfs_available)-th period if we started from 0.
    # If LDFs imply more periods than original, extend the DataFrame

    original_cols_are_int = all(isinstance(c, int) for c in cumulative_triangle.columns)
    max_dev_period_covered_by_ldfs = n_cols_orig + (
                num_ldfs_available - 1 - (n_cols_orig - 1))  # This simplifies to num_ldfs_available
    # if we consider LDFs map one-to-one to future periods

    # The LDF array ldfs has length k. ldfs[0] takes col 0 to 1, ..., ldfs[k-2] takes col k-2 to k-1. ldfs[k-1] is tail.
    # So, k-1 LDFs define transitions between k development periods.
    # The "Ultimate" means after all k-1 development periods defined by LDFs (excluding tail applied separately)
    # are completed + tail application.

    current_max_col_name_numeric = 0
    if original_cols_are_int and len(projected_triangle_df.columns) > 0:
        current_max_col_name_numeric = max([col for col in projected_triangle_df.columns if isinstance(col, int)])

    # Extend columns if LDFs suggest more development periods than currently in triangle
    # num_ldfs_available - 1 is the number of explicit age-to-age factors
    # These factors will project the triangle up to n_cols_orig + (num_ldfs_available - 1 - (n_cols_orig-1)) = num_ldfs_available development periods
    # So, we need columns up to index num_ldfs_available - 1 if 0-indexed
    # or named up to num_ldfs_available if 1-indexed.

    # Let's assume columns are 1-indexed for display based on your dev_periods from extraction
    # If original columns are [1, 2, 3], n_cols_orig = 3.
    # If num_ldfs_available = 5 (4 age-to-age, 1 tail). We need to project up to Dev Period 4.
    # ldfs[0] (1->2), ldfs[1] (2->3), ldfs[2] (3->4), ldfs[3] (4->Ult, this is the tail ldfs[num_ldfs_available-1])

    # Number of development columns we need to project *through* using age-to-age LDFs
    # is num_ldfs_available - 1.
    # So the final development column before ultimate will be related to this.

    # Ensure columns for all development periods implied by LDFs exist
    # The ldfs array has k elements. ldfs[0] maps dev period 0 to 1, ... ldfs[k-2] maps k-2 to k-1. ldfs[k-1] is tail.
    # So there are k-1 explicit development periods.
    # If original cols are Dev1, Dev2 (indices 0, 1), n_cols_orig = 2.
    # If ldfs has length 4 (Dev1->2, Dev2->3, Dev3->4, Tail for >4). We need cols for Dev1, Dev2, Dev3, Dev4.
    # The column names are from `cumulative_triangle.columns`.

    target_num_dev_cols = num_ldfs_available  # Number of states, so Dev1 to Dev(num_ldfs_available)
    # if tail is considered to bring to ultimate *from* last explicit dev period
    # Or num_ldfs_available-1 if tail applies *to* the last dev period

    # Simpler: `ldfs` has `k` elements. `ldfs[0]` takes col `j` to `j+1`. `ldfs[n_cols_orig-1]` takes `n_cols_orig-1` to `n_cols_orig`.
    # So we need to ensure columns exist up to what `ldfs[num_ldfs_available-2]` projects *to*.

    # The number of columns the projected triangle should have (excluding 'Ultimate')
    # is `num_ldfs_available -1` if we interpret LDFs as mapping from col i to col i+1
    # and the last LDF is tail.
    # If LDFs are ldf_12, ldf_23, ldf_34, ldf_tail. num_ldfs_available = 4.
    # We need dev cols 1, 2, 3, 4.
    # Let required_dev_cols be num_ldfs_available. (e.g. 4 dev periods means LDFs for 1->2, 2->3, 3->4, and then tail 4->Ult)

    # Let's use the column names from original triangle and extend if necessary
    dev_col_names = list(cumulative_triangle.columns)

    # Number of development periods we need to project through using *age-to-age* LDFs
    # is (num_ldfs_available - 1). If n_cols_orig is 3, and (num_ldfs_available - 1) is 5,
    # we need to add 2 more development columns.
    num_age_to_age_ldfs = num_ldfs_available - 1

    if num_age_to_age_ldfs > n_cols_orig:  # We need more dev columns than original
        last_orig_col_name = dev_col_names[-1]
        for i in range(n_cols_orig, num_age_to_age_ldfs):
            if original_cols_are_int:
                new_col_name = last_orig_col_name + (i - (n_cols_orig - 1))
            else:
                new_col_name = f"DevExt_{i + 1}"  # Generic name
            if new_col_name not in projected_triangle_df.columns:
                projected_triangle_df[new_col_name] = np.nan
                dev_col_names.append(new_col_name)

    if 'Ultimate' not in projected_triangle_df.columns:
        projected_triangle_df['Ultimate'] = np.nan

    # Use the potentially extended list of dev_col_names
    current_dev_cols_in_df = [col for col in projected_triangle_df.columns if col != 'Ultimate']

    for i in range(n_rows):  # For each accident year
        latest_known_cumulative = np.nan
        last_known_dev_col_idx = -1  # 0-indexed within current_dev_cols_in_df

        # Find the last known cumulative payment for this AY in the original part
        for j_orig in range(n_cols_orig):
            original_col_name = cumulative_triangle.columns[j_orig]
            if pd.notna(projected_triangle_df.loc[projected_triangle_df.index[i], original_col_name]):
                latest_known_cumulative = projected_triangle_df.loc[projected_triangle_df.index[i], original_col_name]
                # Find this original_col_name's index in the current_dev_cols_in_df
                try:
                    last_known_dev_col_idx = current_dev_cols_in_df.index(original_col_name)
                except ValueError:  # Should not happen if logic is correct
                    last_known_dev_col_idx = j_orig  # Fallback
            else:  # Data ends for this AY before n_cols_orig
                # If it's NaN from the start, this AY might be all zeros or start later
                if j_orig == 0 and pd.isna(latest_known_cumulative):
                    projected_triangle_df.iloc[i, :len(current_dev_cols_in_df)] = 0  # Fill dev periods with 0
                    latest_known_cumulative = 0
                    # last_known_dev_col_idx remains -1 or is set to where 0 is, effectively making it fully projected
                    # For simplicity, if AY starts with NaN, treat as 0 and project fully.
                    # The projection loop starts from last_known_dev_col_idx + 1.
                    # If all are zero, it will be projected as zero.
                    # To ensure it's projected if it's all NaNs:
                    if projected_triangle_df.iloc[i, :n_cols_orig].isnull().all():
                        projected_triangle_df.iloc[i, :n_cols_orig] = 0.0  # Fill NaNs with 0
                        latest_known_cumulative = 0.0
                        last_known_dev_col_idx = n_cols_orig - 1  # Pretend last original column was 0
                break

        if pd.isna(latest_known_cumulative):  # Should be 0 if all were NaN
            projected_triangle_df.loc[projected_triangle_df.index[i], 'Ultimate'] = 0
            continue

        # Project forward from the last known cumulative amount
        projected_val = latest_known_cumulative

        # Iterate through LDFs to project future development periods
        # last_known_dev_col_idx is the index in current_dev_cols_in_df
        # ldfs index corresponds to the dev period *from which* we are developing
        # e.g., ldfs[k] develops from period k to period k+1.
        # If last_known_dev_col_idx is 1 (i.e. Dev Period 2, if 0-indexed means actual col index 1),
        # we need to apply ldfs[1] to project to Dev Period 3 (col index 2).

        # Project through the defined age-to-age LDFs
        # num_age_to_age_ldfs = num_ldfs_available - 1
        # current_dev_cols_in_df now has at least num_age_to_age_ldfs columns

        for k in range(last_known_dev_col_idx + 1, len(current_dev_cols_in_df)):
            # The LDF to use is ldfs[k-1] if last_known_dev_col_idx was correct start
            # Or more simply, ldf_idx_to_apply is the index of the column we are projecting FROM
            ldf_idx_to_apply = k - 1  # This is the ldf that takes us from col k-1 to col k
            # if k is 0-indexed current_dev_cols_in_df

            if ldf_idx_to_apply < (num_ldfs_available - 1):  # ensure we use an age-to-age LDF
                ldf = ldfs[ldf_idx_to_apply]
                ldf = max(1.0, ldf) if pd.notna(ldf) else 1.0  # LDFs should not be less than 1
                projected_val *= ldf
                projected_triangle_df.loc[projected_triangle_df.index[i], current_dev_cols_in_df[k]] = projected_val
            else:  # Should not happen if columns align with LDFs before tail
                break

                # Apply tail factor (the last LDF in the ldfs array)
        tail_ldf = ldfs[num_ldfs_available - 1]
        tail_ldf = max(1.0, tail_ldf) if pd.notna(tail_ldf) else 1.0
        ultimate_val = projected_val * tail_ldf
        projected_triangle_df.loc[projected_triangle_df.index[i], 'Ultimate'] = ultimate_val

    return projected_triangle_df


# --- 新增和修改的函数，用于资本成本法 ---

def _get_average_payout_pattern_factors(
        reference_projected_cumulative_triangle_with_ultimate,
        cumulative_actual_triangle,
        ref_ibnr_total_mean
):
    """
    根据参考的最佳估计预测三角和实际已付数据，推导平均的未来支付模式因子。
    """
    if ref_ibnr_total_mean <= 1e-6:  # 如果参考IBNR几乎为零
        return np.array([])

    # 1. 从参考预测中获取未来增量支付
    ref_dev_columns = [col for col in reference_projected_cumulative_triangle_with_ultimate.columns if
                       col != 'Ultimate']
    if not ref_dev_columns:
        return np.array([1.0])  # 如果没有展开列，假设一期付清

    ref_cumulative_dev_periods = reference_projected_cumulative_triangle_with_ultimate[ref_dev_columns]
    ref_incremental_full_triangle = cumulative_to_incremental(ref_cumulative_dev_periods.copy())

    n_rows_actual, n_cols_actual = cumulative_actual_triangle.shape
    ref_n_rows, ref_n_cols_projected = ref_incremental_full_triangle.shape

    # 确定未来支付序列的最大长度
    # 这应该是参考增量三角的最大展开期数
    max_future_periods = ref_n_cols_projected
    average_future_incremental_payments_by_period = np.zeros(max_future_periods)

    for r in range(ref_n_rows):  # 遍历每一个出险年 (AY)
        if r >= n_rows_actual:  # 如果参考三角中的AY超出了实际数据的AY（不太可能，除非参考三角是外部的）
            last_observed_col_for_ay = -1  # 假设这个AY的所有支付都是未来的
        else:
            last_observed_col_for_ay = -1  # 0-indexed
            non_na_indices = np.where(pd.notna(cumulative_actual_triangle.iloc[r, :n_cols_actual]))[0]
            if len(non_na_indices) > 0:
                last_observed_col_for_ay = non_na_indices[-1]

        for c_proj in range(ref_n_cols_projected):  # 遍历参考增量三角的每一个展开期
            # 如果当前展开期 c_proj > 该AY的最后一个已观测展开期，则为未来支付
            if c_proj > last_observed_col_for_ay:
                payment = ref_incremental_full_triangle.iloc[r, c_proj]
                if pd.notna(payment) and payment > 0:  # 只考虑正的未来支付
                    # future_period_idx: 0代表紧随最后一个观测期之后的第一期未来支付
                    future_period_idx = c_proj - (last_observed_col_for_ay + 1)
                    if 0 <= future_period_idx < max_future_periods:
                        average_future_incremental_payments_by_period[future_period_idx] += payment

    # 清理尾部：移除支付额非常小的期数，并确保总支付额与 ref_ibnr_total_mean 大致匹配
    # (或者直接使用到所有有支付的期间)
    actual_sum_avg_payments = np.sum(average_future_incremental_payments_by_period)

    # 截断到实际有支付的最后一个期间
    if actual_sum_avg_payments > 1e-6:
        last_positive_payment_idx = np.where(average_future_incremental_payments_by_period > 1e-6)[0]
        if len(last_positive_payment_idx) > 0:
            average_future_incremental_payments_by_period = average_future_incremental_payments_by_period[
                                                            :last_positive_payment_idx[-1] + 1]
        else:  # 没有显著的正支付
            average_future_incremental_payments_by_period = np.array(
                [ref_ibnr_total_mean]) if ref_ibnr_total_mean > 1e-6 else np.array([])
    elif ref_ibnr_total_mean > 1e-6:  # IBNR存在，但计算出的序列支付为0
        average_future_incremental_payments_by_period = np.array([ref_ibnr_total_mean])  # 假设一期付清
    else:  # IBNR为0
        average_future_incremental_payments_by_period = np.array([])

    # 3. 标准化得到支付模式因子
    sum_avg_future_payments = np.sum(average_future_incremental_payments_by_period)
    if sum_avg_future_payments > 1e-6:
        average_payout_pattern_factors = average_future_incremental_payments_by_period / sum_avg_future_payments
        # 修正由于截断或加总逻辑可能导致的总和不完全等于1的情况
        average_payout_pattern_factors = average_payout_pattern_factors / np.sum(average_payout_pattern_factors)
    elif ref_ibnr_total_mean > 1e-6 and len(average_future_incremental_payments_by_period) == 1:  # IBNR存在且只有一期支付
        average_payout_pattern_factors = np.array([1.0])
    else:  # 无IBNR或无有效支付模式
        average_payout_pattern_factors = np.array([])

    return average_payout_pattern_factors


def _get_sim_payout_schedule(sim_total_ibnr, avg_pattern_factors):
    """
    根据单次模拟的总IBNR和平均支付模式因子，生成该模拟的未来支付计划。
    返回一个列表，每个元素是一个字典，包含:
    'period_index', 'payment', 'outstanding_start', 'outstanding_end'
    """
    payout_schedule = []
    outstanding_at_start_of_period = sim_total_ibnr

    if sim_total_ibnr < 1e-6:  # 如果IBNR本身就很小
        return payout_schedule

    if not avg_pattern_factors.any():  # 如果支付模式为空
        # 假设一期付清
        payout_schedule.append({
            'period_index': 0,
            'payment': sim_total_ibnr,
            'outstanding_start': outstanding_at_start_of_period,
            'outstanding_end': 0.0
        })
        return payout_schedule

    for i, factor in enumerate(avg_pattern_factors):
        if outstanding_at_start_of_period < 1e-6:  # 基本上付清了
            break

        payment_this_period = sim_total_ibnr * factor

        # 确保支付额不超过当期期初的未决赔款额（特别是在最后几期，由于因子加总可能不精确为1）
        payment_this_period = min(payment_this_period, outstanding_at_start_of_period)
        # 如果是最后一个支付因子，确保付清所有剩余的 (处理因子总和不为1的情况)
        if i == len(avg_pattern_factors) - 1:
            payment_this_period = outstanding_at_start_of_period

        outstanding_at_end_of_period = outstanding_at_start_of_period - payment_this_period

        payout_schedule.append({
            'period_index': i,  # 0-索引代表未来第一个期间
            'payment': payment_this_period,
            'outstanding_start': outstanding_at_start_of_period,
            'outstanding_end': outstanding_at_end_of_period
        })
        outstanding_at_start_of_period = max(0, outstanding_at_end_of_period)  # 确保非负

    return payout_schedule


def calculate_ra_coc(  # 重命名函数以区分
        triangle_df,
        n_simulations,
        ra_percentile,  # 仍可用于计算参考的百分位IBNR
        cost_of_capital_rate,  # 新增CoC参数
        discount_rate,  # 新增CoC参数
        capital_percentage  # 新增CoC参数
):
    if triangle_df is None or triangle_df.empty:
        print("输入三角数据为空或None。无法计算RA。")
        return None, None, None, None, None, None, None, None  # 调整返回值

    cumulative_actual_triangle = triangle_df.apply(pd.to_numeric, errors='coerce').astype(float)
    if cumulative_actual_triangle.isnull().all().all():
        print("错误: 三角数据在数值转换后全为NaN。请检查数据。")
        return None, None, None, None, None, None, None, None

    # 1. 计算平均LDFs (与之前一致)
    selected_ldfs = calculate_weighted_ldfs(cumulative_actual_triangle.copy(), trim_extremes=True)
    print(f"选定的LDFs (包含尾部因子): {selected_ldfs}")
    if selected_ldfs is None or len(selected_ldfs) == 0 or np.all(np.isnan(selected_ldfs)):
        print("错误: 所有LDFs为NaN或LDF数组为空/None。请检查数据质量或LDF计算。")
        return None, None, None, None, None, None, None, None

    # 2. 计算参考的最佳估计预测三角 (用于推导平均支付模式)
    reference_projected_triangle_full = project_triangle(cumulative_actual_triangle.copy(), selected_ldfs)
    if reference_projected_triangle_full is None or reference_projected_triangle_full.empty:
        print("错误: 计算参考预测三角失败。")
        return None, None, None, None, None, None, None, None

    # 3. 计算当前已付赔款总额 (对角线之和)
    current_latest_paid = 0
    for r_idx in range(cumulative_actual_triangle.shape[0]):
        row_data = cumulative_actual_triangle.iloc[r_idx, :].dropna()
        if not row_data.empty:
            current_latest_paid += row_data.iloc[-1]
    if pd.isna(current_latest_paid): current_latest_paid = 0

    # 4. 计算参考的最佳估计总IBNR
    ref_ibnr_total_mean_estimate = reference_projected_triangle_full['Ultimate'].sum() - current_latest_paid
    ref_ibnr_total_mean_estimate = max(0, ref_ibnr_total_mean_estimate)
    print(f"参考的最佳估计总IBNR: {ref_ibnr_total_mean_estimate:,.2f}")

    # 5. 推导平均支付模式因子 (在模拟循环之前计算一次)
    average_payout_pattern_factors = _get_average_payout_pattern_factors(
        reference_projected_triangle_full,
        cumulative_actual_triangle,
        ref_ibnr_total_mean_estimate
    )
    print(f"平均支付模式因子 (未来各期支付占比): {average_payout_pattern_factors}")
    if not average_payout_pattern_factors.any() and ref_ibnr_total_mean_estimate > 1e-6:
        print("警告: 未能生成有效的平均支付模式因子，但参考IBNR大于0。CoC RA可能不准确。")

    # --- 拟合与残差计算 (与之前一致) ---
    n_rows, n_cols = cumulative_actual_triangle.shape
    fitted_observed_cumulative_triangle = pd.DataFrame(np.nan, index=cumulative_actual_triangle.index,
                                                       columns=cumulative_actual_triangle.columns)
    for r in range(n_rows):
        if pd.notna(cumulative_actual_triangle.iloc[r, 0]):
            fitted_observed_cumulative_triangle.iloc[r, 0] = cumulative_actual_triangle.iloc[r, 0]
            current_actual_cum_for_fitting = cumulative_actual_triangle.iloc[r, 0]
            for c in range(1, n_cols):  # 对于后续的展开期
                # LDF的索引是 c-1 (例如，从展开期0到1的LDF是selected_ldfs[0])
                if pd.notna(current_actual_cum_for_fitting) and (c - 1) < (len(selected_ldfs) - 1):  # -1因为最后一个LDF是尾部因子
                    ldf_to_apply = selected_ldfs[c - 1]
                    if pd.isna(ldf_to_apply): ldf_to_apply = 1.0

                    fitted_value = current_actual_cum_for_fitting * ldf_to_apply
                    fitted_observed_cumulative_triangle.iloc[r, c] = fitted_value

                    # 更新用于下一期拟合的基准：如果实际观测存在，用实际的；否则，停止该AY的拟合
                    if pd.notna(cumulative_actual_triangle.iloc[r, c]):
                        current_actual_cum_for_fitting = cumulative_actual_triangle.iloc[r, c]
                    else:  # 如果实际观测中断，则拟合也中断
                        current_actual_cum_for_fitting = np.nan  # 或者 break，取决于如何处理部分观测行
                        break
                else:  # 如果上一期的累积额是NaN，或者没有对应的LDF了
                    break

    true_incremental_fitted_triangle = cumulative_to_incremental(fitted_observed_cumulative_triangle)
    incremental_actual_triangle = cumulative_to_incremental(cumulative_actual_triangle)
    residuals_inc = incremental_actual_triangle - true_incremental_fitted_triangle

    scaling_denominator_sqrt = pd.DataFrame(1.0, index=cumulative_actual_triangle.index,
                                            columns=cumulative_actual_triangle.columns)
    if n_cols > 1:
        actual_cumul_at_interval_start = cumulative_actual_triangle.iloc[:, :-1].copy()  # 取所有行，除了最后一列的所有列
        # 对于第一列的增量，其“期初”是0，但通常我们不为第一列标准化或其残差就是0
        # 这里要确保 actual_cumul_at_interval_start 的值不为负，并处理0的情况
        actual_cumul_at_interval_start[actual_cumul_at_interval_start < 0] = 0  # 将负值视为0
        scaling_denominator_sqrt.iloc[:, 1:] = np.sqrt(np.maximum(actual_cumul_at_interval_start.values, 1e-9))

    standardized_residuals_matrix = residuals_inc / scaling_denominator_sqrt

    print("\n--- 用于残差抽样的拟合增量三角 ---")
    print(true_incremental_fitted_triangle.to_string(float_format="%.2f"))
    print("\n--- 标准化残差 ---")
    standardized_residuals_df = pd.DataFrame(standardized_residuals_matrix.values,
                                             index=cumulative_actual_triangle.index,
                                             columns=cumulative_actual_triangle.columns)
    print(standardized_residuals_df.to_string(float_format="%.2f"))

    pool_of_residuals = []
    for r_idx in range(n_rows):
        for c_idx in range(n_cols):
            # 确保该单元格可以计算残差：即实际累积和拟合累积都存在
            # 并且标准化因子有效，标准化残差也有效
            if pd.notna(cumulative_actual_triangle.iloc[r_idx, c_idx]) and \
                    pd.notna(true_incremental_fitted_triangle.iloc[r_idx, c_idx]) and \
                    pd.notna(standardized_residuals_matrix.iloc[r_idx, c_idx]) and \
                    np.isfinite(standardized_residuals_matrix.iloc[r_idx, c_idx]):
                # 添加条件：期初累积（用于标准化的分母）不能太小，避免极端残差
                if c_idx == 0 or (c_idx > 0 and pd.notna(cumulative_actual_triangle.iloc[r_idx, c_idx - 1]) and
                                  cumulative_actual_triangle.iloc[r_idx, c_idx - 1] > 1):  # 阈值可调
                    pool_of_residuals.append(standardized_residuals_matrix.iloc[r_idx, c_idx])

    if not pool_of_residuals:
        print("错误: 残差池为空。可能所有期初累积值都很小或数据问题。使用虚拟残差池 [0]。")
        pool_of_residuals = [0.0]
    pool_of_residuals = np.array(pool_of_residuals)
    print(
        f"\n残差池大小: {len(pool_of_residuals)}, 均值: {np.mean(pool_of_residuals):.4f}, 标准差: {np.std(pool_of_residuals):.4f}")

    # --- Bootstrap 模拟循环 ---
    simulated_ibnrs = []
    simulated_coc_ras = []  # 存储每次模拟的CoC RA

    for sim_num in range(n_simulations):
        sim_incremental_triangle = true_incremental_fitted_triangle.copy()

        # 对可以应用残差的单元格进行抽样和去标准化
        # (与之前逻辑类似，但确保只在有有效拟合值的地方应用)
        cells_for_residual_application = []
        for r_sim in range(n_rows):
            for c_sim in range(n_cols):
                if pd.notna(true_incremental_fitted_triangle.iloc[r_sim, c_sim]):  # 只有有拟合值的地方才可能应用残差
                    # 并且实际值也存在（这样才有原始的标准化残差可以参考）
                    # 或者更简单，只要有拟合值，就抽样一个残差
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
                sqrt_val_for_destd = scaling_denominator_sqrt.iloc[r_sim, c_sim]

                sim_payment_inc = fitted_inc_val + sampled_std_residual * sqrt_val_for_destd
                sim_incremental_triangle.iloc[r_sim, c_sim] = max(0, sim_payment_inc)  # 确保非负
            # else: No residual to apply if pool was smaller than cells, keep fitted

        # 将模拟的增量三角转换为累积三角
        sim_cumulative_triangle = sim_incremental_triangle.cumsum(axis=1)
        # 填补由于早期残差导致后续NaN的情况 (如果cumsum后仍有NaN)
        sim_cumulative_triangle = sim_cumulative_triangle.ffill(axis=1).fillna(0)

        # 使用选定的LDFs将模拟的累积三角预测到最终
        sim_ultimate_triangle_full = project_triangle(sim_cumulative_triangle.copy(), selected_ldfs)
        if sim_ultimate_triangle_full is None or sim_ultimate_triangle_full.empty:
            simulated_ibnrs.append(0)
            simulated_coc_ras.append(0)
            if (sim_num + 1) % (n_simulations // 10 if n_simulations >= 10 else 1) == 0:
                print(f"模拟 {sim_num + 1}/{n_simulations} 完成 (预测失败)")
            continue

        sim_ultimate_claims_sum = sim_ultimate_triangle_full['Ultimate'].sum()
        if pd.isna(sim_ultimate_claims_sum): sim_ultimate_claims_sum = 0

        # 计算本次模拟的总IBNR
        current_sim_total_ibnr = sim_ultimate_claims_sum - current_latest_paid
        current_sim_total_ibnr = max(0, current_sim_total_ibnr)
        simulated_ibnrs.append(current_sim_total_ibnr)

        # --- CoC RA 计算 ---
        payout_schedule_this_sim = _get_sim_payout_schedule(current_sim_total_ibnr, average_payout_pattern_factors)

        sim_specific_coc_ra_value = 0
        if current_sim_total_ibnr > 1e-6 and not payout_schedule_this_sim:  # IBNR存在但没有支付计划
            # 如果没有支付模式但IBNR存在，CoC RA可能为0或需要特殊处理
            # 这里假设如果无法分配支付，则CoC为0，但这可能低估
            pass  # sim_specific_coc_ra_value remains 0

        for period_data in payout_schedule_this_sim:
            outstanding_at_period_start = period_data['outstanding_start']
            period_idx = period_data['period_index']  # 0-indexed

            if outstanding_at_period_start < 1e-6:
                continue

            required_capital = outstanding_at_period_start * capital_percentage
            cost_for_capital_this_period = required_capital * cost_of_capital_rate

            pv_cost = cost_for_capital_this_period / ((1 + discount_rate) ** (period_idx + 1))
            sim_specific_coc_ra_value += pv_cost

        simulated_coc_ras.append(sim_specific_coc_ra_value)

        if (sim_num + 1) % (n_simulations // 20 if n_simulations >= 20 else 1) == 0:  # 减少打印频率
            print(
                f"模拟 {sim_num + 1}/{n_simulations} 完成. SimIBNR: {current_sim_total_ibnr:,.0f}, SimCoCRA: {sim_specific_coc_ra_value:,.0f}")

    # --- 汇总结果 ---
    mean_ibnr_from_sims = np.mean(simulated_ibnrs) if simulated_ibnrs else np.nan
    final_coc_ra = np.mean(simulated_coc_ras) if simulated_coc_ras else np.nan

    # 计算参考的百分位IBNR (基于ra_percentile)
    percentile_based_ibnr_val = np.nan
    percentile_based_ra_val = np.nan
    if simulated_ibnrs and ra_percentile is not None:
        sorted_ibnrs = sorted(simulated_ibnrs)
        ra_idx = int(len(sorted_ibnrs) * ra_percentile)  # ra_index should be 0 to N-1
        ra_idx = min(max(0, ra_idx), len(sorted_ibnrs) - 1)
        if ra_idx < len(sorted_ibnrs):
            percentile_based_ibnr_val = sorted_ibnrs[ra_idx]
            percentile_based_ra_val = percentile_based_ibnr_val - mean_ibnr_from_sims
    # --- 新增：计算 CoC RA 分布的指定百分位值 ---
    coc_ra_percentiles_to_calculate = [75, 80, 85, 90, 95, 99, 99.5]
    coc_ra_percentile_values = {}
    if simulated_coc_ras and len(simulated_coc_ras) > 0:  # 确保列表非空
        # np.percentile可以直接处理未排序列表，但如果需要排序版本也可先排序
        # sorted_coc_ras = sorted(simulated_coc_ras)
        for p_value in coc_ra_percentiles_to_calculate:
            coc_ra_percentile_values[p_value] = np.percentile(simulated_coc_ras, p_value)
    # --- 新增结束 ---
    return (mean_ibnr_from_sims, final_coc_ra,
            percentile_based_ibnr_val, percentile_based_ra_val,  # 新增百分位结果
            simulated_ibnrs, simulated_coc_ras,  # 返回分布
            reference_projected_triangle_full,  # 用于展示的最佳估计预测三角
            residuals_inc, standardized_residuals_df,coc_ra_percentile_values)  # 返回残差用于分析


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)

    # --- CoC 参数设定 ---
    COC_RATE = 0.06  # 资本成本率
    DISCOUNT_RATE = 0.02  # 贴现率 (无风险利率)
    CAPITAL_PERCENTAGE = 0.20  # 所需资本占期初未决的百分比
    RA_PERCENTILE_FOR_COMPARISON = 0.75  # 用于对比的IBNR百分位

    # --- 文件和工作表设定 ---
    EXCEL_FILE_PATH = "202412三角形.xlsx"  # 您的Excel文件名
    SHEET_NAME_TO_PROCESS = "秋粮"  # 您想要处理的工作表名称

    try:
        print(f"尝试从Excel文件 '{EXCEL_FILE_PATH}' 的工作表 '{SHEET_NAME_TO_PROCESS}' 加载数据...")
        df_sheet_data = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME_TO_PROCESS, header=None, dtype=str)
        print(f"成功加载工作表 '{SHEET_NAME_TO_PROCESS}' 从 '{EXCEL_FILE_PATH}'")

        # 假设关键词适用于您Excel工作表的结构
        extracted_triangle = extract_triangle_from_df(df_sheet_data, "再保前", "累计已决三角形")

        if extracted_triangle is not None and not extracted_triangle.empty:
            print(f"\n成功从 '{SHEET_NAME_TO_PROCESS}' 工作表中提取三角表:")
            # ... (后续的数据处理、调用 calculate_ra_coc 和结果打印逻辑与之前CoC完整代码中的一致) ...
            # VVVVVV 从这里开始，与上一份CoC完整代码的 main 部分相同 VVVVVV
            for col in extracted_triangle.columns:
                extracted_triangle[col] = pd.to_numeric(extracted_triangle[col], errors='coerce')
            extracted_triangle = extracted_triangle.dropna(axis=0, how='all').dropna(axis=1, how='all')

            if extracted_triangle.empty:
                print("三角表在数值转换和NaN移除后变为空。请检查数据。")
            else:
                print(f"\n处理中的三角表 '{SHEET_NAME_TO_PROCESS}' (行列数: {extracted_triangle.shape}):")
                print(extracted_triangle.to_string(float_format="%.0f"))

                print("\n开始CoC风险边际计算...")
                num_sims = 1000  # 实际使用时建议 5000 或 10000
                print(
                    f"参数: CoC利率={COC_RATE}, 贴现率={DISCOUNT_RATE}, 资本比例={CAPITAL_PERCENTAGE}, 模拟次数={num_sims}")

                (mean_ibnr, coc_ra,
                 pctl_ibnr, pctl_ra,
                 ibnr_dist, coc_ra_dist,
                 ref_proj_tri,
                 raw_res, std_res,
                 coc_ra_pctl_values  # 接收CoC RA的百分位值字典
                 ) = calculate_ra_coc(
                    extracted_triangle,
                    n_simulations=num_sims,
                    ra_percentile=RA_PERCENTILE_FOR_COMPARISON,  # 用于对比的IBNR百分位
                    cost_of_capital_rate=COC_RATE,
                    discount_rate=DISCOUNT_RATE,
                    capital_percentage=CAPITAL_PERCENTAGE
                )

                if coc_ra is not None:
                    print(f"\n--- 精算结果 for '{SHEET_NAME_TO_PROCESS}' ---")
                    print(f"预估的均值 IBNR (参考): {mean_ibnr:,.0f}")
                    print(f"资本成本法风险边际 (CoC RA): {coc_ra:,.0f}")
                    if coc_ra_dist is not None and len(coc_ra_dist) > 0:
                        print(f"  (基于 {len(coc_ra_dist)} 次模拟CoC RA的均值)")
                    else:
                        print("  (未能生成CoC RA分布)")
                        # --- 新增：打印CoC RA的百分位值 ---
                    if coc_ra_pctl_values:
                        print(f"\n--- CoC RA 分布的百分位值 (基于 {len(coc_ra_dist)} 次模拟) ---")
                        for pctl, val in coc_ra_pctl_values.items():
                            print(f"CoC RA 在 {pctl}% 百分位: {val:,.0f}")
                    # --- 新增结束 ---
                    print(f"对比: {RA_PERCENTILE_FOR_COMPARISON * 100:.0f}th 百分位IBNR: {pctl_ibnr:,.0f}")
                    print(
                        f"对比: 基于百分位的RA ({RA_PERCENTILE_FOR_COMPARISON * 100:.0f}th Pctl IBNR - Mean IBNR): {pctl_ra:,.0f}")
                    # --- 新增：计算并打印 CoC RA 对应的隐含IBNR置信水平 ---
                    if ibnr_dist and len(ibnr_dist) > 0:  # 确保 ibnr_dist 有数据
                        target_ibnr_provision = mean_ibnr + coc_ra
                        count_less_equal_target = sum(1 for x in ibnr_dist if x <= target_ibnr_provision)
                        total_simulations_for_ibnr = len(ibnr_dist)  # 使用ibnr_dist的长度

                        implied_confidence_level = (count_less_equal_target / total_simulations_for_ibnr) * 100
                        print(f"\nCoC RA ({coc_ra:,.0f}) 对应的隐含IBNR置信水平: {implied_confidence_level:.2f}%")
                        print(
                            f"  (即，{implied_confidence_level:.2f}% 的概率下，实际IBNR不高于 {target_ibnr_provision:,.0f})")
                    else:
                        print("\n未能计算隐含IBNR置信水平，因为IBNR模拟分布为空或计算失败。")
                    # --- 新增结束 ---
                    # ... (绘图和打印参考三角数据的逻辑与之前CoC完整代码中的一致) ...
                    if coc_ra_dist and np.std(coc_ra_dist) > 1e-6:  # 仅当有意义的分布时绘图
                        plt.figure(figsize=(18, 6))

                        plt.subplot(1, 3, 1)
                        if ibnr_dist:  # 检查 ibnr_dist 是否有数据
                            plt.hist(ibnr_dist, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
                            plt.axvline(mean_ibnr, color='red', linestyle='dashed', linewidth=1,
                                        label=f'均值 IBNR: {mean_ibnr:,.0f}')
                            plt.axvline(pctl_ibnr, color='purple', linestyle='dashed', linewidth=1,
                                        label=f'{RA_PERCENTILE_FOR_COMPARISON * 100:.0f}th Pctl IBNR: {pctl_ibnr:,.0f}')
                        plt.title(f'模拟IBNR分布 ({SHEET_NAME_TO_PROCESS})')
                        plt.xlabel('模拟IBNR值')
                        plt.ylabel('频数')
                        plt.legend()
                        plt.ticklabel_format(style='plain', axis='x')

                        plt.subplot(1, 3, 2)
                        if ibnr_dist:  # 检查 ibnr_dist 是否有数据
                            sns.kdeplot(ibnr_dist, fill=True, color="olive", linewidth=2, alpha=0.7)
                            plt.axvline(mean_ibnr, color='red', linestyle='dashed', linewidth=1)
                            plt.axvline(pctl_ibnr, color='purple', linestyle='dashed', linewidth=1)
                        plt.title(f'模拟IBNR核密度估计 ({SHEET_NAME_TO_PROCESS})')
                        plt.xlabel('模拟IBNR值')
                        plt.ylabel('密度')
                        plt.ticklabel_format(style='plain', axis='x')

                        plt.subplot(1, 3, 3)
                        plt.hist(coc_ra_dist, bins=50, color='salmon', edgecolor='black', alpha=0.7)
                        plt.title(f'模拟CoC RA分布 ({SHEET_NAME_TO_PROCESS})')
                        plt.xlabel('模拟CoC RA值')
                        plt.ylabel('频数')
                        plt.axvline(coc_ra, color='blue', linestyle='dashed', linewidth=1,
                                    label=f'最终 CoC RA (均值): {coc_ra:,.0f}')
                        plt.legend()
                        plt.ticklabel_format(style='plain', axis='x')

                        plt.tight_layout()
                        plot_filename = f"coc_ra_ibnr_distribution_{SHEET_NAME_TO_PROCESS.replace(' ', '_')}.png"
                        plt.savefig(plot_filename)
                        print(f"\n分布图已保存为 '{plot_filename}'")
                        # plt.show()
                    elif coc_ra_dist:
                        print("\n由于模拟CoC RA值的方差过小或为零，未生成其分布图。")
                    else:
                        print("\n未生成CoC RA分布图，因为没有模拟出CoC RA值。")

                    print("\n参考的最佳估计完整累积三角数据 (包含Ultimate):")
                    if ref_proj_tri is not None:
                        formatted_ref_triangle = ref_proj_tri.copy()
                        for c_col in formatted_ref_triangle.columns:
                            if formatted_ref_triangle[c_col].dtype in [np.int64, np.float64, int, float]:
                                formatted_ref_triangle[c_col] = formatted_ref_triangle[c_col].map('{:,.0f}'.format)
                        print(formatted_ref_triangle.to_string())
                    else:
                        print("参考预测三角数据为None。")
                else:
                    print("\nCoC RA 计算失败或返回None值。")
        else:
            print(f"\n无法从工作表 '{SHEET_NAME_TO_PROCESS}' 中提取三角表。")

    except FileNotFoundError:
        print(
            f"错误: Excel 文件 '{EXCEL_FILE_PATH}' 未找到。请确保文件路径正确，并且您已安装 'openpyxl' (pip install openpyxl)。")
    except ValueError as ve:
        if "Worksheet" in str(ve) and SHEET_NAME_TO_PROCESS in str(ve):  # 更精确的错误捕获
            print(f"错误: 工作表 '{SHEET_NAME_TO_PROCESS}' 在Excel文件 '{EXCEL_FILE_PATH}' 中未找到。")
        else:
            print(f"发生数值或参数错误: {ve}")
            import traceback

            traceback.print_exc()
    except ImportError:
        print("错误: 读取Excel文件需要 'openpyxl' 库。请运行 'pip install openpyxl' 进行安装。")
    except Exception as e:
        print(f"发生未预期错误: {e}")
        import traceback

        traceback.print_exc()