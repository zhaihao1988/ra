# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime

# --- 全局设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# --- 核心精算函数 ---

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
    """累积三角形转增量三角形"""
    incremental_triangle = cumulative_triangle.copy()
    for col_idx in range(1, incremental_triangle.shape[1]):
        current_col_name = incremental_triangle.columns[col_idx]
        prev_col_name = incremental_triangle.columns[col_idx - 1]
        incremental_triangle[current_col_name] = cumulative_triangle[current_col_name] - cumulative_triangle[
            prev_col_name]
    return incremental_triangle


def calculate_weighted_ldfs_from_ra_lic(cumulative_triangle, trim_extremes=True):
    """
    计算加权赔款发展因子(LDF)。
    !! 此函数逻辑完全复刻自最初的 ra_lic.py !!
    """
    n_rows, n_cols = cumulative_triangle.shape
    ldfs_final_selection = []
    triangle_columns = cumulative_triangle.columns

    for j in range(n_cols - 1):
        col1_vals_all = cumulative_triangle[triangle_columns[j]]
        col2_vals_all = cumulative_triangle[triangle_columns[j + 1]]
        current_period_ldfs_data = []
        for i in range(n_rows - (j + 1)):
            val1, val2 = col1_vals_all.iloc[i], col2_vals_all.iloc[i]
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
            if not trimmed_ldf_values or sum(trimmed_weights) == 0:
                avg_ldf = np.average(ldf_values, weights=weights) if sum(weights) > 0 else np.mean(ldf_values)
            else:
                numerator = sum(ldf * w for ldf, w in zip(trimmed_ldf_values, trimmed_weights))
                avg_ldf = numerator / sum(trimmed_weights)
        else:
            avg_ldf = np.average(ldf_values, weights=weights) if sum(weights) > 0 else np.mean(ldf_values)

        ldfs_final_selection.append(avg_ldf if pd.notna(avg_ldf) else 1.0)

    valid_ldfs_for_tail_calc = [ldf for ldf in ldfs_final_selection[-3:] if pd.notna(ldf) and ldf > 0.1]
    tail_factor = np.mean(valid_ldfs_for_tail_calc) if valid_ldfs_for_tail_calc else 1.0
    ldfs_final_selection.append(max(1.0, tail_factor))
    return np.array(ldfs_final_selection)


def project_triangle(cumulative_triangle, ldfs):
    """(逻辑来自 ra_lic.py)"""
    projected_triangle_df = cumulative_triangle.copy().astype(float)
    n_rows, n_cols_orig = projected_triangle_df.shape
    num_ldfs_available = len(ldfs)

    if 'Ultimate' not in projected_triangle_df.columns:
        max_dev_periods_from_ldfs = num_ldfs_available
        original_cols_are_int = all(isinstance(c, int) for c in cumulative_triangle.columns)
        for dev_col_idx in range(n_cols_orig, max_dev_periods_from_ldfs):
            if original_cols_are_int and cumulative_triangle.columns.max() < dev_col_idx + 1:
                new_col_name = dev_col_idx + 1
            else:
                new_col_name = f"Dev_{dev_col_idx + 1}" if not original_cols_are_int else dev_col_idx + 1
            if new_col_name not in projected_triangle_df.columns:
                projected_triangle_df[new_col_name] = np.nan
        projected_triangle_df['Ultimate'] = np.nan

    dev_period_cols = [col for col in projected_triangle_df.columns if col != 'Ultimate']
    num_dev_period_cols_total = len(dev_period_cols)

    for i in range(n_rows):
        current_cumulative_val = np.nan
        last_known_col_idx_in_orig = -1
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
        current_col_being_projected_from_idx = last_known_col_idx_in_orig

        for k_target_dev_col_idx in range(last_known_col_idx_in_orig + 1, num_dev_period_cols_total):
            if current_col_being_projected_from_idx < num_ldfs_available - 1:
                ldf_to_apply = ldfs[current_col_being_projected_from_idx]
                ldf_to_apply = max(1.0, ldf_to_apply) if pd.notna(ldf_to_apply) else 1.0
                temp_val_for_ultimate *= ldf_to_apply
                projected_triangle_df.iloc[i, k_target_dev_col_idx] = temp_val_for_ultimate
                current_col_being_projected_from_idx += 1
            else:
                break

        tail_ldf = ldfs[num_ldfs_available - 1]
        tail_ldf = max(1.0, tail_ldf) if pd.notna(tail_ldf) else 1.0
        ultimate_val = temp_val_for_ultimate * tail_ldf
        projected_triangle_df.loc[projected_triangle_df.index[i], 'Ultimate'] = ultimate_val
    return projected_triangle_df



def calculate_ra_coc_with_ra_lic_fitting(triangle_df, n_simulations, scr_percentile, coc_rate, discount_rate):
    """
    资本成本法主函数，但使用 ra_lic.py 的逻辑进行模型拟合。
    """
    # 1. 数据准备与模型拟合
    cumulative_actual_triangle = triangle_df.apply(pd.to_numeric, errors='coerce').astype(float)

    # !! 步骤 1: 使用 ra_lic.py 的LDF计算逻辑 !!
    selected_ldfs = calculate_weighted_ldfs_from_ra_lic(cumulative_actual_triangle.copy(), trim_extremes=True)
    logging.info(f"选定的LDFs (来自ra_lic.py的逻辑): \n{np.round(selected_ldfs, 4)}")

    n_rows, n_cols = cumulative_actual_triangle.shape

    # !! 步骤 2: 使用 ra_lic.py 的“重新锚定”方式来拟合三角形 !!
    fitted_cum_tri = pd.DataFrame(np.nan, index=triangle_df.index, columns=triangle_df.columns)
    for r in range(n_rows):
        if pd.notna(cumulative_actual_triangle.iloc[r, 0]):
            fitted_cum_tri.iloc[r, 0] = cumulative_actual_triangle.iloc[r, 0]
            # 使用一个变量来跟踪用于拟合的值，并在每个实际观测点进行“锚定”
            current_val_for_fitting = cumulative_actual_triangle.iloc[r, 0]
            for c in range(1, n_cols):
                if pd.notna(current_val_for_fitting) and (c - 1) < len(selected_ldfs):
                    ldf = max(1.0, selected_ldfs[c - 1])
                    fitted_cum_tri.iloc[r, c] = current_val_for_fitting * ldf
                    # 锚定到下一个已知的实际值
                    if pd.notna(cumulative_actual_triangle.iloc[r, c]):
                        current_val_for_fitting = cumulative_actual_triangle.iloc[r, c]
                    else:  # 如果下一个是未知值，则继续用拟合值传递
                        current_val_for_fitting = fitted_cum_tri.iloc[r, c]

    # 步骤 3: 基于上述拟合结果计算残差
    fitted_inc_tri = cumulative_to_incremental(fitted_cum_tri)
    actual_inc_tri = cumulative_to_incremental(cumulative_actual_triangle)
    residuals = actual_inc_tri - fitted_inc_tri
    scaling_denom = np.sqrt(np.maximum(cumulative_actual_triangle.shift(1, axis=1).fillna(0), 1e-9))
    std_residuals = residuals / scaling_denom
    pool_of_residuals = std_residuals.values[~np.isnan(std_residuals.values)]
    logging.info(f"已构建残差池 (基于ra_lic.py的拟合逻辑)，大小: {len(pool_of_residuals)}")
    if not pool_of_residuals.any(): pool_of_residuals = np.array([0.0])

    # 2. 全路径模拟 (资本成本法逻辑)
    all_sim_outstanding_paths, all_sim_total_ibnr = [], []
    for _ in range(n_simulations):
        sampled_std_res = np.random.choice(pool_of_residuals, size=std_residuals.shape, replace=True)
        sim_inc_tri = fitted_inc_tri + pd.DataFrame(sampled_std_res * scaling_denom.values, index=fitted_inc_tri.index,
                                                    columns=fitted_inc_tri.columns)
        sim_inc_tri[sim_inc_tri < 0] = 0
        sim_inc_tri = sim_inc_tri.where(pd.notna(cumulative_actual_triangle))
        sim_cum_tri = sim_inc_tri.cumsum(axis=1).ffill(axis=1).fillna(0)

        full_sim_tri = project_triangle(sim_cum_tri.copy(), selected_ldfs)

        # 提取未来支付路径
        future_payments_by_ay = []
        for r in range(n_rows):
            last_observed_idx = n_cols - (n_rows - r) - 1
            if last_observed_idx < full_sim_tri.shape[1] - 2:
                # 提取未来增量值
                future_increments = np.diff(full_sim_tri.iloc[r, last_observed_idx + 1:].values)
                future_increments = np.insert(future_increments, 0,
                                              full_sim_tri.iloc[r, last_observed_idx + 1] - full_sim_tri.iloc[
                                                  r, last_observed_idx])
                future_payments_by_ay.append(future_increments)

        max_future_len = max(len(p) for p in future_payments_by_ay) if future_payments_by_ay else 0
        total_future_payments = np.zeros(max_future_len)
        for p in future_payments_by_ay:
            total_future_payments[:len(p)] += p

        all_sim_total_ibnr.append(total_future_payments.sum())
        all_sim_outstanding_paths.append(np.flip(np.flip(total_future_payments).cumsum()))

    # 3. 逐期计算 BEL, VaR, SCR (资本成本法逻辑)
    max_len = max(len(p) for p in all_sim_outstanding_paths) if all_sim_outstanding_paths else 0
    outstanding_matrix = np.array([np.pad(p, (0, max_len - len(p))) for p in all_sim_outstanding_paths])
    bel_path = np.mean(outstanding_matrix, axis=0)
    var_path = np.percentile(outstanding_matrix, scr_percentile * 100, axis=0)
    scr_path = np.maximum(0, var_path - bel_path)

    # 4. 计算最终RA (资本成本法逻辑)
    time_periods = np.arange(1, len(scr_path) + 1)
    discount_factors = (1 + discount_rate) ** time_periods
    final_ra = np.sum((scr_path * coc_rate) / discount_factors)

    # 5. 反算隐含置信水平
    mean_ibnr_total = np.mean(all_sim_total_ibnr)
    total_provision = mean_ibnr_total + final_ra
    implied_confidence = np.sum(all_sim_total_ibnr <= total_provision) / n_simulations

    return final_ra, mean_ibnr_total, implied_confidence, all_sim_total_ibnr, scr_path


# --- 主执行框架 ---
if __name__ == '__main__':
    EXCEL_FILE_PATH = "202412三角形.xlsx"
    SHEET_NAMES_TO_PROCESS = ["非融资性保证险", "融资性保证险", "财产险", "车其他", "车损", "健康险", "交强险", "奶牛",
                              "能繁母猪", "秋粮", "商业三者", "夏粮", "养殖险其他", "意外险", "育肥猪", "种植险特色"]
    COC_RATE, DISCOUNT_RATE, SCR_PERCENTILE, NUM_SIMULATIONS = 0.06, 0.02, 0.995, 1000

    output_dir = f"RA_Results_CoC_ReplicatedFit_{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(output_dir, exist_ok=True)
    summary_results_list = []

    for sheet_name in SHEET_NAMES_TO_PROCESS:
        log_filename = os.path.join(output_dir, f"{sheet_name}_资本成本法_复刻拟合.log")
        logger = logging.getLogger(sheet_name)
        logger.setLevel(logging.INFO)
        if logger.hasHandlers(): logger.handlers.clear()
        fh = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

        logger.info(f"--- 开始处理险种: {sheet_name} (资本成本法, 但使用ra_lic.py的拟合逻辑) ---")
        try:
            df_sheet_data = pd.read_excel(EXCEL_FILE_PATH, sheet_name=sheet_name, header=None, dtype=str)
            triangle = extract_triangle_from_df(df_sheet_data, "再保前", "累计已决三角形")
            if triangle is None or triangle.empty: logger.error("未能提取到有效三角表，跳过。"); continue

            (ra, bel, implied_p, ibnr_dist, scr_term) = calculate_ra_coc_with_ra_lic_fitting(
                triangle, NUM_SIMULATIONS, SCR_PERCENTILE, COC_RATE, DISCOUNT_RATE
            )
            if ra is None: logger.error("核心计算失败，跳过。"); continue

            logger.info(f"\n最优估计 IBNR (BEL): {bel:,.0f} | 风险边际 (RA): {ra:,.0f} | 隐含置信水平: {implied_p:.2%}")
            summary_results_list.append(
                {"LineOfBusiness": sheet_name, "Method": "CostOfCapital_ra_lic_Fit", "Mean_IBNR_BEL": bel,
                 "RiskAdjustment_RA": ra, "ImpliedConfidenceLevel": implied_p})

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            sns.histplot(ibnr_dist, bins=50, kde=True, ax=ax1, color='teal')
            ax1.set_title(f'{sheet_name} - IBNR 模拟分布');
            ax1.legend()
            ax2.bar(range(len(scr_term)), scr_term, color='coral')
            ax2.set_title(f'{sheet_name} - SCR 期限结构')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{sheet_name}_资本成本法_复刻拟合.png"));
            plt.close()

        except Exception as e:
            logger.error(f"处理险种 '{sheet_name}' 时发生未知错误: {e}", exc_info=True)

    if summary_results_list:
        summary_df = pd.DataFrame(summary_results_list)
        summary_filename = os.path.join(output_dir, "RA_Summary_Results_CoC_ReplicatedFit.csv")
        summary_df.to_csv(summary_filename, index=False, encoding='utf-8-sig')
        print(f"\n所有计算完成！总摘要文件已保存至: {summary_filename}")