# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime

# --- 全局设置 ---

# 设置绘图以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# --- 核心精算函数 (完全来自 ra_lic.py 的原始逻辑) ---

def extract_triangle_from_df(df, triangle_type_keyword, specific_triangle_keyword, logger):
    """
    从DataFrame中提取特定的赔款发展三角形。
    (逻辑来自 ra_lic.py, print替换为logger)
    """
    try:
        paid_keyword_rows = df[
            df.apply(lambda r: r.astype(str).str.contains(specific_triangle_keyword).any(), axis=1)].index
        if not paid_keyword_rows.any():
            logger.warning(f"关键词 '{specific_triangle_keyword}' 未找到。")
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
            # Fallback logic from ra_lic.py
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
            logger.error(f"无法在 '{triangle_type_keyword}' 和 '{specific_triangle_keyword}' 下定位 '事故月份'。")
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
            logger.warning("在 '事故月份' 下未找到有效的展开期。")
            return None

        triangle_data = []
        accident_years = []
        if data_start_row < df.shape[0]:
            for r in range(data_start_row, df.shape[0]):
                acc_year_val = df.iloc[r, header_col_idx]
                if pd.isna(acc_year_val) or str(acc_year_val).strip() == "" or "合计" in str(
                        acc_year_val) or "Total" in str(acc_year_val):
                    break
                try:
                    str_acc_year = str(acc_year_val).strip()
                    first_data_cell_val = df.iloc[r, header_col_idx + 1]
                    pd.to_numeric(first_data_cell_val)
                    accident_years.append(str_acc_year)
                    row_data = df.iloc[r, header_col_idx + 1: header_col_idx + 1 + num_dev_periods].tolist()
                    numeric_row_data = [pd.to_numeric(val, errors='coerce') for val in row_data]
                    triangle_data.append(numeric_row_data)
                except (ValueError, TypeError):
                    if len(accident_years) > len(triangle_data): accident_years.pop()
                    break

        if not triangle_data:
            logger.warning("未提取到任何三角表数据。")
            return None

        extracted_df = pd.DataFrame(triangle_data, index=pd.Index(accident_years, name="AccidentYear"),
                                    columns=dev_periods)
        return extracted_df.dropna(axis=1, how='all').dropna(axis=0, how='all')

    except Exception as e:
        logger.error(f"提取三角表时发生错误: {e}", exc_info=True)
        return None


def cumulative_to_incremental(cumulative_triangle):
    """(逻辑来自 ra_lic.py)"""
    incremental_triangle = cumulative_triangle.copy()
    for col_idx in range(1, incremental_triangle.shape[1]):
        current_col_name = incremental_triangle.columns[col_idx]
        prev_col_name = incremental_triangle.columns[col_idx - 1]
        incremental_triangle[current_col_name] = cumulative_triangle[current_col_name] - cumulative_triangle[
            prev_col_name]
    return incremental_triangle


def calculate_weighted_ldfs(cumulative_triangle, trim_extremes=True):
    """(逻辑来自 ra_lic.py)"""
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


def calculate_ra(triangle_df, n_simulations, logger):
    """
    核心计算函数，完全复刻 ra_lic.py 的逻辑。
    返回完整的模拟结果，以便后续计算多个置信水平。
    """
    if triangle_df is None or triangle_df.empty:
        logger.error("输入三角表为空，无法计算RA。")
        return None, None, None, None, None

    cumulative_actual_triangle = triangle_df.apply(pd.to_numeric, errors='coerce').astype(float)
    if cumulative_actual_triangle.isnull().all().all():
        logger.error("三角表在数值转换后全为空值。")
        return None, None, None, None, None

    # --- 1. LDFs and Fitted Triangle (ra_lic.py logic) ---
    selected_ldfs = calculate_weighted_ldfs(cumulative_actual_triangle.copy(), trim_extremes=True)
    logger.info(f"选定的LDFs (包含尾部因子): \n{np.round(selected_ldfs, 4)}")

    n_rows, n_cols = cumulative_actual_triangle.shape
    fitted_observed_cumulative_triangle = pd.DataFrame(np.nan, index=cumulative_actual_triangle.index,
                                                       columns=cumulative_actual_triangle.columns)

    # !!关键的错误逻辑复刻!!
    for r in range(n_rows):
        if pd.notna(cumulative_actual_triangle.iloc[r, 0]):
            fitted_observed_cumulative_triangle.iloc[r, 0] = cumulative_actual_triangle.iloc[r, 0]
            current_actual_cum_for_fitting = cumulative_actual_triangle.iloc[r, 0]  # 使用实际值开始
            for c in range(1, n_cols):
                if pd.notna(current_actual_cum_for_fitting) and (c - 1) < (len(selected_ldfs) - 1):
                    ldf_to_apply = selected_ldfs[c - 1]
                    if pd.isna(ldf_to_apply): ldf_to_apply = 1.0
                    fitted_value = current_actual_cum_for_fitting * ldf_to_apply
                    fitted_observed_cumulative_triangle.iloc[r, c] = fitted_value

                    # !!每次都用实际值重新锚定!!
                    if pd.notna(cumulative_actual_triangle.iloc[r, c]):
                        current_actual_cum_for_fitting = cumulative_actual_triangle.iloc[r, c]
                    else:
                        current_actual_cum_for_fitting = np.nan
                else:
                    break

    # --- 2. Residuals Calculation (ra_lic.py logic) ---
    true_incremental_fitted_triangle = cumulative_to_incremental(fitted_observed_cumulative_triangle)
    incremental_actual_triangle = cumulative_to_incremental(cumulative_actual_triangle)
    residuals_inc = incremental_actual_triangle - true_incremental_fitted_triangle

    scaling_denominator_sqrt = pd.DataFrame(1.0, index=cumulative_actual_triangle.index,
                                            columns=cumulative_actual_triangle.columns)
    if n_cols > 1:
        actual_cumul_at_interval_start = cumulative_actual_triangle.iloc[:, :-1].values
        scaling_denominator_sqrt.iloc[:, 1:] = np.sqrt(np.maximum(actual_cumul_at_interval_start, 1e-9))

    standardized_residuals_matrix = residuals_inc / scaling_denominator_sqrt

    pool_of_residuals = []
    for r_idx in range(n_rows):
        for c_idx in range(n_cols):
            if pd.notna(cumulative_actual_triangle.iloc[r_idx, c_idx]) and (
                    c_idx == 0 or pd.notna(cumulative_actual_triangle.iloc[r_idx, c_idx - 1])):
                if pd.notna(true_incremental_fitted_triangle.iloc[r_idx, c_idx]) and pd.notna(
                        standardized_residuals_matrix.iloc[r_idx, c_idx]) and np.isfinite(
                        standardized_residuals_matrix.iloc[r_idx, c_idx]):
                    pool_of_residuals.append(standardized_residuals_matrix.iloc[r_idx, c_idx])

    if not pool_of_residuals: pool_of_residuals = [0.0]
    pool_of_residuals = np.array(pool_of_residuals)

    # --- 3. Bootstrap Simulation (ra_lic.py logic) ---
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
                if pd.notna(true_incremental_fitted_triangle.iloc[r_sim, c_sim]) and pd.notna(
                        standardized_residuals_matrix.iloc[r_sim, c_sim]) and np.isfinite(
                        standardized_residuals_matrix.iloc[r_sim, c_sim]):
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
                sim_incremental_triangle.iloc[r_sim, c_sim] = max(0, sim_payment_inc)

        sim_cumulative_triangle = sim_incremental_triangle.cumsum(axis=1)
        sim_cumulative_triangle = sim_cumulative_triangle.ffill(axis=1).fillna(0)

        sim_ultimate_triangle_full = project_triangle(sim_cumulative_triangle.copy(), selected_ldfs)
        sim_ultimate_claims_sum = sim_ultimate_triangle_full['Ultimate'].sum()
        if pd.isna(sim_ultimate_claims_sum): sim_ultimate_claims_sum = 0

        ibnr = sim_ultimate_claims_sum - current_latest_paid
        simulated_ibnrs.append(max(0, ibnr))

    # --- 4. Return results ---
    reference_projected_triangle_full = project_triangle(cumulative_actual_triangle.copy(), selected_ldfs)

    return simulated_ibnrs, reference_projected_triangle_full, residuals_inc, standardized_residuals_matrix, selected_ldfs


# --- 主执行框架 ---
if __name__ == '__main__':
    # --- 1. 全局参数设定 ---
    EXCEL_FILE_PATH = "202412三角形.xlsx"
    SHEET_NAMES_TO_PROCESS = [
        "非融资性保证险", "融资性保证险", "财产险", "车其他", "车损",
        "健康险", "交强险", "奶牛", "能繁母猪", "秋粮",
        "商业三者", "夏粮", "养殖险其他", "意外险", "育肥猪", "种植险特色"
    ]
    CONFIDENCE_LEVELS = [0.65, 0.75, 0.85, 0.95, 0.99, 0.995]
    NUM_SIMULATIONS = 5000

    output_dir = f"RA_Calculation_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    summary_results_list = []

    # --- 2. 循环处理所有险种 ---
    for sheet_name in SHEET_NAMES_TO_PROCESS:
        log_filename = os.path.join(output_dir, f"{sheet_name}_calculation.log")
        logger = logging.getLogger(sheet_name)
        logger.setLevel(logging.INFO)
        if logger.hasHandlers(): logger.handlers.clear()

        fh = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        logger.info(f"--- 开始处理险种: {sheet_name} ---")
        logger.info(f"全局参数: 模拟次数={NUM_SIMULATIONS}, 置信水平={CONFIDENCE_LEVELS}")
        logger.info("核心计算逻辑: 完全复刻 ra_lic.py")

        try:
            df_sheet_data = pd.read_excel(EXCEL_FILE_PATH, sheet_name=sheet_name, header=None, dtype=str)
            logger.info(f"成功加载工作表 '{sheet_name}'")

            triangle = extract_triangle_from_df(df_sheet_data, "再保前", "累计已决三角形", logger)

            if triangle is None or triangle.empty:
                logger.error("未能提取到有效的三角表，跳过此险种。")
                continue

            triangle = triangle.apply(pd.to_numeric, errors='coerce')
            triangle = triangle.dropna(axis=0, how='all').dropna(axis=1, how='all')

            if triangle.empty:
                logger.error("三角表在数据清洗后为空，跳过。")
                continue

            logger.info(f"成功提取并清洗三角表，形状: {triangle.shape}")
            logger.info(f"清洗后的三角表数据:\n{triangle.to_string(float_format='{:,.0f}'.format)}")

            # --- C. 执行核心计算 (使用ra_lic.py的逻辑) ---
            (simulated_ibnrs, ref_proj_tri, raw_res, std_res, ldfs) = calculate_ra(
                triangle,
                n_simulations=NUM_SIMULATIONS,
                logger=logger
            )

            if simulated_ibnrs is None:
                logger.error("核心计算函数返回失败，跳过此险种。")
                continue

            # --- D. 基于单次模拟结果，计算所有置信水平 ---
            simulated_ibnrs.sort()
            mean_ibnr = np.mean(simulated_ibnrs)

            logger.info("\n--- 最终计算结果 ---")
            logger.info(f"最优估计 IBNR (模拟均值): {mean_ibnr:,.0f}")
            logger.info("-" * 25)

            for p in CONFIDENCE_LEVELS:
                p_index = int(NUM_SIMULATIONS * p) - 1
                p_index = min(max(0, p_index), NUM_SIMULATIONS - 1)
                ibnr_at_p = simulated_ibnrs[p_index]
                ra_at_p = ibnr_at_p - mean_ibnr

                logger.info(f"置信水平: {p * 100:.1f}%")
                logger.info(f"  - IBNR 在该分位点值: {ibnr_at_p:,.0f}")
                logger.info(f"  - 风险边际 (RA): {ra_at_p:,.0f}")

                summary_results_list.append({
                    "LineOfBusiness": sheet_name,
                    "ConfidenceLevel": p,
                    "Mean_IBNR": mean_ibnr,
                    "IBNR_at_Percentile": ibnr_at_p,
                    "RiskAdjustment": ra_at_p,
                    "IBNR_StdDev": np.std(simulated_ibnrs),
                    "IBNR_CoV": np.std(simulated_ibnrs) / mean_ibnr if mean_ibnr else 0
                })

            # --- E. 绘制并保存图表 ---
            plt.figure(figsize=(10, 6))
            sns.histplot(simulated_ibnrs, bins=50, kde=True, color='skyblue', stat='density')
            plt.axvline(mean_ibnr, color='red', linestyle='--', linewidth=2, label=f'最优估计 (均值): {mean_ibnr:,.0f}')

            colors = plt.cm.viridis(np.linspace(0, 1, len(CONFIDENCE_LEVELS)))
            for i, p in enumerate(CONFIDENCE_LEVELS):
                p_index = int(NUM_SIMULATIONS * p) - 1
                p_index = min(max(0, p_index), NUM_SIMULATIONS - 1)
                val = simulated_ibnrs[p_index]
                plt.axvline(val, color=colors[i], linestyle=':', linewidth=1.5,
                            label=f'{p * 100:.1f}% Pctl: {val:,.0f}')

            plt.title(f'{sheet_name} - IBNR模拟分布 (复刻ra_lic.py逻辑)')
            plt.xlabel('模拟IBNR值')
            plt.ylabel('密度')
            plt.legend(loc='upper right')
            plt.ticklabel_format(style='plain', axis='x')
            plt.grid(True, linestyle='--', alpha=0.6)

            plot_filename = os.path.join(output_dir, f"{sheet_name}_distribution.png")
            plt.savefig(plot_filename)
            plt.close()
            logger.info(f"分析图表已保存至: {plot_filename}")

        except FileNotFoundError:
            logger.error(f"Excel文件 '{EXCEL_FILE_PATH}' 未找到。")
            break
        except Exception as e:
            logger.error(f"处理险种 '{sheet_name}' 时发生未知错误: {e}", exc_info=True)
            continue

    # --- 3. 保存总摘要文件 ---
    if summary_results_list:
        summary_df = pd.DataFrame(summary_results_list)
        summary_filename = os.path.join(output_dir, "RA_Summary_Results_Replicated.csv")
        summary_df.to_csv(summary_filename, index=False, encoding='utf-8-sig')
        print(f"\n所有计算完成！总摘要文件已保存至: {summary_filename}")
    else:
        print("\n计算完成，但未能生成任何有效结果。")