import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback


# --- 核心函数 PART 1: 完全复刻自 ra_lic.py (无任何修改) ---

def extract_triangle_from_df(df, triangle_type_keyword, specific_triangle_keyword):
    """(函数来源: ra_lic.py)"""
    # ... 此处代码与上一版完全相同，为简洁省略 ...
    try:
        paid_keyword_rows = df[
            df.apply(lambda r: r.astype(str).str.contains(specific_triangle_keyword).any(), axis=1)].index
        if not paid_keyword_rows.any():
            print(f"关键词 '{specific_triangle_keyword}' 未找到。")
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
            print(f"无法在 '{triangle_type_keyword}' 和 '{specific_triangle_keyword}' 下定位 '事故月份'。")
            return None

        dev_period_header_row, data_start_row, header_col_idx = accident_month_loc[0] + 1, accident_month_loc[0] + 2, \
        accident_month_loc[1]
        dev_periods = []
        if dev_period_header_row < df.shape[0]:
            for c in range(header_col_idx + 1, df.shape[1]):
                try:
                    dev_periods.append(int(float(df.iloc[dev_period_header_row, c])))
                except (ValueError, TypeError):
                    break
        if not dev_periods: print("未找到有效的展开期。"); return None

        triangle_data, accident_years = [], []
        if data_start_row < df.shape[0]:
            for r in range(data_start_row, df.shape[0]):
                acc_year_val = df.iloc[r, header_col_idx]
                if pd.isna(acc_year_val) or "合计" in str(acc_year_val): break
                try:
                    pd.to_numeric(df.iloc[r, header_col_idx + 1])
                    accident_years.append(str(acc_year_val).strip())
                    row_data = df.iloc[r, header_col_idx + 1: header_col_idx + 1 + len(dev_periods)].tolist()
                    triangle_data.append([pd.to_numeric(val, errors='coerce') for val in row_data])
                except (ValueError, TypeError):
                    if len(accident_years) > len(triangle_data): accident_years.pop()
                    break
        if not triangle_data: print("未提取到任何三角表数据。"); return None

        return pd.DataFrame(triangle_data, index=pd.Index(accident_years, name="AccidentYear"),
                            columns=dev_periods).dropna(how='all', axis=1).dropna(how='all', axis=0)
    except Exception as e:
        print(f"提取三角表时发生错误: {e}");
        traceback.print_exc();
        return None


def cumulative_to_incremental(cumulative_triangle):
    """(函数来源: ra_lic.py)"""
    incremental_triangle = cumulative_triangle.copy()
    for col_idx in range(1, incremental_triangle.shape[1]):
        current_col_name, prev_col_name = incremental_triangle.columns[col_idx], incremental_triangle.columns[
            col_idx - 1]
        incremental_triangle[current_col_name] = cumulative_triangle[current_col_name] - cumulative_triangle[
            prev_col_name]
    return incremental_triangle


def calculate_weighted_ldfs(cumulative_triangle, trim_extremes=True):
    """(函数来源: ra_lic.py)"""
    # ... 此处代码与上一版完全相同，为简洁省略 ...
    n_rows, n_cols = cumulative_triangle.shape
    ldfs_final_selection = []
    for j in range(n_cols - 1):
        col1_vals_all, col2_vals_all = cumulative_triangle.iloc[:, j], cumulative_triangle.iloc[:, j + 1]
        current_period_ldfs_data = []
        for i in range(n_rows - (j + 1)):
            val1, val2 = col1_vals_all.iloc[i], col2_vals_all.iloc[i]
            if pd.notna(val1) and pd.notna(val2) and val1 != 0:
                current_period_ldfs_data.append({'ldf': val2 / val1, 'weight': val1})
            elif pd.notna(val1) and pd.notna(val2) and val1 == 0 and val2 == 0:
                current_period_ldfs_data.append({'ldf': 1.0, 'weight': 1e-9})
            elif pd.notna(val1) and pd.notna(val2) and val1 == 0 and val2 != 0:
                current_period_ldfs_data.append({'ldf': 999.0, 'weight': 1e-9})
        if not current_period_ldfs_data: ldfs_final_selection.append(1.0); continue

        ldf_values = [d['ldf'] for d in current_period_ldfs_data]
        weights = [d['weight'] for d in current_period_ldfs_data]
        if trim_extremes and len(ldf_values) >= 5:
            sorted_indices = np.argsort(ldf_values)
            indices_to_keep = sorted_indices[1:-1]
            trimmed_ldfs = [ldf_values[i] for i in indices_to_keep]
            trimmed_weights = [weights[i] for i in indices_to_keep]
            if not trimmed_ldfs or sum(trimmed_weights) == 0:
                avg_ldf = np.average(ldf_values, weights=weights) if sum(weights) > 0 else np.mean(ldf_values)
            else:
                avg_ldf = sum(l * w for l, w in zip(trimmed_ldfs, trimmed_weights)) / sum(trimmed_weights)
        else:
            avg_ldf = np.average(ldf_values, weights=weights) if sum(weights) > 0 else np.mean(ldf_values)
        ldfs_final_selection.append(avg_ldf if pd.notna(avg_ldf) else 1.0)

    valid_ldfs_for_tail_calc = [ldf for ldf in ldfs_final_selection[-3:] if pd.notna(ldf) and ldf > 0.1]
    tail_factor = np.mean(valid_ldfs_for_tail_calc) if valid_ldfs_for_tail_calc else 1.0
    ldfs_final_selection.append(max(1.0, tail_factor))
    return np.array(ldfs_final_selection)


def project_triangle(cumulative_triangle, ldfs):
    """(函数来源: ra_lic.py) -- 未做任何修改"""
    projected_triangle = cumulative_triangle.copy().astype(float)
    n_rows, n_cols_orig = projected_triangle.shape
    n_dev_periods = len(ldfs)

    new_cols = [f'Dev_{j + 1}' for j in range(n_cols_orig, n_dev_periods)]
    if new_cols:
        projected_triangle = pd.concat(
            [projected_triangle, pd.DataFrame(columns=new_cols, index=projected_triangle.index)], axis=1)

    for i in range(n_rows):
        last_known_val, last_known_idx = np.nan, -1
        for j in range(n_cols_orig):
            if pd.notna(projected_triangle.iloc[i, j]):
                last_known_val, last_known_idx = projected_triangle.iloc[i, j], j
        if pd.isna(last_known_val): projected_triangle.iloc[i, :] = 0; continue

        current_val = last_known_val
        for j in range(last_known_idx + 1, n_dev_periods):
            ldf = max(1.0, ldfs[j - 1])
            current_val *= ldf
            projected_triangle.iloc[i, j] = current_val

    tail_ldf = max(1.0, ldfs[-1])
    projected_triangle['Ultimate'] = projected_triangle.iloc[:, n_dev_periods - 1] * tail_ldf
    return projected_triangle


# --- 核心函数 PART 2: 资本成本法计算，增加返回和打印功能 ---

def calculate_ra_coc_single_sheet(triangle_df, n_simulations, scr_percentile, coc_rate, discount_rate):
    """
    主计算函数：
    1. 使用 ra_lic.py 的逻辑进行拟合和残差生成。
    2. 使用资本成本法（CoC）进行风险边际计算。
    3. !! 新增 !! 返回最佳估计和不利情景的三角表以供打印。
    """
    cumulative_actual_triangle = triangle_df.apply(pd.to_numeric, errors='coerce').astype(float)
    if cumulative_actual_triangle.isnull().all().all(): print("错误: 三角表数据无效。"); return [None] * 7

    # --- 步骤 1: 完全复刻 ra_lic.py 的前端逻辑 (无修改) ---
    print("\n步骤 1: 使用 ra_lic.py 的逻辑计算LDF和拟合三角形...")
    selected_ldfs = calculate_weighted_ldfs(cumulative_actual_triangle.copy(), trim_extremes=True)
    print(f"选定的LDFs (包含尾部因子): \n{np.round(selected_ldfs, 4)}")

    n_rows, n_cols = cumulative_actual_triangle.shape
    fitted_cumulative_triangle = pd.DataFrame(np.nan, index=triangle_df.index, columns=triangle_df.columns)

    for r in range(n_rows):
        if pd.notna(cumulative_actual_triangle.iloc[r, 0]):
            fitted_cumulative_triangle.iloc[r, 0] = cumulative_actual_triangle.iloc[r, 0]
            current_val_for_fitting = cumulative_actual_triangle.iloc[r, 0]
            for c in range(1, n_cols):
                if pd.notna(current_val_for_fitting) and (c - 1) < len(selected_ldfs) - 1:
                    ldf = max(1.0, selected_ldfs[c - 1])
                    fitted_cumulative_triangle.iloc[r, c] = current_val_for_fitting * ldf
                    if pd.notna(cumulative_actual_triangle.iloc[r, c]):
                        current_val_for_fitting = cumulative_actual_triangle.iloc[r, c]
                    else:
                        current_val_for_fitting = fitted_cumulative_triangle.iloc[r, c]

    fitted_incremental_triangle = cumulative_to_incremental(fitted_cumulative_triangle)
    actual_incremental_triangle = cumulative_to_incremental(cumulative_actual_triangle)
    residuals = actual_incremental_triangle - fitted_incremental_triangle
    scaling_denom = np.sqrt(np.maximum(cumulative_actual_triangle.shift(1, axis=1).fillna(1), 1e-9))
    std_residuals = residuals / scaling_denom
    pool_of_residuals = std_residuals.values[~np.isnan(std_residuals.values)]
    print(f"残差池已生成 (基于ra_lic.py的拟合逻辑)，大小: {len(pool_of_residuals)}")
    if not pool_of_residuals.any(): pool_of_residuals = np.array([0.0])

    # !! 新增 !! 计算最佳估计三角表
    best_estimate_triangle = project_triangle(cumulative_actual_triangle.copy(), selected_ldfs)

    # --- 步骤 2: 执行资本成本法（CoC）的后端逻辑 (增加三角表存储) ---
    print(f"\n步骤 2: 执行资本成本法全路径模拟 (共 {n_simulations} 次)...")
    all_sim_outstanding_paths, all_sim_total_ibnr = [], []
    all_full_sim_triangles = []  # !! 新增 !! 用于存储所有模拟三角表

    for _ in range(n_simulations):
        sampled_std_res = np.random.choice(pool_of_residuals, size=std_residuals.shape, replace=True)
        sim_inc_tri = fitted_incremental_triangle + pd.DataFrame(sampled_std_res * scaling_denom.values,
                                                                 index=fitted_incremental_triangle.index,
                                                                 columns=fitted_incremental_triangle.columns)
        sim_inc_tri[sim_inc_tri < 0] = 0
        sim_inc_tri = sim_inc_tri.where(pd.notna(cumulative_actual_triangle))
        sim_cum_tri = sim_inc_tri.cumsum(axis=1).ffill(axis=1).fillna(0)

        full_sim_tri = project_triangle(sim_cum_tri.copy(), selected_ldfs)
        all_full_sim_triangles.append(full_sim_tri)  # !! 新增 !!

        future_payments_by_ay = []
        for r in range(n_rows):
            last_observed_idx = n_cols - (n_rows - r) - 1
            if last_observed_idx < full_sim_tri.shape[1] - 2:
                future_cum_path = full_sim_tri.iloc[r, last_observed_idx + 1:].values
                future_increments = np.diff(future_cum_path, prepend=full_sim_tri.iloc[r, last_observed_idx])
                future_payments_by_ay.append(future_increments)

        max_future_len = max(len(p) for p in future_payments_by_ay) if future_payments_by_ay else 0
        total_future_payments = np.zeros(max_future_len)
        for p in future_payments_by_ay: total_future_payments[:len(p)] += p

        all_sim_total_ibnr.append(total_future_payments.sum())
        all_sim_outstanding_paths.append(np.flip(np.flip(total_future_payments).cumsum()))

    print("模拟完成，开始计算路径统计量...")
    max_len = max(len(p) for p in all_sim_outstanding_paths) if all_sim_outstanding_paths else 0
    if max_len == 0: print("警告: 未发现未来支付路径。"); return [None] * 7

    outstanding_matrix = np.array([np.pad(p, (0, max_len - len(p))) for p in all_sim_outstanding_paths])
    bel_path = np.mean(outstanding_matrix, axis=0)
    var_path = np.percentile(outstanding_matrix, scr_percentile * 100, axis=0)
    scr_path = np.maximum(0, var_path - bel_path)

    time_periods = np.arange(1, len(scr_path) + 1)
    discount_factors = (1 + discount_rate) ** time_periods
    final_ra = np.sum((scr_path * coc_rate) / discount_factors)

    mean_ibnr_total = np.mean(all_sim_total_ibnr)
    total_provision = mean_ibnr_total + final_ra
    implied_confidence = np.sum(np.array(all_sim_total_ibnr) <= total_provision) / n_simulations

    # !! 新增 !! 查找并确定不利情景三角表
    percentile_ibnr_value = np.percentile(all_sim_total_ibnr, scr_percentile * 100)
    # 找到最接近该分位点值的模拟IBNR的索引
    adverse_scenario_index = (np.abs(np.array(all_sim_total_ibnr) - percentile_ibnr_value)).argmin()
    adverse_scenario_triangle = all_full_sim_triangles[adverse_scenario_index]

    return final_ra, mean_ibnr_total, implied_confidence, ibnr_dist, scr_path, best_estimate_triangle, adverse_scenario_triangle


# --- 主执行框架 ---
if __name__ == '__main__':
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.width', 200)

    # --- 1. 全局参数设定 ---
    EXCEL_FILE_PATH = "202412三角形.xlsx"
    SHEET_NAME = "奶牛"  # <--- 在这里指定要计算的单一险种名称

    COC_RATE, DISCOUNT_RATE, SCR_PERCENTILE, NUM_SIMULATIONS = 0.06, 0.02, 0.995, 1000

    # --- 2. 执行计算 ---
    print(f"--- 开始处理险种: {SHEET_NAME} (资本成本法, 复刻ra_lic.py前端逻辑) ---")
    try:
        df_sheet_data = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME, header=None, dtype=str)
        print(f"成功加载工作表 '{SHEET_NAME}'")

        triangle = extract_triangle_from_df(df_sheet_data, "再保前", "累计已决三角形")

        if triangle is not None and not triangle.empty:
            result = calculate_ra_coc_single_sheet(
                triangle, NUM_SIMULATIONS, SCR_PERCENTILE, COC_RATE, DISCOUNT_RATE
            )
            ra, bel, implied_p, ibnr_dist, scr_term, best_tri, adverse_tri = result

            if ra is not None:
                # --- 3. 打印所有要求的信息 ---
                print("\n" + "=" * 80)
                print("--- 最终计算结果汇总 ---")
                print(f"最优估计 IBNR (BEL): {bel:,.0f}")
                print(f"资本成本法风险边际 (RA): {ra:,.0f}")
                print(f"总准备金 (BEL + RA): {bel + ra:,.0f}")
                print(f"RA 对应的隐含置信水平: {implied_p:.2%}")

                print("\n" + "=" * 80)
                print("--- SCR 期限结构详情 ---")
                for i, scr_val in enumerate(scr_term, 1):
                    print(f"未来第 {i:<2} 年 SCR: {scr_val:,.0f}")

                print("\n" + "=" * 80)
                print("--- 最佳估计预测三角形 ---")
                print(best_tri.to_string(float_format='{:,.0f}'.format))

                print("\n" + "=" * 80)
                print(f"--- {SCR_PERCENTILE:.1%}分位点下的不利情景预测三角形 ---")
                print(adverse_tri.to_string(float_format='{:,.0f}'.format))
                print("=" * 80)

                # --- 4. 绘制图表 ---
                print("\n正在生成结果图表...")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                sns.histplot(ibnr_dist, bins=50, kde=True, ax=ax1, color='teal', label='IBNR分布')
                ax1.axvline(bel, color='red', ls='--', label=f'最优估计 BEL: {bel:,.0f}')
                ax1.axvline(bel + ra, color='purple', ls=':', label=f'总准备金 (BEL+RA): {bel + ra:,.0f}')
                ax1.set_title(f'{SHEET_NAME} - IBNR 模拟分布');
                ax1.legend()
                ax2.bar(range(1, len(scr_term) + 1), scr_term, color='coral')
                ax2.set_title(f'{SHEET_NAME} - SCR 期限结构');
                ax2.set_xlabel("未来年度");
                ax2.set_ylabel("所需资本 (SCR)")
                plt.tight_layout()
                plot_filename = f"{SHEET_NAME}_CoC_Result_With_Details.png"
                plt.savefig(plot_filename)
                plt.show()
                print(f"分析图表已保存至: {plot_filename}")
        else:
            print(f"\n未能从工作表 '{SHEET_NAME}' 提取到有效数据。")

    except FileNotFoundError:
        print(f"错误: Excel文件 '{EXCEL_FILE_PATH}' 未找到。")
    except Exception as e:
        print(f"处理险种 '{SHEET_NAME}' 时发生未知错误: {e}")
        traceback.print_exc()