import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import gamma as gamma_function, psi, polygamma
import matplotlib.pyplot as plt
import warnings

# --- 中文字体设置 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题
except Exception as e:
    print(f"中文字体设置警告: {e}. 图表中的中文可能无法正常显示。")
# --- 中文字体设置结束 ---

def get_distribution_mode(dist_obj, params, data_min, data_max):
    # (此函数与上一版本相同，为简洁起见，此处省略，实际代码中应保留)
    def neg_pdf(x, dist, p_args):
        loc = p_args[-2] if len(p_args) >= 2 else 0.0
        scale = p_args[-1] if len(p_args) >= 1 else 1.0
        shape_args = p_args[:-2] if len(p_args) >= 2 else ()
        try:
            if dist.name == 'genextreme' and len(p_args)==3:
                 shape_args = p_args[0]; loc = p_args[1]; scale = p_args[2]
                 return -dist.pdf(x, shape_args, loc=loc, scale=scale)
            elif dist.name == 'norm' and len(p_args)==2:
                loc = p_args[0]; scale = p_args[1]
                return -dist.pdf(x, loc=loc, scale=scale)
            elif dist.name == 't' and len(p_args)==3: # df, loc, scale
                shape_args = p_args[0]; loc = p_args[1]; scale = p_args[2]
                return -dist.pdf(x, shape_args, loc=loc, scale=scale)
            elif dist.name == 'pareto' and len(p_args)==3: # b, loc, scale
                shape_args = p_args[0]; loc = p_args[1]; scale = p_args[2]
                return -dist.pdf(x, shape_args, loc=loc, scale=scale)
            elif len(shape_args) > 0 :
                return -dist.pdf(x, *shape_args, loc=loc, scale=scale)
            else:
                return -dist.pdf(x, loc=loc, scale=scale)
        except:
            return np.inf

    try:
        shape_param_val = params[0] if len(params)>0 else None
        loc_param_val = params[-2] if len(params) >= 2 else 0.0

        if dist_obj.name == 'weibull_min' and shape_param_val is not None and shape_param_val < 1:
            return loc_param_val
        if dist_obj.name == 't':
            return params[1]
        if dist_obj.name == 'pareto':
            b_shape = params[0]
            current_loc = params[1] if len(params) == 3 else 0.0
            current_scale = params[2] if len(params) == 3 else (params[1] if len(params)==2 else params[0])
            if b_shape <=1:
                return current_loc + current_scale
            pass

        search_lower_bound = loc_param_val
        if data_min > loc_param_val :
            search_lower_bound = data_min
        if search_lower_bound <=0 and dist_obj.name in ['lognorm','gamma','weibull_min','pareto']:
             search_lower_bound = 1e-9

        search_min = max(search_lower_bound, data_min * 0.8)
        search_max = data_max * 1.2
        if search_min >= search_max:
            search_min = data_min * 0.8 if data_min > 0 else 1e-9
            search_max = data_max * 1.2 if data_max > search_min else search_min + 1.0

        res = minimize_scalar(neg_pdf, args=(dist_obj, params), method='bounded',
                              bounds=(search_min, search_max) )

        if res.success and np.isfinite(res.x):
            return res.x
        else:
            print(f"警告: 未能通过优化找到分布 {dist_obj.name} 的众数 (res.success={res.success}, x={res.x})。将尝试使用均值。")
            return calculate_distribution_mean(dist_obj, params)
    except Exception as e:
        print(f"警告: 计算分布 {dist_obj.name} 的众数时出错: {e}。将尝试使用均值。")
        return calculate_distribution_mean(dist_obj, params)

def calculate_distribution_mean(dist_obj, params):
    # (此函数与上一版本相同)
    try:
        if dist_obj.name == 'genextreme':
            c, loc, scale = params[0], params[1], params[2]
            if np.abs(c) < 1e-8:
                return loc + scale * np.euler_gamma
            elif c < 1:
                mean_val = dist_obj.mean(*params)
                if np.isnan(mean_val) or np.isinf(mean_val):
                    print(f"警告: Scipy的genextreme.mean()对参数 {params} 返回了 {mean_val}。尝试理论公式。")
                    mean_val_theoretic = loc + scale * (gamma_function(1 - c) - 1) / c
                    if np.isnan(mean_val_theoretic) or np.isinf(mean_val_theoretic):
                        print(f"警告: GEV均值理论公式计算也失败或为nan/inf。参数: c={c}, loc={loc}, scale={scale}")
                        return np.nan
                    return mean_val_theoretic
                return mean_val
            else:
                print(f"警告: GEV分布形状参数 c={c} (>=1)，均值不存在或为无穷。返回np.nan。")
                return np.nan
        elif dist_obj.name == 't':
            df = params[0]
            if df > 1:
                return dist_obj.mean(*params)
            else:
                print(f"警告: t分布自由度 df={df} (<=1)，均值不存在或为无穷。返回np.nan。")
                return np.nan
        elif dist_obj.name == 'pareto':
            b = params[0]
            if b > 1:
                return dist_obj.mean(*params)
            else:
                print(f"警告: Pareto分布形状参数 b={b} (<=1)，均值不存在或为无穷。返回np.nan。")
                return np.nan
        else:
            mean_val = dist_obj.mean(*params)
            if np.isnan(mean_val) or np.isinf(mean_val):
                print(f"警告: 分布 {dist_obj.name} 的均值计算为 {mean_val}。参数: {params}")
                return np.nan
            return mean_val
    except Exception as e:
        print(f"计算分布 {dist_obj.name} 的均值时出错: {e}")
        return np.nan

# 主函数 calculate_lrc_ra_ratio (与上一版本相同，为简洁省略，实际代码中应保留)
def calculate_lrc_ra_ratio(dataset_name,
                           historical_cor_data,
                           expected_cor_basis='fixed',
                           fixed_expected_cor_value=0.95,
                           confidence_level=0.75,
                           manual_dist_choice=None):
    print(f"\n{'='*10} 开始为数据集: [{dataset_name}] 计算LRC RA比率 {'='*10}")
    data = np.array(historical_cor_data)
    if not np.all((data > 0)):
        print("警告: 输入数据包含非正值，某些分布可能无法拟合或行为异常。")

    distributions_to_try = [
        stats.norm, stats.lognorm, stats.gamma,
        stats.weibull_min, stats.genextreme, stats.t, stats.pareto
    ]

    best_fit_distribution = None
    best_aic = np.inf
    best_params = None
    fit_results = []

    print("正在尝试拟合以下分布模型...\n")
    for distribution in distributions_to_try:
        dist_name = distribution.name
        print(f"--- 数据集 [{dataset_name}] - 拟合分布: {dist_name} ---")
        try:
            if dist_name in ['lognorm', 'gamma', 'weibull_min', 'pareto'] and not np.all(data > 0):
                print(f"跳过分布 {dist_name} 因为数据包含非正值或零。")
                fit_results.append({'name': dist_name, 'params': None, 'aic': np.inf,
                                    'ks_stat': np.nan, 'ks_p_value': np.nan, 'log_likelihood': np.nan})
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = distribution.fit(data)

            ll = np.sum(distribution.logpdf(data, *params))
            k = len(params)
            aic = 2 * k - 2 * ll
            D_ks, p_value_ks = stats.kstest(data, dist_name, args=params)

            print(f"参数: {params}")
            print(f"对数似然 (Log-Likelihood): {ll:.4f}")
            print(f"AIC: {aic:.4f}")
            print(f"K-S 检验统计量 D: {D_ks:.4f}, p-value: {p_value_ks:.4f}")

            fit_results.append({
                'name': dist_name, 'params': params, 'aic': aic,
                'ks_stat': D_ks, 'ks_p_value': p_value_ks, 'log_likelihood': ll
            })

            if manual_dist_choice is None:
                if np.isfinite(aic) and aic < best_aic:
                    best_aic = aic
                    best_fit_distribution = distribution
                    best_params = params
            elif dist_name == manual_dist_choice:
                    best_aic = aic
                    best_fit_distribution = distribution
                    best_params = params
        except Exception as e:
            print(f"拟合分布 {dist_name} 失败: {e}")
            fit_results.append({'name': dist_name, 'params': None, 'aic': np.inf,
                                'ks_stat': np.nan, 'ks_p_value': np.nan, 'log_likelihood': np.nan})
        print("-" * 30)


    if manual_dist_choice and best_fit_distribution is None:
        print(f"\n错误：数据集 [{dataset_name}] - 用户指定的手动选择分布 '{manual_dist_choice}' 未能成功拟合或其拟合结果无效。")
        return None
    elif manual_dist_choice and best_fit_distribution is not None:
        selected_params_found = False
        for res in fit_results:
            if res['name'] == manual_dist_choice and res['params'] is not None:
                best_fit_distribution = getattr(stats, manual_dist_choice)
                best_params = res['params']
                best_aic = res['aic']
                selected_params_found = True
                break
        if not selected_params_found:
             print(f"\n错误：数据集 [{dataset_name}] - 用户指定的手动选择分布 '{manual_dist_choice}' 虽然在尝试列表中，但未能成功拟合有效参数。")
             return None
        print(f"\n数据集 [{dataset_name}] - 用户手动选择使用分布: {best_fit_distribution.name}")
        print(f"其估计参数为: {best_params}")
        print(f"其AIC值为: {best_aic:.4f}")

    elif best_fit_distribution is None :
        print(f"\n数据集 [{dataset_name}] - 未能找到合适的分布模型（基于AIC）。")
        return None
    else:
        print(f"\n数据集 [{dataset_name}] - 根据AIC，初步选择的最佳拟合分布为: {best_fit_distribution.name}")
        print(f"其估计参数为: {best_params}")
        print(f"其AIC值为: {best_aic:.4f}")

    fig_title_suffix = f" - 数据集: {dataset_name}"
    plt.figure(figsize=(12, 7))
    plt.hist(data, bins='auto', density=True, alpha=0.7, label='历史综合成本率直方图')
    data_plot_min = min(data) * 0.8 if min(data) > 0 else (min(data) - 0.1 if min(data) <=0 else 0.01)
    data_plot_max = max(data) * 1.2
    current_loc = best_params[-2] if len(best_params) >=2 else 0.0
    if best_fit_distribution.name in ['lognorm','gamma','weibull_min', 'pareto', 'genextreme']:
        if data_plot_min < current_loc :
            data_plot_min = current_loc
        if current_loc <= 0 and best_fit_distribution.name in ['lognorm','gamma','weibull_min', 'pareto']:
            data_plot_min = 1e-9
    if best_fit_distribution.name == 'weibull_min' and len(best_params)>=3 and best_params[0] < 1:
        data_plot_min = best_params[1]

    x_plot = np.linspace(data_plot_min, data_plot_max, 300)
    pdf_fitted = best_fit_distribution.pdf(x_plot, *best_params)
    plt.plot(x_plot, pdf_fitted, 'r-', lw=2, label=f'拟合的 {best_fit_distribution.name} 分布 PDF')
    plt.title(f'历史综合成本率与拟合的 {best_fit_distribution.name} 分布{fig_title_suffix}')
    plt.xlabel('综合成本率')
    plt.ylabel('密度')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    plt.figure(figsize=(7, 7))
    shape_params_for_probplot = ()
    if len(best_params) > 2: shape_params_for_probplot = best_params[:-2]
    if best_fit_distribution.name in ['genextreme', 'weibull_min', 'gamma', 'lognorm', 't', 'pareto']:
        if len(best_params) >=1 :
             shape_params_for_probplot = best_params[0] if len(best_params) ==1 else best_params[:-2] if len(best_params) >2 else best_params[0]
             if best_fit_distribution.name == 't': shape_params_for_probplot = best_params[0]
             elif best_fit_distribution.name == 'pareto': shape_params_for_probplot = best_params[0]
    elif best_fit_distribution.name == 'norm': shape_params_for_probplot = ()

    stats.probplot(data, dist=best_fit_distribution, sparams=shape_params_for_probplot, plot=plt)
    plt.title(f'{best_fit_distribution.name} 分布的 Q-Q 图{fig_title_suffix}')
    plt.grid(True)
    plt.show(block=False)

    cor_at_confidence_level = best_fit_distribution.ppf(confidence_level, *best_params)
    print(f"\n数据集 [{dataset_name}] - 拟合的 {best_fit_distribution.name} 分布在 {confidence_level*100:.0f}% 分位点的综合成本率为: {cor_at_confidence_level:.4f} (即 {cor_at_confidence_level*100:.2f}%)")

    expected_cor_to_use = np.nan
    if expected_cor_basis == 'fixed':
        expected_cor_to_use = fixed_expected_cor_value
        print(f"数据集 [{dataset_name}] - RA计算基准 (COR_expected): 固定值 {expected_cor_to_use:.4f} ({expected_cor_to_use*100:.2f}%)")
    elif expected_cor_basis == 'mean':
        expected_cor_to_use = calculate_distribution_mean(best_fit_distribution, best_params)
        print(f"数据集 [{dataset_name}] - RA计算基准 (COR_expected): 拟合分布的均值 {expected_cor_to_use:.4f} ({expected_cor_to_use*100:.2f}%)")
    elif expected_cor_basis == 'mode':
        data_min_val, data_max_val = np.min(data), np.max(data)
        expected_cor_to_use = get_distribution_mode(best_fit_distribution, best_params, data_min_val, data_max_val)
        print(f"数据集 [{dataset_name}] - RA计算基准 (COR_expected): 拟合分布的众数 (峰值) {expected_cor_to_use:.4f} ({expected_cor_to_use*100:.2f}%)")
    else:
        print(f"错误: 无效的 'expected_cor_basis': {expected_cor_basis}。请选择 'fixed', 'mean', 或 'mode'")
        plt.show()
        return None

    ra_ratio = np.nan
    if expected_cor_to_use is not None and not np.isnan(expected_cor_to_use) and expected_cor_to_use > 0 :
        ra_ratio = cor_at_confidence_level / expected_cor_to_use - 1
    else:
        print(f"警告: 数据集 [{dataset_name}] - 计算得到的预期综合成本率 ({expected_cor_to_use}) 无效 (零、负或NaN)，RA比率无法计算。")

    print(f"数据集 [{dataset_name}] - 计算得到的 LRC RA 比率: {ra_ratio:.4f} (即 {ra_ratio*100:.2f}%)")

    return {
        'dataset_name': dataset_name,
        'selected_distribution_name': best_fit_distribution.name,
        'estimated_parameters': best_params,
        'aic_of_selected': best_aic,
        'cor_at_confidence_level': cor_at_confidence_level,
        'expected_cor_basis': expected_cor_basis,
        'expected_cor_used_for_ra': expected_cor_to_use,
        'lrc_ra_ratio': ra_ratio,
        'all_fit_results': pd.DataFrame(fit_results).sort_values(by='aic').reset_index(drop=True)
    }


# --- 主程序执行 ---
if __name__ == "__main__":
    # 定义不同的数据集
    datasets = {
        "育肥猪": [ # 您之前用于替换的数据
            0.828094482, 0.543436579, 0.833697162, 1.265072284, 0.975212324,
            0.729494495, 0.894991516, 0.732280431, 0.798998131, 0.714633169
        ],
        "特色种植险": [ # 您在“真实数据”请求中引用的那组 (我根据您的描述保留)
            1.496343733, 1.336118436, 0.70185022,  0.82899037,  0.602292538,
            0.824430755, 0.641732692, 0.68410927,  0.772624554, 0.746709797
        ],
        "商业三者": [ # 您最新提供的
            0.425299364, 0.596419551, 0.560003463, 0.605537298, 0.668452032,
            0.777330604, 0.876000948, 0.796779993, 0.664239748, 0.455203044
        ]
    }

    all_results_summary = []

    # 您可以选择要运行的数据集名称
    # datasets_to_run = ["育肥猪", "特色种植险", "商业三者"] # 运行所有
    datasets_to_run = ["商业三者"] # 或者只运行一个进行测试

    for name in datasets_to_run:
        if name not in datasets:
            print(f"警告: 数据集 '{name}' 未在datasets字典中定义，跳过。")
            continue

        data_values = datasets[name]
        print(f"\n\n{'#'*20} 开始处理数据集: {name} {'#'*20}")
        print(f"历史综合成本率数据: {data_values}")
        print("-" * 70)

        current_data_mean = np.mean(data_values)
        print(f"(参考：数据集 [{name}] 的均值为: {current_data_mean:.4f})")
        # 您可以为每个数据集和每个场景自定义 fixed_expected_cor_value 和 manual_dist_choice
        # 例如，您可以先对每个数据集运行一次 scenario 1 (manual_dist_choice=None)
        # 然后根据输出的图形和统计数据，为该数据集确定一个最佳分布，
        # 再用这个确定的分布作为 manual_dist_choice 运行 scenario 2 和 3。

        # 示例：为当前数据集运行三种场景
        # 场景1: 固定预期COR (使用该数据集的均值), 自动选择分布
        print(f"\n--- 场景1 for [{name}]: 使用固定的预期COR ({current_data_mean*100:.2f}%), 自动选择分布 ---")
        results_s1 = calculate_lrc_ra_ratio(
            dataset_name=name,
            historical_cor_data=data_values,
            expected_cor_basis='fixed',
            fixed_expected_cor_value=current_data_mean,
            manual_dist_choice='weibull_min' # 自动选择
        )
        if results_s1: all_results_summary.append(results_s1)

        # 场景2: 使用拟合分布的均值作为预期COR
        # (假设根据场景1的输出，我们为当前数据集选择了一个最佳分布)
        # 这里我们仍然用None让它自动选，您可以根据需要修改
        manual_choice_for_s2_s3 = results_s1['selected_distribution_name'] if results_s1 else None
        print(f"\n--- 场景2 for [{name}]: 使用拟合分布的均值作为预期COR, 选择分布: {manual_choice_for_s2_s3} ---")
        results_s2 = calculate_lrc_ra_ratio(
            dataset_name=name,
            historical_cor_data=data_values,
            expected_cor_basis='mean',
            manual_dist_choice='weibull_min'
        )
        if results_s2: all_results_summary.append(results_s2)

        # 场景3: 使用拟合分布的众数(峰值)作为预期COR
        print(f"\n--- 场景3 for [{name}]: 使用拟合分布的众数(峰值)作为预期COR, 选择分布: {manual_choice_for_s2_s3} ---")
        results_s3 = calculate_lrc_ra_ratio(
            dataset_name=name,
            historical_cor_data=data_values,
            expected_cor_basis='mode',
            manual_dist_choice='weibull_min'
        )
        if results_s3: all_results_summary.append(results_s3)

    # 在所有计算结束后，显示所有非阻塞的图形
    plt.show()

    print("\n\n" + "="*30 + " 所有计算结果概览 " + "="*30)
    for summary in all_results_summary:
        if summary:
            print(f"\n数据集: {summary['dataset_name']}")
            print(f"  选择的分布: {summary['selected_distribution_name']} (AIC: {summary['aic_of_selected']:.2f})")
            print(f"  75%分位COR: {summary['cor_at_confidence_level']:.4f}")
            print(f"  预期COR基准: {summary['expected_cor_basis']}")
            print(f"  预期COR (用于计算): {summary['expected_cor_used_for_ra']:.4f}")
            print(f"  LRC RA 比率: {summary['lrc_ra_ratio']:.4%}")
            print(f"  --- 详细拟合结果 (按AIC排序) ---")
            print(summary['all_fit_results'][['name', 'aic', 'ks_p_value', 'params']].head())