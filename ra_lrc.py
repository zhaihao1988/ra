import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题
# 主函数
def calculate_lrc_ra_ratio(historical_cor_data, expected_cor, confidence_level=0.75):
    """
    根据历史综合成本率数据，拟合分布，计算75%分位点，并计算LRC RA比率。

    参数:
    historical_cor_data (list or np.array): 历史综合成本率数据 (例如 [90, 98, ...])
    expected_cor (float): 预期的最优估计综合成本率 (例如 0.95)
    confidence_level (float): 置信水平 (例如 0.75)

    返回:
    dict: 包含所选分布名称、估计参数、75%分位综合成本率和RA比率的结果字典。
          如果无法确定最佳分布，则可能返回None或错误信息。
    """
    # 1. 数据预处理
    data = np.array(historical_cor_data) / 100.0  # 将百分比转换为小数

    # 2. 尝试拟合多种分布并评估
    # 定义候选分布列表 (scipy.stats中的对象)
    distributions_to_try = [
        stats.norm,
        stats.lognorm,
        stats.gamma,
        stats.weibull_min, # Weibull (minimum)
        stats.genextreme   # Generalized Extreme Value (GEV)
        # stats.beta # Beta分布通常用于[0,1]区间，如果COR可能超过100%，需要调整loc和scale，或者不优先选择
    ]

    best_fit_distribution = None
    best_aic = np.inf
    best_params = None
    fit_results = []

    print("正在尝试拟合以下分布模型...\n")

    for distribution in distributions_to_try:
        dist_name = distribution.name
        print(f"--- 拟合分布: {dist_name} ---")
        try:
            # 尝试拟合分布，获取参数
            # 对于某些分布，可能需要指定初始参数或边界，特别是样本量较小时
            # fit()方法返回MLE参数，通常是形状参数(shape), 位置参数(loc), 尺度参数(scale)
            with warnings.catch_warnings(): # 忽略拟合过程中可能出现的RuntimeWarning
                warnings.simplefilter("ignore")
                params = distribution.fit(data)

            # 计算AIC (赤池信息准则)
            # AIC = 2k - 2*ln(L)  k:参数数量, L:最大似然值
            # 对数似然: ll = np.sum(distribution.logpdf(data, *params))
            # 参数数量k: len(params) - (1 if hasattr(distribution, 'loc') and params[-2] == 0 else 0) - (1 if hasattr(distribution, 'scale') and params[-1] == 1 else 0)
            # Scipy的fit会返回所有参数，包括固定的loc和scale（如果它们在分布中是标准的话）
            # 我们通常只关心自由参数的数量。对于大多数双参数分布，是2个；三参数是3个。
            # 为了简化，我们直接使用params的长度，但这可能不完全精确，更精确的做法是检查自由参数个数。
            # 或者，更简单地，我们可以仅基于统计检验和图形来做初步判断，因为AIC/BIC对小样本也敏感。

            # 为了更稳健地计算AIC，我们可以计算对数似然
            ll = np.sum(distribution.logpdf(data, *params))
            k = len(params) # 参数个数
            aic = 2 * k - 2 * ll
            print(f"参数: {params}")
            print(f"对数似然 (Log-Likelihood): {ll:.4f}")
            print(f"AIC: {aic:.4f}")

            # 拟合优度检验 (Kolmogorov-Smirnov test)
            # K-S test的原假设是数据来自于指定的分布（使用从数据中估计的参数）
            # 对于参数是估计出来的情况，K-S检验的p值需要谨慎解读
            D, p_value = stats.kstest(data, dist_name, args=params)
            print(f"K-S 检验统计量 D: {D:.4f}, p-value: {p_value:.4f}")

            fit_results.append({
                'name': dist_name,
                'params': params,
                'aic': aic,
                'ks_stat': D,
                'ks_p_value': p_value,
                'log_likelihood': ll
            })

            if aic < best_aic and np.isfinite(aic) : # 确保AIC是有效值
                # 进一步检查K-S检验的p值，虽然不是决定性的，但可以作为参考
                # 一个简单的规则：如果p值非常小（例如 < 0.01 或 0.05），则该分布可能不太合适
                # 但对于小样本，所有p值都可能较大。
                # 我们主要依赖AIC进行比较，同时结合图形判断。
                best_aic = aic
                best_fit_distribution = distribution
                best_params = params

        except Exception as e:
            print(f"拟合分布 {dist_name} 失败: {e}")
        print("-" * 30)

    if best_fit_distribution is None:
        print("\n未能找到合适的分布模型。")
        return None

    print(f"\n根据AIC，初步选择的最佳拟合分布为: {best_fit_distribution.name}")
    print(f"其估计参数为: {best_params}")
    print(f"其AIC值为: {best_aic:.4f}")
    # !!! 手动选择的示例 !!!
    # 如果你想强制使用genextreme
    selected_dist_name_manual = 'gamma'
    manual_params = None
    for res in fit_results:
        if res['name'] == selected_dist_name_manual:
            best_fit_distribution = getattr(stats, selected_dist_name_manual)  # 获取scipy.stats中的对象
            best_params = res['params']
            print(f"\n!!! 用户手动选择使用分布: {best_fit_distribution.name} !!!")
            print(f"其估计参数为: {best_params}")
            break
    # !!! 手动选择结束 !!!
    # 3. 可视化最佳拟合分布 (可选，但强烈推荐)
    plt.figure(figsize=(12, 7))
    # 绘制直方图
    plt.hist(data, bins='auto', density=True, alpha=0.7, label='历史综合成本率直方图')
    # 绘制拟合的PDF曲线
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    pdf_fitted = best_fit_distribution.pdf(x, *best_params)
    plt.plot(x, pdf_fitted, 'r-', lw=2, label=f'拟合的 {best_fit_distribution.name} 分布 PDF')
    plt.title(f'历史综合成本率与拟合的 {best_fit_distribution.name} 分布')
    plt.xlabel('综合成本率 (小数形式)')
    plt.ylabel('密度')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Q-Q图
    plt.figure(figsize=(7, 7))
    stats.probplot(data, dist=best_fit_distribution, sparams=best_params[:-2] if len(best_params)>2 else best_params, plot=plt) # sparams通常是形状参数
    plt.title(f'{best_fit_distribution.name} 分布的 Q-Q 图')
    plt.grid(True)
    plt.show()


    # 4. 计算选定分布的75%分位点的值
    cor_at_confidence_level = best_fit_distribution.ppf(confidence_level, *best_params)
    print(f"\n拟合的 {best_fit_distribution.name} 分布在 {confidence_level*100:.0f}% 分位点的综合成本率为: {cor_at_confidence_level:.4f} (即 {cor_at_confidence_level*100:.2f}%)")

    # 5. 计算LRC的RA比率
    if expected_cor <= 0:
        raise ValueError("预期的最优估计综合成本率必须为正。")
    ra_ratio = cor_at_confidence_level / expected_cor - 1
    print(f"预期的最优估计综合成本率 (COR_expected): {expected_cor:.4f} (即 {expected_cor*100:.2f}%)")
    print(f"计算得到的 LRC RA 比率: {ra_ratio:.4f} (即 {ra_ratio*100:.2f}%)")

    return {
        'best_distribution_name': best_fit_distribution.name,
        'estimated_parameters': best_params,
        'aic': best_aic,
        'cor_at_confidence_level': cor_at_confidence_level,
        'lrc_ra_ratio': ra_ratio,
        'all_fit_results': fit_results # 返回所有尝试分布的结果，供进一步分析
    }

# --- 主程序执行 ---
if __name__ == "__main__":
    # 1. 假设中原农险过去十年农业险的综合成本率为（单位%）
    historical_cors = [90, 98, 91, 92, 110, 101, 90, 92, 95, 94]
    # 2. 预期的最优估计综合成本率
    expected_combined_ratio = 0.95 # 95%

    print("开始计算LRC RA比率...")
    print(f"历史综合成本率数据 (单位%): {historical_cors}")
    print(f"预期的最优估计综合成本率: {expected_combined_ratio*100:.0f}%")
    print("-" * 50)

    results = calculate_lrc_ra_ratio(historical_cors, expected_combined_ratio, confidence_level=0.75)

    if results:
        print("\n--- 计算结果总结 ---")
        print(f"最佳拟合分布: {results['best_distribution_name']}")
        print(f"估计参数: {results['estimated_parameters']}")
        print(f"AIC值: {results['aic']:.4f}")
        print(f"75%分位综合成本率: {results['cor_at_confidence_level']:.4f} ({results['cor_at_confidence_level']*100:.2f}%)")
        print(f"LRC RA 比率: {results['lrc_ra_ratio']:.4f} ({results['lrc_ra_ratio']*100:.2f}%)")

        # 可以打印所有尝试分布的结果以供比较
        print("\n--- 所有尝试分布的拟合详情 ---")
        df_results = pd.DataFrame(results['all_fit_results'])
        print(df_results.sort_values(by='aic'))