# -*- coding: utf-8 -*-
"""
期权定价库使用示例

展示如何使用提取和优化的期权定价函数库。
"""

import io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 设置默认编码
if sys.version_info[0] == 3 and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (ValueError, AttributeError):
        pass

# 导入期权定价库
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from option_pricing.models.bsm import bsm_model, bsm_model_with_strike_ratio
from option_pricing.models.binomial import binary_tree_model_with_strike_ratio
from option_pricing.core.volatility import calculate_implied_volatility, find_nearest_value
from option_pricing.products.vanilla import vanillatype, VanillaProduct
from option_pricing.products.structured import cp_eu_call_1, cp_eu_diff
from option_pricing.strategies.spreads import BullSpreadStrategy, BearSpreadStrategy
from option_pricing.utils.helpers import find_closest_number
from option_pricing.utils.config import default_config


def example_bsm_pricing():
    """示例1: BSM定价模型"""
    print("=" * 60)
    print("示例1: BSM期权定价模型")
    print("=" * 60)

    # 参数设置
    S0 = 100.0  # 标的资产价格
    K = 100.0   # 行权价格
    T = 1.0     # 到期时间（年）
    r = 0.05    # 无风险利率
    sigma = 0.2  # 波动率

    # 使用原notebook兼容的接口
    call_price, vega = bsm_model('call', S0, K, T, r, sigma)
    put_price, _ = bsm_model('put', S0, K, T, r, sigma)

    print(f"标的价格: {S0}")
    print(f"行权价格: {K}")
    print(f"到期时间: {T}年")
    print(f"无风险利率: {r:.2%}")
    print(f"波动率: {sigma:.2%}")
    print(f"看涨期权价格: {call_price}")
    print(f"看跌期权价格: {put_price}")
    print(f"Vega值: {vega:.4f}")

    # 使用行权比价接口
    KS = 1.0  # 行权比价（平值）
    call_price_ratio, _ = bsm_model_with_strike_ratio('call', S0, KS, T, r, sigma)
    print(f"\n使用行权比价接口:")
    print(f"行权比价: {KS}")
    print(f"看涨期权价格: {call_price_ratio}")

    return call_price, put_price


def example_binomial_pricing():
    """示例2: 二叉树定价模型"""
    print("\n" + "=" * 60)
    print("示例2: 二叉树期权定价模型")
    print("=" * 60)

    # 参数设置
    S = 100.0  # 标的资产价格
    KS = 1.0   # 行权比价
    T = 1.0    # 到期时间
    r = 0.05   # 无风险利率
    sigma = 0.2  # 波动率
    N = 100    # 二叉树步数

    # 美式看涨期权
    american_call = binary_tree_model_with_strike_ratio(
        S, KS, T, r, sigma, N,
        option_type='call', is_american=True
    )

    # 欧式看涨期权
    european_call = binary_tree_model_with_strike_ratio(
        S, KS, T, r, sigma, N,
        option_type='call', is_american=False
    )

    # 美式看跌期权
    american_put = binary_tree_model_with_strike_ratio(
        S, KS, T, r, sigma, N,
        option_type='put', is_american=True
    )

    print(f"标的价格: {S}")
    print(f"行权比价: {KS}")
    print(f"到期时间: {T}年")
    print(f"二叉树步数: {N}")
    print(f"美式看涨期权价格: {american_call:.4f}")
    print(f"欧式看涨期权价格: {european_call:.4f}")
    print(f"美式看跌期权价格: {american_put:.4f}")

    # 比较美式和欧式期权的差异
    print(f"\n美式 vs 欧式看涨期权差异: {american_call - european_call:.4f}")

    return american_call, european_call, american_put


def example_implied_volatility():
    """示例3: 隐含波动率计算（修复bug）"""
    print("\n" + "=" * 60)
    print("示例3: 隐含波动率计算")
    print("=" * 60)

    # 已知市场价格
    market_price = 10.45  # 看涨期权市场价格
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05

    # 计算隐含波动率
    implied_vol = calculate_implied_volatility(
        market_price, 'call', S0, K, T, r,
        initial_guess=0.5, method='newton', verbose=False
    )

    print(f"期权市场价格: {market_price}")
    print(f"标的价格: {S0}")
    print(f"行权价格: {K}")
    print(f"到期时间: {T}年")
    print(f"无风险利率: {r:.2%}")
    print(f"计算得到的隐含波动率: {implied_vol:.4f} ({implied_vol*100:.2f}%)")

    # 验证：使用隐含波动率重新计算价格
    from option_pricing.core.pricing import bsm_price
    calculated_price = bsm_price('call', S0, K, T, r, implied_vol, return_greeks=False)
    print(f"使用隐含波动率重新计算的价格: {calculated_price:.4f}")
    print(f"价格差异: {abs(market_price - calculated_price):.6f}")

    return implied_vol


def example_find_nearest_value():
    """示例4: 寻找最接近的数（修复bug）"""
    print("\n" + "=" * 60)
    print("示例4: 寻找最接近的数")
    print("=" * 60)

    # 测试数据
    numbers = [95, 100, 105, 110, 115]
    target = 102.5

    # 使用修复后的函数
    nearest = find_nearest_value(target, numbers)

    # 使用原notebook兼容的函数名
    nearest_compat = find_closest_number(target, numbers)

    print(f"目标值: {target}")
    print(f"候选数组: {numbers}")
    print(f"最接近的数（新函数）: {nearest}")
    print(f"最接近的数（兼容函数）: {nearest_compat}")

    # 测试多个目标值
    targets = [97, 102.5, 107, 112]
    for t in targets:
        n = find_nearest_value(t, numbers)
        print(f"目标值 {t} -> 最接近 {n} (差值: {abs(t-n):.2f})")

    return nearest


def example_structured_product():
    """示例5: 结构化产品收益计算"""
    print("\n" + "=" * 60)
    print("示例5: 结构化产品收益计算")
    print("=" * 60)

    # 欧式看涨保本结构
    asset = 10000000  # 本金1000万
    call_price = 10.0  # 期权价格
    strike_price = 100.0  # 行权价格
    participation_rate = 0.55  # 参与率55%
    capital_protection_rate = 1.0  # 100%保本
    T = 1.0  # 1年

    # 计算固收产品收益率
    r_fixed = cp_eu_call_1(
        asset, call_price, strike_price,
        participation_rate, capital_protection_rate, T
    )

    print(f"本金: {asset:,}元")
    print(f"期权价格: {call_price}元/股")
    print(f"行权价格: {strike_price}元")
    print(f"参与率: {participation_rate:.2%}")
    print(f"保本率: {capital_protection_rate:.2%}")
    print(f"期限: {T}年")
    print(f"需要的固收产品年化收益率: {r_fixed:.4f} ({r_fixed*100:.2f}%)")

    # 计算头寸规模
    v = asset / strike_price * participation_rate
    cost_call = call_price * v
    cost_fixed = asset - cost_call

    print(f"\n头寸规模: {v:.2f}股")
    print(f"期权成本: {cost_call:,.2f}元")
    print(f"固收产品投资: {cost_fixed:,.2f}元")
    print(f"期权成本比例: {cost_call/asset:.2%}")

    # 欧式看涨价差结构
    print("\n" + "-" * 40)
    print("欧式看涨价差结构示例")

    call_sell_price = 5.0  # 卖出期权价格
    r_fixed_diff = cp_eu_diff(
        asset, call_price, call_sell_price, strike_price,
        participation_rate, capital_protection_rate, T
    )

    print(f"买入期权价格: {call_price}元/股")
    print(f"卖出期权价格: {call_sell_price}元/股")
    print(f"需要的固收产品年化收益率: {r_fixed_diff:.4f} ({r_fixed_diff*100:.2f}%)")

    return r_fixed, r_fixed_diff


def example_vanilla_product():
    """示例6: 香草产品定价（高级接口）"""
    print("\n" + "=" * 60)
    print("示例6: 香草产品定价")
    print("=" * 60)

    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    n_days = len(dates)

    # 生成价格序列（几何布朗运动）
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = 100 * np.exp(np.cumsum(returns))

    # 创建DataFrame
    data = pd.DataFrame({
        'close': prices,
        'changeRatio': returns,
        'sigma': 0.2,  # 恒定波动率
        'turnoverRatio': np.random.uniform(0.5, 2.0, n_days)
    }, index=dates)

    # 计算滚动波动率
    data['sigma'] = data['changeRatio'].rolling(126).std() * np.sqrt(252)
    data = data.dropna()

    print(f"数据期间: {data.index[0].date()} 到 {data.index[-1].date()}")
    print(f"当前价格: {data['close'].iloc[-1]:.2f}")
    print(f"当前波动率: {data['sigma'].iloc[-1]:.4f}")

    # 欧式看涨保本结构
    print("\n" + "-" * 40)
    print("欧式看涨保本结构")

    product = VanillaProduct(
        data=data,
        option_type='eu_call',
        asset=10000000,
        participation_rate=0.8,
        capital_protection_rate=0.95,
        T=1.0,
        r=0.03,
        turnover_adjustment=True
    )

    product_df = product.price_eu_call(strike_ratio=1.0, verbose=True)

    print("\n产品信息摘要:")
    print(product_df.to_string(index=False))

    # 绘制损益图
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        product.plot_payoff(ax=ax)
        plt.title('欧式看涨保本结构损益图')
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("注意: matplotlib未安装，无法显示图表")

    return product_df


def example_spread_strategy():
    """示例7: 价差策略"""
    print("\n" + "=" * 60)
    print("示例7: 价差策略")
    print("=" * 60)

    # 牛市价差策略
    print("牛市价差策略示例:")

    K_buy = 100.0
    K_sell = 120.0
    c_buy = 10.0
    c_sell = 5.0

    bull_spread = BullSpreadStrategy(K_buy, K_sell, c_buy, c_sell)

    # 计算盈亏平衡点
    lower_be, upper_be = bull_spread.calculate_breakeven_points()
    max_profit, min_profit = bull_spread.calculate_max_min_profit()

    print(f"买入行权价: {K_buy}")
    print(f"卖出行权价: {K_sell}")
    print(f"买入期权价格: {c_buy}")
    print(f"卖出期权价格: {c_sell}")
    print(f"下盈亏平衡点: {lower_be:.2f}")
    print(f"上盈亏平衡点: {upper_be:.2f}")
    print(f"最大收益: {max_profit:,.2f}元")
    print(f"最小收益: {min_profit:,.2f}元")
    print(f"净期权费收入: {c_sell - c_buy:.2f}元/股")

    # 创建收益数据框
    revenue_df = bull_spread.create_revenue_dataframe()
    print(f"\n收益数据框形状: {revenue_df.shape}")
    print("收益数据框前5行:")
    print(revenue_df.head())

    # 熊市价差策略
    print("\n" + "-" * 40)
    print("熊市价差策略示例:")

    K_buy_put = 120.0
    K_sell_put = 100.0
    p_buy = 10.0
    p_sell = 5.0

    bear_spread = BearSpreadStrategy(K_buy_put, K_sell_put, p_buy, p_sell)
    lower_be_bear, upper_be_bear = bear_spread.calculate_breakeven_points()

    print(f"买入行权价: {K_buy_put}")
    print(f"卖出行权价: {K_sell_put}")
    print(f"买入期权价格: {p_buy}")
    print(f"卖出期权价格: {p_sell}")
    print(f"下盈亏平衡点: {lower_be_bear:.2f}")
    print(f"上盈亏平衡点: {upper_be_bear:.2f}")

    return bull_spread, bear_spread


def example_config_management():
    """示例8: 配置管理"""
    print("\n" + "=" * 60)
    print("示例8: 配置管理")
    print("=" * 60)

    # 使用默认配置
    config = default_config

    print("默认配置信息:")
    print(f"BSM价格小数位数: {config.pricing.bsm_round_decimals}")
    print(f"默认二叉树步数: {config.pricing.default_tree_steps}")
    print(f"默认资产规模: {config.strategy.default_asset:,}元")
    print(f"价格模拟范围: {config.strategy.price_simulation_range}")
    print(f"图表DPI: {config.visualization.figure_dpi}")
    print(f"标的资产颜色: {config.visualization.colors['underlying']}")

    # 动态更新配置
    config.update_config(
        pricing.bsm_round_decimals=6,
        strategy.default_asset=5000000,
        visualization.figure_dpi=150
    )

    print("\n更新后的配置:")
    print(f"BSM价格小数位数: {config.pricing.bsm_round_decimals}")
    print(f"默认资产规模: {config.strategy.default_asset:,}元")
    print(f"图表DPI: {config.visualization.figure_dpi}")

    # 保存和加载配置
    config_file = "option_pricing_config.json"
    config.save_to_file(config_file)
    print(f"\n配置已保存到: {config_file}")

    # 创建新配置并加载
    new_config = default_config.__class__()
    new_config.load_from_file(config_file)
    print(f"配置已加载，BSM小数位数: {new_config.pricing.bsm_round_decimals}")

    # 清理临时文件
    import os
    if os.path.exists(config_file):
        os.remove(config_file)

    return config


def main():
    """运行所有示例"""
    print("期权定价函数库使用示例")
    print("=" * 60)

    results = {}

    # 运行各个示例
    try:
        results['bsm'] = example_bsm_pricing()
    except Exception as e:
        print(f"BSM定价示例失败: {e}")

    try:
        results['binomial'] = example_binomial_pricing()
    except Exception as e:
        print(f"二叉树定价示例失败: {e}")

    try:
        results['implied_vol'] = example_implied_volatility()
    except Exception as e:
        print(f"隐含波动率示例失败: {e}")

    try:
        results['nearest_value'] = example_find_nearest_value()
    except Exception as e:
        print(f"寻找最接近数示例失败: {e}")

    try:
        results['structured'] = example_structured_product()
    except Exception as e:
        print(f"结构化产品示例失败: {e}")

    try:
        results['vanilla'] = example_vanilla_product()
    except Exception as e:
        print(f"香草产品示例失败: {e}")

    try:
        results['spread'] = example_spread_strategy()
    except Exception as e:
        print(f"价差策略示例失败: {e}")

    try:
        results['config'] = example_config_management()
    except Exception as e:
        print(f"配置管理示例失败: {e}")

    print("\n" + "=" * 60)
    print("所有示例完成!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()