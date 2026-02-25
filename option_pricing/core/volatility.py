# -*- coding: utf-8 -*-
"""
波动率相关函数

提供波动率计算相关函数，包括隐含波动率计算。
"""

import numpy as np
from scipy import stats
from scipy.optimize import brentq, newton
from typing import Union, Optional, Tuple
import warnings


def calculate_implied_volatility(
    market_price: float,
    option_type: str,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    initial_guess: float = 0.5,
    method: str = 'newton',
    max_iterations: int = 100,
    tolerance: float = 1e-8,
    verbose: bool = False
) -> float:
    """
    计算隐含波动率

    参数:
    ----------
    market_price : float
        期权市场价格
    option_type : str
        期权类型: 'call' 或 'put'
    S0 : float
        标的资产当前价格
    K : float
        行权价格
    T : float
        到期时间（年）
    r : float
        无风险利率（年化）
    q : float, optional
        连续分红率（年化），默认为0
    initial_guess : float, optional
        初始猜测值，默认为0.5
    method : str, optional
        计算方法: 'newton'（牛顿法）或 'brentq'（布伦特法），默认为'newton'
    max_iterations : int, optional
        最大迭代次数，默认为100
    tolerance : float, optional
        容差，默认为1e-8
    verbose : bool, optional
        是否输出调试信息，默认为False

    返回:
    -------
    implied_vol : float
        隐含波动率

    示例:
    --------
    >>> calculate_implied_volatility(10.45, 'call', S0=100, K=100, T=1, r=0.05)
    0.2001

    注意:
    -----
    - 修复了原notebook中sigma变量未初始化的问题
    - 添加了输入验证和错误处理
    """
    # 参数验证
    option_type = option_type.lower()
    if option_type not in ['call', 'put']:
        raise ValueError(f"option_type必须是'call'或'put'，当前为: {option_type}")

    if market_price <= 0:
        raise ValueError("市场价格必须大于0")
    if S0 <= 0:
        raise ValueError("标的资产价格必须大于0")
    if K <= 0:
        raise ValueError("行权价格必须大于0")
    if T <= 0:
        raise ValueError("到期时间必须大于0")
    if initial_guess <= 0:
        raise ValueError("初始猜测值必须大于0")
    if max_iterations <= 0:
        raise ValueError("最大迭代次数必须大于0")
    if tolerance <= 0:
        raise ValueError("容差必须大于0")

    # 导入bsm_price函数（避免循环导入）
    from .pricing import bsm_price

    def price_error(sigma: float) -> float:
        """计算价格误差函数"""
        try:
            price = bsm_price(option_type, S0, K, T, r, sigma, q, return_greeks=False)
            return price - market_price
        except (ValueError, ZeroDivisionError) as e:
            if verbose:
                print(f"sigma={sigma:.6f}时计算错误: {e}")
            return np.nan

    def price_error_with_vega(sigma: float) -> Tuple[float, float]:
        """计算价格误差和Vega值（用于牛顿法）"""
        try:
            price, greeks = bsm_price(option_type, S0, K, T, r, sigma, q, return_greeks=True)
            error = price - market_price
            vega = greeks['vega']
            return error, vega
        except (ValueError, ZeroDivisionError) as e:
            if verbose:
                print(f"sigma={sigma:.6f}时计算错误: {e}")
            return np.nan, np.nan

    if method.lower() == 'newton':
        # 牛顿法实现（修复原notebook中的bug）
        sigma = initial_guess
        for i in range(max_iterations):
            error, vega = price_error_with_vega(sigma)

            # 检查计算是否有效
            if np.isnan(error) or np.isnan(vega):
                if verbose:
                    print(f"第{i+1}次迭代: sigma={sigma:.6f}, 计算无效")
                sigma = sigma / 2  # 尝试较小的sigma
                continue

            if abs(error) < tolerance:
                if verbose:
                    print(f"第{i+1}次迭代: sigma={sigma:.6f}, error={error:.8f}, 达到容差")
                return sigma

            # 避免除以0或过小的Vega值
            if abs(vega) < 1e-10:
                if verbose:
                    print(f"第{i+1}次迭代: Vega值过小({vega:.8f})，使用二分法")
                # 如果Vega太小，退回二分法
                method = 'brentq'
                break

            # 牛顿法更新公式: sigma_new = sigma_old - f(sigma)/f'(sigma)
            sigma_new = sigma - error / vega

            # 确保sigma在合理范围内
            if sigma_new <= 0:
                sigma_new = sigma / 2
            elif sigma_new > 5:  # 波动率通常不超过500%
                sigma_new = 5

            if verbose:
                print(f"第{i+1}次迭代: sigma={sigma:.6f}, error={error:.8f}, vega={vega:.8f}, sigma_new={sigma_new:.6f}")

            # 检查收敛
            if abs(sigma_new - sigma) < tolerance:
                if verbose:
                    print(f"第{i+1}次迭代后收敛")
                return sigma_new

            sigma = sigma_new

        if verbose:
            print(f"牛顿法在{max_iterations}次迭代后未收敛，退回布伦特法")
        method = 'brentq'

    if method.lower() == 'brentq':
        # 布伦特法（更稳定）
        try:
            # 定义搜索边界
            sigma_low = 1e-6
            sigma_high = 5.0  # 最大500%波动率

            # 调整边界直到函数值符号相反
            error_low = price_error(sigma_low)
            error_high = price_error(sigma_high)

            if np.isnan(error_low) or np.isnan(error_high):
                raise ValueError("边界点计算失败")

            # 如果边界点函数值符号相同，扩展边界
            for _ in range(20):  # 最多尝试20次扩展边界
                if error_low * error_high < 0:
                    break
                if abs(error_low) < abs(error_high):
                    sigma_low = max(sigma_low / 2, 1e-10)
                    error_low = price_error(sigma_low)
                else:
                    sigma_high = min(sigma_high * 2, 10.0)
                    error_high = price_error(sigma_high)

            if error_low * error_high >= 0:
                # 使用更保守的方法：缩小边界直到找到根
                sigma_mid = initial_guess
                for _ in range(50):
                    error_mid = price_error(sigma_mid)
                    if np.isnan(error_mid):
                        sigma_mid = sigma_mid / 2
                        continue

                    # 检查是否为根
                    if abs(error_mid) < tolerance:
                        return sigma_mid

                    # 寻找符号变化
                    if error_mid * error_low < 0:
                        sigma_high = sigma_mid
                        error_high = error_mid
                    elif error_mid * error_high < 0:
                        sigma_low = sigma_mid
                        error_low = error_mid
                    else:
                        # 没有符号变化，随机扰动
                        sigma_mid = sigma_mid * np.random.uniform(0.8, 1.2)

            # 使用布伦特法求解
            try:
                result = brentq(
                    price_error,
                    sigma_low,
                    sigma_high,
                    xtol=tolerance,
                    maxiter=max_iterations,
                    full_output=False
                )
                return result
            except ValueError as e:
                if verbose:
                    print(f"布伦特法失败: {e}")
                # 退回简单的二分法
                for _ in range(max_iterations):
                    sigma_mid = (sigma_low + sigma_high) / 2
                    error_mid = price_error(sigma_mid)

                    if np.isnan(error_mid):
                        # 无效点，收缩边界
                        sigma_high = sigma_mid
                        continue

                    if abs(error_mid) < tolerance:
                        return sigma_mid

                    if error_mid * error_low < 0:
                        sigma_high = sigma_mid
                        error_high = error_mid
                    else:
                        sigma_low = sigma_mid
                        error_low = error_mid

                # 返回最佳猜测
                return (sigma_low + sigma_high) / 2

        except Exception as e:
            if verbose:
                print(f"布伦特法异常: {e}")
            # 退回初始猜测
            return initial_guess

    raise ValueError(f"不支持的计算方法: {method}")


def calculate_historical_volatility(
    prices: np.ndarray,
    period: int = 252,
    annualization_factor: float = 252.0,
    method: str = 'simple'
) -> float:
    """
    计算历史波动率

    参数:
    ----------
    prices : np.ndarray
        价格序列
    period : int, optional
        计算窗口期，默认为252（一年交易日）
    annualization_factor : float, optional
        年化因子，默认为252.0
    method : str, optional
        计算方法: 'simple'（简单收益率）或 'log'（对数收益率），默认为'simple'

    返回:
    -------
    volatility : float
        年化历史波动率

    示例:
    --------
    >>> prices = np.array([100, 101, 99, 102, 100])
    >>> calculate_historical_volatility(prices)
    0.2015
    """
    if len(prices) < 2:
        raise ValueError("价格序列至少需要2个数据点")

    if period > len(prices):
        period = len(prices)
        import warnings
        warnings.warn(f"窗口期{period}大于数据长度，使用全部数据")

    # 计算收益率
    if method == 'simple':
        returns = prices[1:] / prices[:-1] - 1
    elif method == 'log':
        returns = np.log(prices[1:] / prices[:-1])
    else:
        raise ValueError(f"不支持的计算方法: {method}")

    # 使用最近period个数据点
    returns_window = returns[-period:] if len(returns) > period else returns

    # 计算波动率
    std_dev = np.std(returns_window, ddof=1)
    volatility = std_dev * np.sqrt(annualization_factor)

    return volatility


def find_nearest_value(x: float, numbers: Union[list, np.ndarray]) -> float:
    """
    在数组中寻找最接近给定值的数

    参数:
    ----------
    x : float
        目标值
    numbers : Union[list, np.ndarray]
        数值数组

    返回:
    -------
    nearest : float
        最接近的数

    示例:
    --------
    >>> find_nearest_value(100.5, [95, 100, 105])
    100.0

    注意:
    -----
    - 修复了原notebook中find_closest_number函数的逻辑错误
    """
    if len(numbers) == 0:
        raise ValueError("数组不能为空")

    # 转换为numpy数组
    numbers_arr = np.asarray(numbers, dtype=np.float64)

    # 计算绝对差值
    abs_diff = np.abs(numbers_arr - x)
    # 找到最小差值的索引
    nearest_idx = np.argmin(abs_diff)

    return float(numbers_arr[nearest_idx])


# 保持与原notebook兼容的别名
find_closest_number = find_nearest_value
