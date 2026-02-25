# -*- coding: utf-8 -*-
"""
期权定价核心函数

提供期权定价的核心函数，包括BSM模型和二叉树模型。
"""

import numpy as np
from scipy import stats
from typing import Union, Tuple, Optional
import warnings


def bsm_price(
    option_type: str,
    S0: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: float,
    sigma: Union[float, np.ndarray],
    q: float = 0.0,
    return_greeks: bool = False
) -> Union[float, Tuple[float, dict], np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Black-Scholes-Merton期权定价模型

    参数:
    ----------
    option_type : str
        期权类型: 'call' 或 'put'
    S0 : float or np.ndarray
        标的资产当前价格
    K : float or np.ndarray
        行权价格
    T : float or np.ndarray
        到期时间（年）
    r : float
        无风险利率（年化）
    sigma : float or np.ndarray
        波动率（年化）
    q : float, optional
        连续分红率（年化），默认为0
    return_greeks : bool, optional
        是否返回希腊字母，默认为False

    返回:
    -------
    price : float or np.ndarray
        期权价格
    greeks : dict (仅在return_greeks=True时返回)
        希腊字母字典，包含:
        - delta: Delta值
        - gamma: Gamma值
        - theta: Theta值
        - vega: Vega值
        - rho: Rho值

    示例:
    --------
    >>> bsm_price('call', S0=100, K=100, T=1, r=0.05, sigma=0.2)
    10.4506

    注意:
    -----
    - 所有参数都支持标量和向量输入
    - 向量输入时要求各参数形状一致或可广播
    """
    # 参数验证
    option_type = option_type.lower()
    if option_type not in ['call', 'put']:
        raise ValueError(f"option_type必须是'call'或'put'，当前为: {option_type}")

    # 转换为numpy数组以支持向量化计算
    S0_arr = np.asarray(S0, dtype=np.float64)
    K_arr = np.asarray(K, dtype=np.float64)
    T_arr = np.asarray(T, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)

    # 检查参数有效性
    if np.any(S0_arr <= 0):
        raise ValueError("标的资产价格必须大于0")
    if np.any(K_arr <= 0):
        raise ValueError("行权价格必须大于0")
    if np.any(T_arr < 0):
        raise ValueError("到期时间不能为负数")
    if np.any(sigma_arr < 0):
        raise ValueError("波动率不能为负数")

    # 计算d1和d2
    sqrt_T = np.sqrt(T_arr)
    d1 = (np.log(S0_arr / K_arr) + (r - q + 0.5 * sigma_arr**2) * T_arr) / (sigma_arr * sqrt_T)
    d2 = d1 - sigma_arr * sqrt_T

    # 计算期权价格
    if option_type == 'call':
        price = S0_arr * np.exp(-q * T_arr) * stats.norm.cdf(d1) - K_arr * np.exp(-r * T_arr) * stats.norm.cdf(d2)
    else:  # put
        price = K_arr * np.exp(-r * T_arr) * stats.norm.cdf(-d2) - S0_arr * np.exp(-q * T_arr) * stats.norm.cdf(-d1)

    # 四舍五入到4位小数（保持与原notebook一致）
    price = np.round(price, 4)

    if not return_greeks:
        return price.item() if price.size == 1 else price

    # 计算希腊字母
    d1_pdf = stats.norm.pdf(d1)
    cdf_d1 = stats.norm.cdf(d1)
    cdf_d2 = stats.norm.cdf(d2)

    if option_type == 'call':
        delta = np.exp(-q * T_arr) * cdf_d1
        gamma = np.exp(-q * T_arr) * d1_pdf / (S0_arr * sigma_arr * sqrt_T)
        theta = (-S0_arr * sigma_arr * np.exp(-q * T_arr) * d1_pdf / (2 * sqrt_T)
                 - r * K_arr * np.exp(-r * T_arr) * cdf_d2
                 + q * S0_arr * np.exp(-q * T_arr) * cdf_d1)
        vega = S0_arr * np.exp(-q * T_arr) * sqrt_T * d1_pdf
        rho = K_arr * T_arr * np.exp(-r * T_arr) * cdf_d2
    else:  # put
        delta = -np.exp(-q * T_arr) * stats.norm.cdf(-d1)
        gamma = np.exp(-q * T_arr) * d1_pdf / (S0_arr * sigma_arr * sqrt_T)
        theta = (-S0_arr * sigma_arr * np.exp(-q * T_arr) * d1_pdf / (2 * sqrt_T)
                 + r * K_arr * np.exp(-r * T_arr) * stats.norm.cdf(-d2)
                 - q * S0_arr * np.exp(-q * T_arr) * stats.norm.cdf(-d1))
        vega = S0_arr * np.exp(-q * T_arr) * sqrt_T * d1_pdf
        rho = -K_arr * T_arr * np.exp(-r * T_arr) * stats.norm.cdf(-d2)

    greeks = {
        'delta': np.round(delta, 6) if delta.size == 1 else np.round(delta, 6),
        'gamma': np.round(gamma, 6) if gamma.size == 1 else np.round(gamma, 6),
        'theta': np.round(theta, 6) if theta.size == 1 else np.round(theta, 6),
        'vega': np.round(vega, 6) if vega.size == 1 else np.round(vega, 6),
        'rho': np.round(rho, 6) if rho.size == 1 else np.round(rho, 6)
    }

    return (price.item() if price.size == 1 else price, greeks)


def bsm_price_with_strike_ratio(
    option_type: str,
    S0: Union[float, np.ndarray],
    strike_ratio: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: float,
    sigma: Union[float, np.ndarray],
    q: float = 0.0,
    return_greeks: bool = False
) -> Union[float, Tuple[float, dict], np.ndarray, Tuple[np.ndarray, dict]]:
    """
    使用行权比价的BSM定价模型

    参数:
    ----------
    option_type : str
        期权类型: 'call' 或 'put'
    S0 : float or np.ndarray
        标的资产当前价格
    strike_ratio : float or np.ndarray
        行权比价（行权价格与当前价格的比例）
    T : float or np.ndarray
        到期时间（年）
    r : float
        无风险利率（年化）
    sigma : float or np.ndarray
        波动率（年化）
    q : float, optional
        连续分红率（年化），默认为0
    return_greeks : bool, optional
        是否返回希腊字母，默认为False

    返回:
    -------
    price : float or np.ndarray
        期权价格
    greeks : dict (可选)
        希腊字母

    示例:
    --------
    >>> bsm_price_with_strike_ratio('call', S0=100, strike_ratio=1.0, T=1, r=0.05, sigma=0.2)
    10.4506
    """
    K = S0 * strike_ratio
    return bsm_price(option_type, S0, K, T, r, sigma, q, return_greeks)
