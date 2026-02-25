# -*- coding: utf-8 -*-
"""
BSM期权定价模型

提供Black-Scholes-Merton模型的实现，保持与原notebook的兼容性。
"""

import numpy as np
from scipy import stats
from typing import Tuple, Union


def bsm_model(
    Type: str,
    S0: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: float,
    sigma: Union[float, np.ndarray],
    q: float = 0.0
) -> Tuple[Union[float, np.ndarray], float]:
    """
    BSM期权定价模型（与原notebook兼容的接口）

    参数:
    ----------
    Type : str
        期权类型: 'call' 或 'put'
    S0 : float or np.ndarray
        标的物初始价格水平
    K : float or np.ndarray
        行权价格（注意：原notebook中的KS参数已转换为K）
    T : float or np.ndarray
        到期日（年）
    r : float
        固定无风险短期利率
    sigma : float or np.ndarray
        波动率
    q : float, optional
        连续分红率，默认为0

    返回:
    -------
    value : float or np.ndarray
        期权价格（四舍五入到4位小数）
    vega : float
        Vega值（仅返回标量值）

    示例:
    --------
    >>> bsm_model('call', S0=100, K=100, T=1, r=0.05, sigma=0.2)
    (10.4506, 39.8275)

    注意:
    -----
    - 此函数保持与原notebook的兼容性
    - 对于向量输入，Vega值仅返回第一个值
    - 建议使用core.pricing.bsm_price以获得更完整的函数
    """
    # 导入bsm_price函数
    from ..core.pricing import bsm_price

    # 调用bsm_price函数计算价格和希腊字母
    price, greeks = bsm_price(Type, S0, K, T, r, sigma, q, return_greeks=True)

    # 保持与原函数相同的行为：价格四舍五入到4位小数
    price = np.round(price, 4)

    # Vega值：如果是标量则直接返回，如果是数组则返回第一个值
    vega = greeks['vega']
    if isinstance(vega, (np.ndarray, list)):
        vega = float(vega.flat[0]) if hasattr(vega, 'flat') else float(vega[0])

    return price, vega


def bsm_model_with_strike_ratio(
    Type: str,
    S0: Union[float, np.ndarray],
    KS: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: float,
    sigma: Union[float, np.ndarray],
    q: float = 0.0
) -> Tuple[Union[float, np.ndarray], float]:
    """
    使用行权比价的BSM模型（与原notebook完全兼容）

    参数:
    ----------
    Type : str
        期权类型: 'call' 或 'put'
    S0 : float or np.ndarray
        标的物初始价格水平
    KS : float or np.ndarray
        行权比价（行权价格与当前价格的比例）
    T : float or np.ndarray
        到期日（年）
    r : float
        固定无风险短期利率
    sigma : float or np.ndarray
        波动率
    q : float, optional
        连续分红率，默认为0

    返回:
    -------
    value : float or np.ndarray
        期权价格（四舍五入到4位小数）
    vega : float
        Vega值（仅返回标量值）

    示例:
    --------
    >>> bsm_model_with_strike_ratio('call', S0=100, KS=1.0, T=1, r=0.05, sigma=0.2)
    (10.4506, 39.8275)

    注意:
    -----
    - 此函数完全复制原notebook中bsm_model的行为
    - 注意参数名称为KS（行权比价）而非K（行权价格）
    """
    # 计算行权价格
    K = S0 * KS
    return bsm_model(Type, S0, K, T, r, sigma, q)
