# -*- coding: utf-8 -*-
"""
二叉树期权定价模型

提供美式和欧式期权的二叉树定价模型。
"""

import numpy as np
from typing import Union, Optional, Tuple
import warnings


def binary_tree_model(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: float,
    r: float,
    sigma: float,
    N: int,
    option_type: str = 'call',
    is_american: bool = True,
    return_tree: bool = False
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    二叉树期权定价模型

    参数:
    ----------
    S : float or np.ndarray
        初始股票价格（支持向量化）
    K : float or np.ndarray
        行权价格（支持向量化）
    T : float
        到期时间（年）
    r : float
        无风险利率（年化）
    sigma : float
        波动率（年化）
    N : int
        期数（步数）
    option_type : str, optional
        期权类型: 'call' 或 'put'，默认为'call'
    is_american : bool, optional
        是否为美式期权，默认为True
    return_tree : bool, optional
        是否返回价格树，默认为False

    返回:
    -------
    price : float or np.ndarray
        期权价格
    price_tree : np.ndarray (仅当return_tree=True)
        期权价格树

    示例:
    --------
    >>> binary_tree_model(S=100, K=100, T=1, r=0.05, sigma=0.2, N=100)
    10.4234

    注意:
    -----
    - 支持向量化计算：S和K可以是标量或相同形状的数组
    - 增加了输入验证和错误处理
    """
    # 参数验证
    option_type = option_type.lower()
    if option_type not in ['call', 'put']:
        raise ValueError(f"option_type必须是'call'或'put'，当前为: {option_type}")

    if T <= 0:
        raise ValueError("到期时间必须大于0")
    if sigma <= 0:
        raise ValueError("波动率必须大于0")
    if N <= 0:
        raise ValueError("期数必须大于0")

    # 将输入转换为numpy数组以支持向量化
    S_arr = np.atleast_1d(np.asarray(S, dtype=np.float64))
    K_arr = np.atleast_1d(np.asarray(K, dtype=np.float64))

    # 检查形状一致性
    if S_arr.shape != K_arr.shape:
        # 尝试广播
        try:
            S_arr, K_arr = np.broadcast_arrays(S_arr, K_arr)
        except ValueError:
            raise ValueError("S和K的形状不兼容")

    # 检查参数有效性
    if np.any(S_arr <= 0):
        raise ValueError("股票价格必须大于0")
    if np.any(K_arr <= 0):
        raise ValueError("行权价格必须大于0")

    # 计算二叉树参数
    dt = T / N  # 每一期的时间
    u = np.exp(sigma * np.sqrt(dt))  # 上涨因子
    d = np.exp(-sigma * np.sqrt(dt))  # 下跌因子
    p = (np.exp(r * dt) - d) / (u - d)  # 风险中性概率

    # 检查概率有效性
    if p <= 0 or p >= 1:
        warnings.warn(f"风险中性概率p={p:.4f}不在(0,1)范围内，计算结果可能不可靠")

    # 预计算贴现因子
    discount_factor = np.exp(-r * dt)

    # 处理向量化输入
    original_shape = S_arr.shape
    S_flat = S_arr.ravel()
    K_flat = K_arr.ravel()
    results = np.zeros_like(S_flat)

    # 为每个输入计算期权价格
    for idx in range(len(S_flat)):
        S_i = S_flat[idx]
        K_i = K_flat[idx]

        # 构建股票价格树
        stock_price_tree = np.zeros((N + 1, N + 1))
        for i in range(N + 1):  # 第i步
            for j in range(i + 1):  # 可能下跌j次、上涨i-j次
                stock_price_tree[j, i] = S_i * (d ** j) * (u ** (i - j))

        # 构建期权价格树
        option_price_tree = np.zeros((N + 1, N + 1))

        # 计算最后一期的期权价格
        if option_type == 'call':
            option_price_tree[:, N] = np.maximum(
                0, stock_price_tree[:, N] - K_i
            )
        else:  # put
            option_price_tree[:, N] = np.maximum(
                0, K_i - stock_price_tree[:, N]
            )

        # 逐步向后计算期权价格
        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                # 计算持有价值
                holding_value = discount_factor * (
                    p * option_price_tree[j, i + 1] +
                    (1 - p) * option_price_tree[j + 1, i + 1]
                )

                if is_american:  # 美式期权
                    if option_type == 'call':
                        exercise_value = max(0, stock_price_tree[j, i] - K_i)
                    else:  # put
                        exercise_value = max(0, K_i - stock_price_tree[j, i])
                    option_price_tree[j, i] = max(exercise_value, holding_value)
                else:  # 欧式期权
                    option_price_tree[j, i] = holding_value

        results[idx] = option_price_tree[0, 0]

    # 恢复原始形状
    results = results.reshape(original_shape)

    if return_tree:
        # 只返回第一个输入的价格树（为了简化）
        return (results[0] if results.size == 1 else results,
                option_price_tree)
    else:
        return results[0] if results.size == 1 else results


def binary_tree_model_with_strike_ratio(
    S: Union[float, np.ndarray],
    KS: Union[float, np.ndarray],
    T: float,
    r: float,
    sigma: float,
    N: int,
    option_type: str = 'call',
    is_american: bool = True,
    return_tree: bool = False
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    使用行权比价的二叉树模型（与原notebook兼容）

    参数:
    ----------
    S : float or np.ndarray
        初始股票价格
    KS : float or np.ndarray
        行权比价（行权价格与当前价格的比例）
    T : float
        到期时间（年）
    r : float
        无风险利率（年化）
    sigma : float
        波动率（年化）
    N : int
        期数（步数）
    option_type : str, optional
        期权类型: 'call' 或 'put'，默认为'call'
    is_american : bool, optional
        是否为美式期权，默认为True
    return_tree : bool, optional
        是否返回价格树，默认为False

    返回:
    -------
    price : float or np.ndarray
        期权价格
    price_tree : np.ndarray (可选)

    示例:
    --------
    >>> binary_tree_model_with_strike_ratio(S=100, KS=1.0, T=1, r=0.05, sigma=0.2, N=100)
    10.4234
    """
    # 计算行权价格
    K = S * KS
    return binary_tree_model(S, K, T, r, sigma, N, option_type, is_american, return_tree)
