# -*- coding: utf-8 -*-
"""
输入验证工具

提供统一的输入验证和错误处理。
"""

import io
import sys
import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Any, Dict, List
import warnings

# 设置默认编码
if sys.version_info[0] == 3 and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (ValueError, AttributeError):
        pass


def validate_option_type(option_type: str, allowed_types: List[str] = None) -> str:
    """
    验证期权类型

    参数:
    ----------
    option_type : str
        期权类型
    allowed_types : List[str], optional
        允许的类型列表，默认为['call', 'put']

    返回:
    -------
    validated_type : str
        验证后的期权类型（小写）

    异常:
    ------
    ValueError: 当期权类型无效时
    """
    if allowed_types is None:
        allowed_types = ['call', 'put']

    validated = option_type.lower()
    if validated not in allowed_types:
        raise ValueError(
            f"期权类型必须是{allowed_types}之一，当前为: {option_type}"
        )
    return validated


def validate_positive_number(
    value: Union[int, float, np.ndarray],
    name: str,
    allow_zero: bool = False,
    allow_inf: bool = False
) -> Union[float, np.ndarray]:
    """
    验证正数

    参数:
    ----------
    value : Union[int, float, np.ndarray]
        要验证的值
    name : str
        参数名称（用于错误信息）
    allow_zero : bool, optional
        是否允许零值，默认为False
    allow_inf : bool, optional
        是否允许无穷大，默认为False

    返回:
    -------
    validated_value : Union[float, np.ndarray]
        验证后的值

    异常:
    ------
    ValueError: 当值无效时
    """
    value_arr = np.asarray(value, dtype=np.float64)

    # 检查NaN
    if np.any(np.isnan(value_arr)):
        raise ValueError(f"{name}不能包含NaN值")

    # 检查无穷大
    if not allow_inf and np.any(np.isinf(value_arr)):
        raise ValueError(f"{name}不能包含无穷大值")

    # 检查正数
    if allow_zero:
        if np.any(value_arr < 0):
            raise ValueError(f"{name}不能为负数")
    else:
        if np.any(value_arr <= 0):
            raise ValueError(f"{name}必须大于0")

    return value_arr.item() if value_arr.size == 1 else value_arr


def validate_percentage(
    value: Union[int, float, np.ndarray],
    name: str,
    min_value: float = 0.0,
    max_value: float = 1.0,
    allow_outside: bool = False
) -> Union[float, np.ndarray]:
    """
    验证百分比值

    参数:
    ----------
    value : Union[int, float, np.ndarray]
        要验证的值
    name : str
        参数名称（用于错误信息）
    min_value : float, optional
        最小值，默认为0.0
    max_value : float, optional
        最大值，默认为1.0
    allow_outside : bool, optional
        是否允许超出范围，默认为False（仅警告）

    返回:
    -------
    validated_value : Union[float, np.ndarray]
        验证后的值

    异常:
    ------
    ValueError: 当值无效且allow_outside=False时
    """
    value_arr = np.asarray(value, dtype=np.float64)

    # 检查NaN和无穷大
    if np.any(np.isnan(value_arr)):
        raise ValueError(f"{name}不能包含NaN值")

    # 检查范围
    if np.any(value_arr < min_value) or np.any(value_arr > max_value):
        if allow_outside:
            warnings.warn(
                f"{name}的值({value})超出了建议范围[{min_value}, {max_value}]",
                UserWarning
            )
        else:
            raise ValueError(
                f"{name}必须在[{min_value}, {max_value}]范围内，当前为: {value}"
            )

    return value_arr.item() if value_arr.size == 1 else value_arr


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    allow_missing: bool = False,
    index_type: Optional[str] = None
) -> pd.DataFrame:
    """
    验证DataFrame

    参数:
    ----------
    df : pd.DataFrame
        要验证的DataFrame
    required_columns : List[str]
        必需的列名列表
    allow_missing : bool, optional
        是否允许缺失列，默认为False
    index_type : str, optional
        索引类型检查，可选值: 'datetime' 或 None

    返回:
    -------
    validated_df : pd.DataFrame
        验证后的DataFrame

    异常:
    ------
    ValueError: 当DataFrame无效时
    TypeError: 当df不是DataFrame时
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"输入必须是pandas DataFrame，当前类型为: {type(df)}")

    # 检查必需列
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns and not allow_missing:
        raise ValueError(f"DataFrame缺少必需列: {missing_columns}")

    # 检查索引类型
    if index_type == 'datetime':
        if not isinstance(df.index, pd.DatetimeIndex):
            # 尝试转换
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"无法将索引转换为DatetimeIndex: {e}")

    return df


def validate_strike_ratio(
    strike_ratio: Union[float, np.ndarray],
    name: str = 'strike_ratio'
) -> Union[float, np.ndarray]:
    """
    验证行权比价

    参数:
    ----------
    strike_ratio : Union[float, np.ndarray]
        行权比价
    name : str, optional
        参数名称，默认为'strike_ratio'

    返回:
    -------
    validated_ratio : Union[float, np.ndarray]
        验证后的行权比价

    异常:
    ------
    ValueError: 当行权比价无效时
    """
    return validate_positive_number(strike_ratio, name, allow_zero=False)


def validate_time_to_maturity(
    T: Union[float, np.ndarray],
    name: str = 'T'
) -> Union[float, np.ndarray]:
    """
    验证到期时间

    参数:
    ----------
    T : Union[float, np.ndarray]
        到期时间（年）
    name : str, optional
        参数名称，默认为'T'

    返回:
    -------
    validated_T : Union[float, np.ndarray]
        验证后的到期时间

    异常:
    ------
    ValueError: 当到期时间无效时
    """
    return validate_positive_number(T, name, allow_zero=True)


def validate_volatility(
    sigma: Union[float, np.ndarray],
    name: str = 'sigma'
) -> Union[float, np.ndarray]:
    """
    验证波动率

    参数:
    ----------
    sigma : Union[float, np.ndarray]
        波动率
    name : str, optional
        参数名称，默认为'sigma'

    返回:
    -------
    validated_sigma : Union[float, np.ndarray]
        验证后的波动率

    异常:
    ------
    ValueError: 当波动率无效时
    """
    sigma_arr = np.asarray(sigma, dtype=np.float64)

    # 检查NaN
    if np.any(np.isnan(sigma_arr)):
        raise ValueError(f"{name}不能包含NaN值")

    # 检查负数
    if np.any(sigma_arr < 0):
        raise ValueError(f"{name}不能为负数")

    # 警告过高波动率
    if np.any(sigma_arr > 2.0):  # 超过200%波动率
        warnings.warn(
            f"{name}的值({sigma})超过200%，这可能不是合理的市场波动率",
            UserWarning
        )

    return sigma_arr.item() if sigma_arr.size == 1 else sigma_arr


def validate_interest_rate(
    r: float,
    name: str = 'r'
) -> float:
    """
    验证利率

    参数:
    ----------
    r : float
        利率
    name : str, optional
        参数名称，默认为'r'

    返回:
    -------
    validated_r : float
        验证后的利率

    异常:
    ------
    ValueError: 当利率无效时
    """
    r_float = float(r)

    # 检查NaN和无穷大
    if np.isnan(r_float) or np.isinf(r_float):
        raise ValueError(f"{name}不能为NaN或无穷大")

    # 警告异常利率
    if r_float < -0.1 or r_float > 0.5:  # 负10%到50%的合理范围
        warnings.warn(
            f"{name}的值({r_float})可能不在合理市场范围内",
            UserWarning
        )

    return r_float


def validate_pricing_parameters(
    S0: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: float,
    sigma: Union[float, np.ndarray],
    q: float = 0.0
) -> Tuple[
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    float,
    Union[float, np.ndarray],
    float
]:
    """
    验证定价参数

    参数:
    ----------
    S0 : Union[float, np.ndarray]
        标的资产价格
    K : Union[float, np.ndarray]
        行权价格
    T : Union[float, np.ndarray]
        到期时间
    r : float
        无风险利率
    sigma : Union[float, np.ndarray]
        波动率
    q : float, optional
        分红率，默认为0.0

    返回:
    -------
    validated_params : tuple
        验证后的参数元组 (S0, K, T, r, sigma, q)
    """
    S0_valid = validate_positive_number(S0, 'S0')
    K_valid = validate_positive_number(K, 'K')
    T_valid = validate_time_to_maturity(T)
    r_valid = validate_interest_rate(r)
    sigma_valid = validate_volatility(sigma)
    q_valid = validate_interest_rate(q) if q != 0.0 else 0.0

    # 检查S0和K形状一致性
    S0_arr = np.atleast_1d(S0_valid)
    K_arr = np.atleast_1d(K_valid)

    if S0_arr.shape != K_arr.shape:
        try:
            S0_arr, K_arr = np.broadcast_arrays(S0_arr, K_arr)
        except ValueError:
            raise ValueError("S0和K的形状不兼容")

    return (
        S0_arr.item() if S0_arr.size == 1 else S0_arr,
        K_arr.item() if K_arr.size == 1 else K_arr,
        T_valid,
        r_valid,
        sigma_valid,
        q_valid
    )


def validate_product_parameters(
    asset: float,
    participation_rate: float,
    capital_protection_rate: float,
    T: float
) -> Tuple[float, float, float, float]:
    """
    验证产品参数

    参数:
    ----------
    asset : float
        资产规模
    participation_rate : float
        参与率
    capital_protection_rate : float
        保本率
    T : float
        期限

    返回:
    -------
    validated_params : tuple
        验证后的参数元组
    """
    asset_valid = validate_positive_number(asset, 'asset')
    participation_valid = validate_percentage(
        participation_rate, 'participation_rate', min_value=0.0, max_value=5.0, allow_outside=True
    )
    capital_protection_valid = validate_percentage(
        capital_protection_rate, 'capital_protection_rate', min_value=0.0, max_value=1.5, allow_outside=True
    )
    T_valid = validate_time_to_maturity(T)

    return asset_valid, participation_valid, capital_protection_valid, T_valid