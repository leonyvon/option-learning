# -*- coding: utf-8 -*-
"""
辅助函数

提供各种通用辅助函数，包括修复的find_closest_number函数。
"""

import io
import sys
import numpy as np
import pandas as pd
from typing import Union, List, Optional, Tuple, Any
import warnings
from datetime import datetime, timedelta

# 设置默认编码
if sys.version_info[0] == 3 and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (ValueError, AttributeError):
        pass


def find_nearest_value(x: float, numbers: Union[List[float], np.ndarray]) -> float:
    """
    在数组中寻找最接近给定值的数（修复原notebook中的bug）

    参数:
    ----------
    x : float
        目标值
    numbers : Union[List[float], np.ndarray]
        数值数组

    返回:
    -------
    nearest : float
        最接近的数

    示例:
    --------
    >>> find_nearest_value(100.5, [95, 100, 105])
    100.0

    原notebook中的错误:
    ------------------
    原函数find_closest_number存在逻辑错误：
    diff = x-num  # 计算差的绝对值 if
    if abs(diff) <abs(closest - x):  # 如果差小于当前最小差
    这里closest初始化为float('inf')，closest - x也是inf，所以比较无效

    修复后的逻辑：
    1. 计算绝对差值
    2. 找到最小差值的索引
    3. 返回对应的数
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


def validate_stock_data(
    df: pd.DataFrame,
    required_columns: List[str] = None,
    date_col: str = 'time'
) -> pd.DataFrame:
    """
    验证股票数据框

    参数:
    ----------
    df : pd.DataFrame
        股票数据框
    required_columns : List[str], optional
        必需列名列表，默认为['close', 'changeRatio']
    date_col : str, optional
        日期列名，默认为'time'

    返回:
    -------
    validated_df : pd.DataFrame
        验证后的数据框
    """
    if required_columns is None:
        required_columns = ['close', 'changeRatio']

    # 检查必需列
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame缺少必需列: {missing_columns}")

    # 检查日期列
    if date_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except Exception as e:
                raise ValueError(f"无法将列'{date_col}'转换为日期时间格式: {e}")

    # 检查数据有效性
    if df['close'].isnull().any():
        warnings.warn("close列包含空值，将尝试填充", UserWarning)
        df['close'] = df['close'].fillna(method='ffill').fillna(method='bfill')

    return df


def calculate_rolling_volatility(
    returns: pd.Series,
    window: int = 126,
    annualization_factor: float = 252.0,
    min_periods: int = None
) -> pd.Series:
    """
    计算滚动波动率

    参数:
    ----------
    returns : pd.Series
        收益率序列
    window : int, optional
        滚动窗口期，默认为126
    annualization_factor : float, optional
        年化因子，默认为252.0
    min_periods : int, optional
        最小计算期数，默认为window的一半

    返回:
    -------
    volatility : pd.Series
        波动率序列
    """
    if min_periods is None:
        min_periods = max(1, window // 2)

    # 计算滚动标准差
    rolling_std = returns.rolling(window=window, min_periods=min_periods).std()

    # 年化
    volatility = rolling_std * np.sqrt(annualization_factor)

    return volatility


def calculate_turnover_adjustment_factor(
    turnover_series: pd.Series,
    window: int = 252,
    threshold: float = 1.0,
    adjustment_coefficient: float = 0.5
) -> float:
    """
    计算换手率调整系数

    参数:
    ----------
    turnover_series : pd.Series
        换手率序列
    window : int, optional
        滚动窗口期，默认为252
    threshold : float, optional
        阈值，默认为1.0
    adjustment_coefficient : float, optional
        调整系数，默认为0.5

    返回:
    -------
    adjustment_factor : float
        调整系数
    """
    if len(turnover_series) < window:
        warnings.warn(f"数据长度({len(turnover_series)})小于窗口期({window})，使用全部数据", UserWarning)
        turnover_mean = turnover_series.mean()
    else:
        turnover_mean = turnover_series.rolling(window).mean().iloc[-1]

    if pd.isna(turnover_mean):
        return 0.0

    if turnover_mean < threshold:
        return (1 - turnover_mean / threshold) * adjustment_coefficient
    else:
        return 0.0


def prepare_option_pricing_data(
    df: pd.DataFrame,
    current_date: datetime,
    expiration_date: datetime,
    required_columns: List[str] = None
) -> pd.DataFrame:
    """
    准备期权定价数据

    参数:
    ----------
    df : pd.DataFrame
        原始数据框
    current_date : datetime
        当前日期
    expiration_date : datetime
        到期日期
    required_columns : List[str], optional
        必需列名列表

    返回:
    -------
    prepared_df : pd.DataFrame
        准备后的数据框
    """
    if required_columns is None:
        required_columns = ['close', 'sigma', 'r']

    # 复制数据框
    prepared_df = df.copy()

    # 计算到期时间（年）
    prepared_df['t'] = pd.Series(prepared_df.index).apply(
        lambda x: (expiration_date - x).days / 365.0
    ).tolist()

    # 确保必需列存在
    for col in required_columns:
        if col not in prepared_df.columns:
            if col == 'r':
                # 如果没有利率列，添加默认值
                prepared_df[col] = 0.03
            elif col == 'sigma':
                # 如果没有波动率列，计算历史波动率
                if 'changeRatio' in prepared_df.columns:
                    prepared_df[col] = calculate_rolling_volatility(
                        prepared_df['changeRatio']
                    )
                else:
                    raise ValueError(f"缺少必需列: {col}，且无法计算")

    # 过滤掉到期时间为负数的行
    prepared_df = prepared_df[prepared_df['t'] > 0]

    return prepared_df


def create_strike_price_list(
    bottom: float,
    top: float,
    step: float = 0.05
) -> List[float]:
    """
    创建行权价格列表

    参数:
    ----------
    bottom : float
        最低行权价
    top : float
        最高行权价
    step : float, optional
        步长，默认为0.05

    返回:
    -------
    strike_list : List[float]
        行权价格列表

    示例:
    --------
    >>> create_strike_price_list(2, 4, 0.05)
    [2.0, 2.05, 2.1, ..., 3.95, 4.0]
    """
    strike_list = []
    current = bottom
    while current < top:
        strike_list.append(round(current, 2))
        current = round(current + step, 2)

    return strike_list


def calculate_days_to_expiration(
    current_date: Union[datetime, str],
    expiration_date: Union[datetime, str],
    trading_days_per_year: float = 252.0
) -> float:
    """
    计算到期时间（年）

    参数:
    ----------
    current_date : Union[datetime, str]
        当前日期
    expiration_date : Union[datetime, str]
        到期日期
    trading_days_per_year : float, optional
        年交易日数，默认为252.0

    返回:
    -------
    time_to_expiration : float
        到期时间（年）
    """
    # 转换为datetime
    if isinstance(current_date, str):
        current_date = pd.to_datetime(current_date)
    if isinstance(expiration_date, str):
        expiration_date = pd.to_datetime(expiration_date)

    # 计算天数差
    days_diff = (expiration_date - current_date).days

    if days_diff < 0:
        raise ValueError("到期日期不能早于当前日期")

    # 转换为年
    return days_diff / 365.0


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零错误

    参数:
    ----------
    numerator : float
        分子
    denominator : float
        分母
    default : float, optional
        分母为0时的默认值，默认为0.0

    返回:
    -------
    result : float
        除法结果
    """
    if denominator == 0:
        return default
    return numerator / denominator


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    格式化百分比

    参数:
    ----------
    value : float
        原始值
    decimals : int, optional
        小数位数，默认为2

    返回:
    -------
    formatted : str
        格式化后的百分比字符串
    """
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2) -> str:
    """
    格式化货币金额

    参数:
    ----------
    value : float
        原始值
    decimals : int, optional
        小数位数，默认为2

    返回:
    -------
    formatted : str
        格式化后的货币字符串
    """
    if abs(value) >= 1e8:  # 亿
        return f"{value / 1e8:.{decimals}f}亿元"
    elif abs(value) >= 1e4:  # 万
        return f"{value / 1e4:.{decimals}f}万元"
    else:
        return f"{value:.{decimals}f}元"


def check_numerical_stability(
    values: np.ndarray,
    threshold: float = 1e-10,
    name: str = "数值"
) -> bool:
    """
    检查数值稳定性

    参数:
    ----------
    values : np.ndarray
        数值数组
    threshold : float, optional
        阈值，默认为1e-10
    name : str, optional
        数值名称，用于警告信息

    返回:
    -------
    is_stable : bool
        数值是否稳定
    """
    # 检查NaN
    if np.any(np.isnan(values)):
        warnings.warn(f"{name}包含NaN值", UserWarning)
        return False

    # 检查无穷大
    if np.any(np.isinf(values)):
        warnings.warn(f"{name}包含无穷大值", UserWarning)
        return False

    # 检查过小值
    abs_values = np.abs(values)
    if np.any(abs_values > 0) and np.any(abs_values < threshold):
        warnings.warn(f"{name}包含极小的非零值（可能影响数值稳定性）", UserWarning)

    return True