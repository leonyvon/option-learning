# -*- coding: utf-8 -*-
"""
数据获取模块

提供统一的数据获取接口，封装akshare等数据源调用。
"""

import io
import sys
import numpy as np
import pandas as pd
import akshare as ak
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import warnings
import os

# 设置默认编码
if sys.version_info[0] == 3 and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (ValueError, AttributeError):
        pass

# 导入辅助函数
from ..utils.helpers import validate_stock_data, calculate_rolling_volatility


def get_stocks_data(
    start_date: str,
    end_date: str,
    stock_codes: List[str],
    adjust: str = "qfq",
    cache_dir: Optional[str] = None,
    force_update: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    获取多只股票数据

    参数:
    ----------
    start_date : str
        开始日期，格式: 'YYYYMMDD'
    end_date : str
        结束日期，格式: 'YYYYMMDD'
    stock_codes : List[str]
        股票代码列表，例如: ['603088.SH', '300723.SZ']
    adjust : str, optional
        复权类型: 'qfq'(前复权), 'hfq'(后复权), ''(不复权)，默认为'qfq'
    cache_dir : str, optional
        缓存目录，如果为None则不缓存
    force_update : bool, optional
        是否强制更新缓存，默认为False

    返回:
    -------
    data_dict : Dict[str, pd.DataFrame]
        股票数据字典，键为股票代码，值为DataFrame

    示例:
    --------
    >>> data = get_stocks_data('20200101', '20230714', ['603088.SH', '300723.SZ'])
    """
    data_dict = {}

    for stock_code in stock_codes:
        try:
            # 检查缓存
            cache_file = None
            if cache_dir and not force_update:
                cache_file = os.path.join(cache_dir, f"{stock_code}_{start_date}_{end_date}.pkl")
                if os.path.exists(cache_file):
                    df = pd.read_pickle(cache_file)
                    data_dict[stock_code] = df
                    continue

            # 获取股票数据
            df = ak.stock_zh_a_hist(
                symbol=stock_code.split('.')[0],
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )

            # 重命名列以统一格式
            column_mapping = {
                '日期': 'time',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'changeRatio',
                '涨跌额': 'change',
                '换手率': 'turnoverRatio'
            }

            df = df.rename(columns=column_mapping)

            # 转换日期列
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)

            # 计算波动率（如果需要）
            if 'changeRatio' in df.columns:
                df['sigma'] = calculate_rolling_volatility(
                    df['changeRatio'],
                    window=126,
                    annualization_factor=252.0
                )

            # 验证数据
            df = validate_stock_data(df)

            # 保存到缓存
            if cache_file:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                df.to_pickle(cache_file)

            data_dict[stock_code] = df

        except Exception as e:
            warnings.warn(f"获取股票{stock_code}数据失败: {e}", UserWarning)
            continue

    return data_dict


def get_index_data(
    index_symbol: str,
    start_date: str,
    end_date: str,
    cache_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    获取指数数据

    参数:
    ----------
    index_symbol : str
        指数代码，例如: 'sh000905' (中证500), 'sh000852' (中证1000)
    start_date : str
        开始日期，格式: 'YYYYMMDD'
    end_date : str
        结束日期，格式: 'YYYYMMDD'
    cache_dir : str, optional
        缓存目录

    返回:
    -------
    df : pd.DataFrame
        指数数据框
    """
    try:
        # 检查缓存
        cache_file = None
        if cache_dir:
            cache_file = os.path.join(cache_dir, f"index_{index_symbol}_{start_date}_{end_date}.pkl")
            if os.path.exists(cache_file):
                return pd.read_pickle(cache_file)

        # 获取指数数据
        df = ak.stock_zh_index_daily(symbol=index_symbol)

        # 过滤日期
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= pd.to_datetime(start_date)) &
                (df['date'] <= pd.to_datetime(end_date))]

        # 重命名列
        df = df.rename(columns={'date': 'time'})
        df.set_index('time', inplace=True)

        # 计算收益率和波动率
        df['changeRatio'] = df['close'].pct_change()
        df['sigma'] = calculate_rolling_volatility(
            df['changeRatio'],
            window=126,
            annualization_factor=252.0
        )

        # 保存到缓存
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            df.to_pickle(cache_file)

        return df

    except Exception as e:
        warnings.warn(f"获取指数{index_symbol}数据失败: {e}", UserWarning)
        raise


def get_etf_data(
    etf_symbol: str,
    start_date: str,
    end_date: str,
    period: str = "daily",
    adjust: str = "qfq"
) -> pd.DataFrame:
    """
    获取ETF数据

    参数:
    ----------
    etf_symbol : str
        ETF代码，例如: '510050' (50ETF)
    start_date : str
        开始日期
    end_date : str
        结束日期
    period : str, optional
        周期: 'daily' (日线), 'weekly' (周线), 'monthly' (月线)，默认为'daily'
    adjust : str, optional
        复权类型，默认为'qfq'

    返回:
    -------
    df : pd.DataFrame
        ETF数据框
    """
    try:
        df = ak.fund_etf_hist_em(
            symbol=etf_symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )

        # 重命名列
        column_mapping = {
            '日期': 'time',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'changeRatio',
            '涨跌额': 'change',
            '换手率': 'turnoverRatio'
        }

        df = df.rename(columns=column_mapping)

        # 转换日期列
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        # 计算波动率
        if 'changeRatio' in df.columns:
            df['sigma'] = calculate_rolling_volatility(
                df['changeRatio'],
                window=126,
                annualization_factor=252.0
            )

        return df

    except Exception as e:
        warnings.warn(f"获取ETF{etf_symbol}数据失败: {e}", UserWarning)
        raise


def get_interest_rate_data(
    market: str = "上海银行同业拆借市场",
    symbol: str = "Shibor人民币",
    indicator: str = "1年",
    cache_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    获取利率数据

    参数:
    ----------
    market : str, optional
        市场类型，默认为"上海银行同业拆借市场"
    symbol : str, optional
        货币类型，默认为"Shibor人民币"
    indicator : str, optional
        期限，默认为"1年"
    cache_dir : str, optional
        缓存目录

    返回:
    -------
    df : pd.DataFrame
        利率数据框
    """
    try:
        # 检查缓存
        cache_file = None
        if cache_dir:
            cache_file = os.path.join(cache_dir, f"interest_rate_{market}_{symbol}_{indicator}.pkl")
            if os.path.exists(cache_file):
                return pd.read_pickle(cache_file)

        # 获取利率数据
        df = ak.rate_interbank(
            market=market,
            symbol=symbol,
            indicator=indicator
        )

        # 重命名列
        df.columns = ['time', 'r', 'changeRatio']

        # 转换日期和利率
        df['time'] = pd.to_datetime(df['time'])
        df['r'] = df['r'] / 100  # 转换为小数形式
        df.set_index('time', inplace=True)

        # 保存到缓存
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            df.to_pickle(cache_file)

        return df

    except Exception as e:
        warnings.warn(f"获取利率数据失败: {e}", UserWarning)
        raise


def merge_market_data(
    price_data: pd.DataFrame,
    interest_rate_data: pd.DataFrame,
    how: str = 'inner'
) -> pd.DataFrame:
    """
    合并市场价格数据和利率数据

    参数:
    ----------
    price_data : pd.DataFrame
        价格数据框（必须有日期索引）
    interest_rate_data : pd.DataFrame
        利率数据框（必须有日期索引）
    how : str, optional
        合并方式: 'inner', 'outer', 'left', 'right'，默认为'inner'

    返回:
    -------
    merged_df : pd.DataFrame
        合并后的数据框
    """
    # 确保两个数据框的索引都是日期类型
    if not isinstance(price_data.index, pd.DatetimeIndex):
        price_data.index = pd.to_datetime(price_data.index)

    if not isinstance(interest_rate_data.index, pd.DatetimeIndex):
        interest_rate_data.index = pd.to_datetime(interest_rate_data.index)

    # 合并数据
    merged_df = pd.merge(
        price_data,
        interest_rate_data[['r']],
        left_index=True,
        right_index=True,
        how=how
    )

    # 前向填充利率数据（利率数据通常比价格数据稀疏）
    if 'r' in merged_df.columns:
        merged_df['r'] = merged_df['r'].fillna(method='ffill')

    return merged_df


def prepare_option_pricing_dataset(
    etf_symbol: str = "510050",
    start_date: str = "20200101",
    end_date: str = "20230705",
    interest_rate_indicator: str = "1年"
) -> pd.DataFrame:
    """
    准备期权定价数据集（整合ETF价格和利率）

    参数:
    ----------
    etf_symbol : str, optional
        ETF代码，默认为'510050'
    start_date : str, optional
        开始日期，默认为'20200101'
    end_date : str, optional
        结束日期，默认为'20230705'
    interest_rate_indicator : str, optional
        利率期限，默认为'1年'

    返回:
    -------
    df : pd.DataFrame
        准备好的数据集
    """
    # 获取ETF数据
    etf_data = get_etf_data(etf_symbol, start_date, end_date)

    # 获取利率数据
    interest_data = get_interest_rate_data(indicator=interest_rate_indicator)

    # 合并数据
    merged_data = merge_market_data(etf_data, interest_data, how='left')

    # 填充缺失的利率数据
    if merged_data['r'].isnull().any():
        # 使用最近的利率或默认值
        default_rate = 0.03  # 3%默认利率
        merged_data['r'] = merged_data['r'].fillna(method='ffill').fillna(default_rate)

    return merged_data


def calculate_historical_volatility_series(
    df: pd.DataFrame,
    price_col: str = 'close',
    window: int = 126,
    annualization_factor: float = 252.0
) -> pd.Series:
    """
    计算历史波动率序列

    参数:
    ----------
    df : pd.DataFrame
        价格数据框
    price_col : str, optional
        价格列名，默认为'close'
    window : int, optional
        滚动窗口期，默认为126
    annualization_factor : float, optional
        年化因子，默认为252.0

    返回:
    -------
    volatility_series : pd.Series
        波动率序列
    """
    if price_col not in df.columns:
        raise ValueError(f"数据框中没有列: {price_col}")

    # 计算收益率
    returns = df[price_col].pct_change()

    # 计算滚动波动率
    volatility = calculate_rolling_volatility(
        returns,
        window=window,
        annualization_factor=annualization_factor
    )

    return volatility


def create_option_expiration_dates(
    start_date: datetime,
    end_date: datetime,
    frequency: str = 'monthly'
) -> List[datetime]:
    """
    创建期权到期日期列表

    参数:
    ----------
    start_date : datetime
        开始日期
    end_date : datetime
        结束日期
    frequency : str, optional
        频率: 'monthly' (每月), 'weekly' (每周)，默认为'monthly'

    返回:
    -------
    expiration_dates : List[datetime]
        到期日期列表
    """
    expiration_dates = []

    if frequency == 'monthly':
        # 每月到期（通常为当月第四个星期三，这里简化为当月最后一天）
        current_date = start_date.replace(day=1)
        while current_date <= end_date:
            # 当月最后一天
            if current_date.month == 12:
                next_month = current_date.replace(year=current_date.year + 1, month=1, day=1)
            else:
                next_month = current_date.replace(month=current_date.month + 1, day=1)

            last_day = next_month - timedelta(days=1)
            expiration_dates.append(last_day)
            current_date = next_month

    elif frequency == 'weekly':
        # 每周到期（星期三）
        current_date = start_date
        while current_date <= end_date:
            # 找到下一个星期三
            days_ahead = 2 - current_date.weekday()  # 2代表星期三
            if days_ahead <= 0:
                days_ahead += 7
            next_wednesday = current_date + timedelta(days=days_ahead)

            if next_wednesday <= end_date:
                expiration_dates.append(next_wednesday)

            current_date = next_wednesday + timedelta(days=1)

    else:
        raise ValueError(f"不支持的频率: {frequency}")

    return expiration_dates