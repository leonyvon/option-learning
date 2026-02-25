# -*- coding: utf-8 -*-
"""
期权价差策略

提供各种期权价差策略的收益计算和可视化，包括：
- 牛市价差
- 熊市价差
- 盒式价差
- 蝶式价差
"""

import io
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import warnings

# 设置默认编码
if sys.version_info[0] == 3 and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (ValueError, AttributeError):
        pass

# 导入配置和验证工具
from ..utils.config import default_config
from ..utils.validation import validate_positive_number


class BullSpreadStrategy:
    """牛市价差策略"""

    def __init__(
        self,
        K_buy: float,
        K_sell: float,
        c_buy: float,
        c_sell: float,
        config: Optional[Any] = None
    ):
        """
        初始化牛市价差策略

        参数:
        ----------
        K_buy : float
            买入看涨期权行权价
        K_sell : float
            卖出看涨期权行权价
        c_buy : float
            买入看涨期权价格
        c_sell : float
            卖出看涨期权价格
        config : Any, optional
            配置对象，默认为default_config
        """
        self.config = config or default_config

        # 验证参数
        self.K_buy = validate_positive_number(K_buy, 'K_buy')
        self.K_sell = validate_positive_number(K_sell, 'K_sell')
        self.c_buy = validate_positive_number(c_buy, 'c_buy')
        self.c_sell = validate_positive_number(c_sell, 'c_sell')

        if self.K_buy >= self.K_sell:
            raise ValueError("买入行权价必须小于卖出行权价")

        # 初始化计算结果
        self.results = {}
        self.revenue_df = None

    def calculate_revenue(
        self,
        stock_prices: np.ndarray,
        volume: int = 10000
    ) -> np.ndarray:
        """
        计算策略收益

        参数:
        ----------
        stock_prices : np.ndarray
            股票价格序列
        volume : int, optional
            期权交易量（合约数），默认为10000

        返回:
        -------
        revenue : np.ndarray
            收益序列
        """
        # 计算买入看涨期权收益
        revenue_buy = np.where(
            self.K_buy <= stock_prices,
            (stock_prices - self.K_buy) * volume - self.c_buy * volume,
            -self.c_buy * volume
        )

        # 计算卖出看涨期权收益
        revenue_sell = np.where(
            self.K_sell <= stock_prices,
            (self.K_sell - stock_prices) * volume + self.c_sell * volume,
            self.c_sell * volume
        )

        # 总收益
        revenue = revenue_buy + revenue_sell

        # 保存结果
        self.results['revenue'] = revenue
        self.results['revenue_buy'] = revenue_buy
        self.results['revenue_sell'] = revenue_sell
        self.results['stock_prices'] = stock_prices
        self.results['volume'] = volume

        return revenue

    def calculate_breakeven_points(self) -> Tuple[float, float]:
        """
        计算盈亏平衡点

        返回:
        -------
        lower_breakeven : float
            下盈亏平衡点
        upper_breakeven : float
            上盈亏平衡点
        """
        # 净期权费
        net_premium = self.c_sell - self.c_buy

        # 盈亏平衡点
        lower_breakeven = self.K_buy + net_premium
        upper_breakeven = self.K_sell + net_premium

        return lower_breakeven, upper_breakeven

    def calculate_max_min_profit(self) -> Tuple[float, float]:
        """
        计算最大和最小收益

        返回:
        -------
        max_profit : float
            最大收益
        min_profit : float
            最小收益
        """
        # 净期权费
        net_premium = self.c_sell - self.c_buy

        # 最大收益（股价高于卖出行权价）
        max_profit = (self.K_sell - self.K_buy) * 10000 + net_premium * 10000

        # 最小收益（股价低于买入行权价）
        min_profit = net_premium * 10000

        return max_profit, min_profit

    def create_revenue_dataframe(
        self,
        price_range: Optional[Tuple[float, float]] = None,
        step: float = 1.0
    ) -> pd.DataFrame:
        """
        创建收益数据框

        参数:
        ----------
        price_range : Tuple[float, float], optional
            价格范围，默认为[K_buy*0.5, K_sell*1.5]
        step : float, optional
            价格步长，默认为1.0

        返回:
        -------
        df : pd.DataFrame
            收益数据框
        """
        if price_range is None:
            price_min = min(self.K_buy, self.K_sell) * 0.5
            price_max = max(self.K_buy, self.K_sell) * 1.5
        else:
            price_min, price_max = price_range

        stock_prices = np.arange(price_min, price_max, step)
        underlying_returns = (stock_prices - self.K_buy) / self.K_buy

        # 计算收益
        revenue = self.calculate_revenue(stock_prices)
        revenue_ratio = revenue / (self.K_buy * 10000)  # 相对于初始投资的收益率

        # 创建数据框
        df = pd.DataFrame({
            '股价': stock_prices,
            '标的收益率': underlying_returns,
            '策略收益': revenue,
            '策略收益率': revenue_ratio
        })

        self.revenue_df = df
        return df

    def plot_payoff(self, ax=None, **plot_kwargs):
        """
        绘制损益图

        参数:
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib坐标轴对象
        **plot_kwargs : dict
            绘图参数

        返回:
        -------
        ax : matplotlib.axes.Axes
            绘图坐标轴
        """
        if self.revenue_df is None:
            self.create_revenue_dataframe()

        import matplotlib.pyplot as plt
        from ..utils.visualization import plot_strategy_payoff

        ax = plot_strategy_payoff(
            stock_prices=self.revenue_df['股价'].values,
            strategy_returns=self.revenue_df['策略收益率'].values,
            underlying_returns=self.revenue_df['标的收益率'].values,
            current_price=self.K_buy,
            strategy_name='牛市价差',
            ax=ax,
            config=self.config,
            **plot_kwargs
        )

        return ax


class BearSpreadStrategy:
    """熊市价差策略"""

    def __init__(
        self,
        K_buy: float,
        K_sell: float,
        p_buy: float,
        p_sell: float,
        config: Optional[Any] = None
    ):
        """
        初始化熊市价差策略

        参数:
        ----------
        K_buy : float
            买入看跌期权行权价
        K_sell : float
            卖出看跌期权行权价
        p_buy : float
            买入看跌期权价格
        p_sell : float
            卖出看跌期权价格
        config : Any, optional
            配置对象，默认为default_config
        """
        self.config = config or default_config

        # 验证参数
        self.K_buy = validate_positive_number(K_buy, 'K_buy')
        self.K_sell = validate_positive_number(K_sell, 'K_sell')
        self.p_buy = validate_positive_number(p_buy, 'p_buy')
        self.p_sell = validate_positive_number(p_sell, 'p_sell')

        if self.K_buy <= self.K_sell:
            raise ValueError("买入行权价必须大于卖出行权价")

        # 初始化计算结果
        self.results = {}
        self.revenue_df = None

    def calculate_revenue(
        self,
        stock_prices: np.ndarray,
        volume: int = 10000
    ) -> np.ndarray:
        """
        计算策略收益

        参数:
        ----------
        stock_prices : np.ndarray
            股票价格序列
        volume : int, optional
            期权交易量（合约数），默认为10000

        返回:
        -------
        revenue : np.ndarray
            收益序列
        """
        # 计算买入看跌期权收益
        revenue_buy = np.where(
            self.K_buy >= stock_prices,
            (self.K_buy - stock_prices) * volume - self.p_buy * volume,
            -self.p_buy * volume
        )

        # 计算卖出看跌期权收益
        revenue_sell = np.where(
            self.K_sell >= stock_prices,
            (stock_prices - self.K_sell) * volume + self.p_sell * volume,
            self.p_sell * volume
        )

        # 总收益
        revenue = revenue_buy + revenue_sell

        # 保存结果
        self.results['revenue'] = revenue
        self.results['revenue_buy'] = revenue_buy
        self.results['revenue_sell'] = revenue_sell
        self.results['stock_prices'] = stock_prices
        self.results['volume'] = volume

        return revenue

    def calculate_breakeven_points(self) -> Tuple[float, float]:
        """
        计算盈亏平衡点

        返回:
        -------
        lower_breakeven : float
            下盈亏平衡点
        upper_breakeven : float
            上盈亏平衡点
        """
        # 净期权费
        net_premium = self.p_sell - self.p_buy

        # 盈亏平衡点
        lower_breakeven = self.K_sell - net_premium
        upper_breakeven = self.K_buy - net_premium

        return lower_breakeven, upper_breakeven

    def calculate_max_min_profit(self) -> Tuple[float, float]:
        """
        计算最大和最小收益

        返回:
        -------
        max_profit : float
            最大收益
        min_profit : float
            最小收益
        """
        # 净期权费
        net_premium = self.p_sell - self.p_buy

        # 最大收益（股价低于卖出行权价）
        max_profit = (self.K_buy - self.K_sell) * 10000 + net_premium * 10000

        # 最小收益（股价高于买入行权价）
        min_profit = net_premium * 10000

        return max_profit, min_profit

    def create_revenue_dataframe(
        self,
        price_range: Optional[Tuple[float, float]] = None,
        step: float = 1.0
    ) -> pd.DataFrame:
        """
        创建收益数据框

        参数:
        ----------
        price_range : Tuple[float, float], optional
            价格范围，默认为[K_sell*0.5, K_buy*1.5]
        step : float, optional
            价格步长，默认为1.0

        返回:
        -------
        df : pd.DataFrame
            收益数据框
        """
        if price_range is None:
            price_min = min(self.K_buy, self.K_sell) * 0.5
            price_max = max(self.K_buy, self.K_sell) * 1.5
        else:
            price_min, price_max = price_range

        stock_prices = np.arange(price_min, price_max, step)
        underlying_returns = (stock_prices - self.K_sell) / self.K_sell

        # 计算收益
        revenue = self.calculate_revenue(stock_prices)
        revenue_ratio = revenue / (self.K_sell * 10000)  # 相对于初始投资的收益率

        # 创建数据框
        df = pd.DataFrame({
            '股价': stock_prices,
            '标的收益率': underlying_returns,
            '策略收益': revenue,
            '策略收益率': revenue_ratio
        })

        self.revenue_df = df
        return df

    def plot_payoff(self, ax=None, **plot_kwargs):
        """
        绘制损益图

        参数:
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib坐标轴对象
        **plot_kwargs : dict
            绘图参数

        返回:
        -------
        ax : matplotlib.axes.Axes
            绘图坐标轴
        """
        if self.revenue_df is None:
            self.create_revenue_dataframe()

        import matplotlib.pyplot as plt
        from ..utils.visualization import plot_strategy_payoff

        ax = plot_strategy_payoff(
            stock_prices=self.revenue_df['股价'].values,
            strategy_returns=self.revenue_df['策略收益率'].values,
            underlying_returns=self.revenue_df['标的收益率'].values,
            current_price=self.K_sell,
            strategy_name='熊市价差',
            ax=ax,
            config=self.config,
            **plot_kwargs
        )

        return ax


class BoxSpreadStrategy:
    """盒式价差策略（一熊一牛组合）"""

    def __init__(
        self,
        bull_spread: BullSpreadStrategy,
        bear_spread: BearSpreadStrategy,
        config: Optional[Any] = None
    ):
        """
        初始化盒式价差策略

        参数:
        ----------
        bull_spread : BullSpreadStrategy
            牛市价差策略
        bear_spread : BearSpreadStrategy
            熊市价差策略
        config : Any, optional
            配置对象，默认为default_config
        """
        self.config = config or default_config
        self.bull_spread = bull_spread
        self.bear_spread = bear_spread

        # 初始化计算结果
        self.results = {}
        self.revenue_df = None

    def calculate_revenue(
        self,
        stock_prices: np.ndarray,
        volume: int = 10000
    ) -> np.ndarray:
        """
        计算策略收益

        参数:
        ----------
        stock_prices : np.ndarray
            股票价格序列
        volume : int, optional
            期权交易量（合约数），默认为10000

        返回:
        -------
        revenue : np.ndarray
            收益序列
        """
        # 计算牛市价差收益
        bull_revenue = self.bull_spread.calculate_revenue(stock_prices, volume)

        # 计算熊市价差收益
        bear_revenue = self.bear_spread.calculate_revenue(stock_prices, volume)

        # 总收益
        revenue = bull_revenue + bear_revenue

        # 保存结果
        self.results['revenue'] = revenue
        self.results['bull_revenue'] = bull_revenue
        self.results['bear_revenue'] = bear_revenue
        self.results['stock_prices'] = stock_prices
        self.results['volume'] = volume

        return revenue

    def create_revenue_dataframe(
        self,
        price_range: Optional[Tuple[float, float]] = None,
        step: float = 1.0
    ) -> pd.DataFrame:
        """
        创建收益数据框

        参数:
        ----------
        price_range : Tuple[float, float], optional
            价格范围
        step : float, optional
            价格步长，默认为1.0

        返回:
        -------
        df : pd.DataFrame
            收益数据框
        """
        if price_range is None:
            # 使用两个策略中较宽的价格范围
            bull_prices = self.bull_spread.results.get('stock_prices')
            bear_prices = self.bear_spread.results.get('stock_prices')

            if bull_prices is not None and bear_prices is not None:
                price_min = min(bull_prices.min(), bear_prices.min())
                price_max = max(bull_prices.max(), bear_prices.max())
            else:
                price_min = min(self.bull_spread.K_buy, self.bear_spread.K_sell) * 0.5
                price_max = max(self.bull_spread.K_sell, self.bear_spread.K_buy) * 1.5
        else:
            price_min, price_max = price_range

        stock_prices = np.arange(price_min, price_max, step)
        current_price = (self.bull_spread.K_buy + self.bear_spread.K_sell) / 2
        underlying_returns = (stock_prices - current_price) / current_price

        # 计算收益
        revenue = self.calculate_revenue(stock_prices)
        revenue_ratio = revenue / (current_price * 10000)

        # 创建数据框
        df = pd.DataFrame({
            '股价': stock_prices,
            '标的收益率': underlying_returns,
            '策略收益': revenue,
            '策略收益率': revenue_ratio
        })

        self.revenue_df = df
        return df

    def plot_payoff(self, ax=None, **plot_kwargs):
        """
        绘制损益图

        参数:
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib坐标轴对象
        **plot_kwargs : dict
            绘图参数

        返回:
        -------
        ax : matplotlib.axes.Axes
            绘图坐标轴
        """
        if self.revenue_df is None:
            self.create_revenue_dataframe()

        import matplotlib.pyplot as plt
        from ..utils.visualization import plot_strategy_payoff

        current_price = (self.bull_spread.K_buy + self.bear_spread.K_sell) / 2

        ax = plot_strategy_payoff(
            stock_prices=self.revenue_df['股价'].values,
            strategy_returns=self.revenue_df['策略收益率'].values,
            underlying_returns=self.revenue_df['标的收益率'].values,
            current_price=current_price,
            strategy_name='盒式价差',
            ax=ax,
            config=self.config,
            **plot_kwargs
        )

        return ax


class ButterflySpreadStrategy:
    """蝶式价差策略（call版本）"""

    def __init__(
        self,
        K_low: float,
        K_mid: float,
        K_high: float,
        c_low: float,
        c_mid: float,
        c_high: float,
        config: Optional[Any] = None
    ):
        """
        初始化蝶式价差策略

        参数:
        ----------
        K_low : float
            低位行权价
        K_mid : float
            中位行权价
        K_high : float
            高位行权价
        c_low : float
            低位看涨期权价格
        c_mid : float
            中位看涨期权价格
        c_high : float
            高位看涨期权价格
        config : Any, optional
            配置对象，默认为default_config
        """
        self.config = config or default_config

        # 验证参数
        self.K_low = validate_positive_number(K_low, 'K_low')
        self.K_mid = validate_positive_number(K_mid, 'K_mid')
        self.K_high = validate_positive_number(K_high, 'K_high')
        self.c_low = validate_positive_number(c_low, 'c_low')
        self.c_mid = validate_positive_number(c_mid, 'c_mid')
        self.c_high = validate_positive_number(c_high, 'c_high')

        if not (self.K_low < self.K_mid < self.K_high):
            raise ValueError("行权价必须满足: K_low < K_mid < K_high")

        # 初始化计算结果
        self.results = {}
        self.revenue_df = None

    def calculate_revenue(
        self,
        stock_prices: np.ndarray,
        volume: int = 10000
    ) -> np.ndarray:
        """
        计算策略收益

        参数:
        ----------
        stock_prices : np.ndarray
            股票价格序列
        volume : int, optional
            期权交易量（合约数），默认为10000

        返回:
        -------
        revenue : np.ndarray
            收益序列
        """
        # 计算低位看涨期权收益（买入）
        revenue_low = np.where(
            self.K_low <= stock_prices,
            (stock_prices - self.K_low) * volume - self.c_low * volume,
            -self.c_low * volume
        )

        # 计算中位看涨期权收益（卖出2份）
        revenue_mid = np.where(
            self.K_mid <= stock_prices,
            (self.K_mid - stock_prices) * 2 * volume + self.c_mid * 2 * volume,
            self.c_mid * 2 * volume
        )

        # 计算高位看涨期权收益（买入）
        revenue_high = np.where(
            self.K_high <= stock_prices,
            (stock_prices - self.K_high) * volume - self.c_high * volume,
            -self.c_high * volume
        )

        # 总收益
        revenue = revenue_low + revenue_mid + revenue_high

        # 保存结果
        self.results['revenue'] = revenue
        self.results['revenue_low'] = revenue_low
        self.results['revenue_mid'] = revenue_mid
        self.results['revenue_high'] = revenue_high
        self.results['stock_prices'] = stock_prices
        self.results['volume'] = volume

        return revenue

    def calculate_breakeven_points(self) -> Tuple[float, float]:
        """
        计算盈亏平衡点

        返回:
        -------
        lower_breakeven : float
            下盈亏平衡点
        upper_breakeven : float
            上盈亏平衡点
        """
        # 净期权费
        net_premium = 2 * self.c_mid - self.c_low - self.c_high

        # 盈亏平衡点
        lower_breakeven = self.K_low + net_premium
        upper_breakeven = self.K_high - net_premium

        return lower_breakeven, upper_breakeven

    def calculate_max_min_profit(self) -> Tuple[float, float]:
        """
        计算最大和最小收益

        返回:
        -------
        max_profit : float
            最大收益
        min_profit : float
            最小收益
        """
        # 净期权费
        net_premium = 2 * self.c_mid - self.c_low - self.c_high

        # 最大收益（股价等于中位行权价）
        max_profit = (self.K_mid - self.K_low) * 10000 + net_premium * 10000

        # 最小收益（股价低于低位或高于高位行权价）
        min_profit = net_premium * 10000

        return max_profit, min_profit

    def create_revenue_dataframe(
        self,
        price_range: Optional[Tuple[float, float]] = None,
        step: float = 1.0
    ) -> pd.DataFrame:
        """
        创建收益数据框

        参数:
        ----------
        price_range : Tuple[float, float], optional
            价格范围，默认为[K_low*0.5, K_high*1.5]
        step : float, optional
            价格步长，默认为1.0

        返回:
        -------
        df : pd.DataFrame
            收益数据框
        """
        if price_range is None:
            price_min = self.K_low * 0.5
            price_max = self.K_high * 1.5
        else:
            price_min, price_max = price_range

        stock_prices = np.arange(price_min, price_max, step)
        underlying_returns = (stock_prices - self.K_mid) / self.K_mid

        # 计算收益
        revenue = self.calculate_revenue(stock_prices)
        revenue_ratio = revenue / (self.K_mid * 10000)

        # 创建数据框
        df = pd.DataFrame({
            '股价': stock_prices,
            '标的收益率': underlying_returns,
            '策略收益': revenue,
            '策略收益率': revenue_ratio
        })

        self.revenue_df = df
        return df

    def plot_payoff(self, ax=None, **plot_kwargs):
        """
        绘制损益图

        参数:
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib坐标轴对象
        **plot_kwargs : dict
            绘图参数

        返回:
        -------
        ax : matplotlib.axes.Axes
            绘图坐标轴
        """
        if self.revenue_df is None:
            self.create_revenue_dataframe()

        import matplotlib.pyplot as plt
        from ..utils.visualization import plot_strategy_payoff

        ax = plot_strategy_payoff(
            stock_prices=self.revenue_df['股价'].values,
            strategy_returns=self.revenue_df['策略收益率'].values,
            underlying_returns=self.revenue_df['标的收益率'].values,
            current_price=self.K_mid,
            strategy_name='蝶式价差',
            ax=ax,
            config=self.config,
            **plot_kwargs
        )

        return ax


# 与原notebook兼容的函数
def calculate_bull_spread_revenue(
    stock_prices: np.ndarray,
    K_buy: float,
    K_sell: float,
    c_buy: float,
    c_sell: float,
    volume: int = 10000
) -> np.ndarray:
    """
    计算牛市价差收益（与原notebook兼容）

    参数:
    ----------
    stock_prices : np.ndarray
        股票价格序列
    K_buy : float
        买入看涨期权行权价
    K_sell : float
        卖出看涨期权行权价
    c_buy : float
        买入看涨期权价格
    c_sell : float
        卖出看涨期权价格
    volume : int, optional
        期权交易量，默认为10000

    返回:
    -------
    revenue : np.ndarray
        收益序列
    """
    strategy = BullSpreadStrategy(K_buy, K_sell, c_buy, c_sell)
    return strategy.calculate_revenue(stock_prices, volume)


def calculate_bear_spread_revenue(
    stock_prices: np.ndarray,
    K_buy: float,
    K_sell: float,
    p_buy: float,
    p_sell: float,
    volume: int = 10000
) -> np.ndarray:
    """
    计算熊市价差收益（与原notebook兼容）

    参数:
    ----------
    stock_prices : np.ndarray
        股票价格序列
    K_buy : float
        买入看跌期权行权价
    K_sell : float
        卖出看跌期权行权价
    p_buy : float
        买入看跌期权价格
    p_sell : float
        卖出看跌期权价格
    volume : int, optional
        期权交易量，默认为10000

    返回:
    -------
    revenue : np.ndarray
        收益序列
    """
    strategy = BearSpreadStrategy(K_buy, K_sell, p_buy, p_sell)
    return strategy.calculate_revenue(stock_prices, volume)