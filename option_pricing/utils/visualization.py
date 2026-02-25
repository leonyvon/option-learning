# -*- coding: utf-8 -*-
"""
可视化工具

提供统一的期权定价和策略可视化函数，提取重复的可视化代码。
"""

import io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from typing import Optional, Tuple, List, Dict, Any, Union
import warnings

# 设置默认编码
if sys.version_info[0] == 3 and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (ValueError, AttributeError):
        pass

# 导入配置
from .config import default_config


def set_chinese_font():
    """设置中文字体（如果需要）"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        warnings.warn("无法设置中文字体，图表可能显示乱码", UserWarning)


def ypercent(ax=None):
    """
    将y轴刻度设置为百分比形式

    参数:
    ----------
    ax : matplotlib.axes.Axes, optional
        坐标轴对象，如果为None则使用当前坐标轴
    """
    if ax is None:
        ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))


def xpercent(ax=None):
    """
    将x轴刻度设置为百分比形式

    参数:
    ----------
    ax : matplotlib.axes.Axes, optional
        坐标轴对象，如果为None则使用当前坐标轴
    """
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))


def create_price_range(
    current_price: float,
    range_multiplier: List[float] = None,
    step: float = None,
    config: Optional[Any] = None
) -> np.ndarray:
    """
    创建价格模拟范围

    参数:
    ----------
    current_price : float
        当前价格
    range_multiplier : List[float], optional
        价格范围乘数，默认为[0.5, 1.5]
    step : float, optional
        价格步长，如果为None则使用配置中的默认值
    config : Any, optional
        配置对象，默认为default_config

    返回:
    -------
    price_range : np.ndarray
        价格序列
    """
    config = config or default_config

    if range_multiplier is None:
        range_multiplier = config.strategy.price_simulation_range

    if step is None:
        step = config.strategy.price_simulation_step

    price_min = round(current_price * range_multiplier[0])
    price_max = round(current_price * range_multiplier[1])

    return np.arange(price_min, price_max, step)


def create_revenue_dataframe(
    current_price: float,
    config: Optional[Any] = None,
    **kwargs
) -> pd.DataFrame:
    """
    创建收益数据框

    参数:
    ----------
    current_price : float
        当前价格
    config : Any, optional
        配置对象，默认为default_config
    **kwargs : dict
        传递给create_price_range的参数

    返回:
    -------
    df : pd.DataFrame
        收益数据框，包含'股价'和'标的'列
    """
    config = config or default_config

    # 创建价格序列
    prices = create_price_range(current_price, config=config, **kwargs)

    # 计算标的收益率
    underlying_returns = prices / current_price - 1

    # 创建数据框
    df = pd.DataFrame({
        '股价': prices,
        '标的': underlying_returns
    })

    return df


def plot_payoff_diagram(
    prices: np.ndarray,
    underlying_returns: np.ndarray,
    product_returns: np.ndarray,
    current_price: float,
    ax=None,
    config: Optional[Any] = None,
    title: str = '产品损益图',
    product_label: str = '产品或有到期收益率',
    underlying_label: str = '标的',
    show_grid: bool = True,
    show_cross_axes: bool = True,
    y_limits: Optional[List[float]] = None
):
    """
    绘制产品损益图

    参数:
    ----------
    prices : np.ndarray
        价格序列
    underlying_returns : np.ndarray
        标的资产收益率序列
    product_returns : np.ndarray
        产品收益率序列
    current_price : float
        当前价格（用于十字坐标轴）
    ax : matplotlib.axes.Axes, optional
        坐标轴对象，如果为None则创建新的
    config : Any, optional
        配置对象，默认为default_config
    title : str, optional
        图表标题，默认为'产品损益图'
    product_label : str, optional
        产品曲线标签，默认为'产品或有到期收益率'
    underlying_label : str, optional
        标的曲线标签，默认为'标的'
    show_grid : bool, optional
        是否显示网格，默认为True
    show_cross_axes : bool, optional
        是否显示十字坐标轴，默认为True
    y_limits : List[float], optional
        Y轴限制，如果为None则使用配置中的默认值

    返回:
    -------
    ax : matplotlib.axes.Axes
        坐标轴对象
    """
    config = config or default_config
    set_chinese_font()

    # 创建图形和坐标轴
    if ax is None:
        fig, ax = plt.subplots(dpi=config.visualization.figure_dpi,
                              figsize=config.visualization.figure_size)
    else:
        fig = ax.figure

    # 绘制数据
    ax.plot(prices, product_returns,
            color=config.visualization.colors['product'],
            label=product_label)
    ax.plot(prices, underlying_returns,
            color=config.visualization.colors['underlying'],
            label=underlying_label)

    # 设置标签
    ax.set_xlabel('股价')
    ax.set_ylabel('损益', rotation=0)

    # 显示网格
    if show_grid:
        ax.grid(color=config.visualization.colors['grid'],
                linestyle=config.visualization.grid_linestyle,
                linewidth=config.visualization.grid_linewidth,
                alpha=config.visualization.grid_alpha,
                zorder=0)

    # 显示图例
    ax.legend()

    # 设置十字坐标轴
    if show_cross_axes and config.visualization.cross_axes_enabled:
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['left'].set_position(('data', current_price))
        ax.xaxis.set_label_coords(1.03, 0.5)
        ax.yaxis.set_label_coords(0.5, 1.03)
    else:
        # 添加零线
        ax.axhline(0, color=config.visualization.colors['zero_line'],
                   linestyle='--', linewidth=0.8, alpha=0.7)

    # 设置Y轴限制
    if y_limits is not None:
        ax.set_ylim(y_limits)
    elif config.visualization.yaxis_limits:
        ax.set_ylim(config.visualization.yaxis_limits)

    # 设置Y轴为百分比形式
    ypercent(ax)

    # 设置标题
    if title:
        ax.set_title(title)

    return ax


def plot_strategy_payoff(
    stock_prices: np.ndarray,
    strategy_returns: np.ndarray,
    underlying_returns: np.ndarray,
    current_price: float,
    strategy_name: str = '策略',
    ax=None,
    config: Optional[Any] = None,
    title: Optional[str] = None,
    show_zero_line: bool = True,
    show_grid: bool = True,
    y_limits: Optional[List[float]] = None
):
    """
    绘制策略损益图

    参数:
    ----------
    stock_prices : np.ndarray
        股票价格序列
    strategy_returns : np.ndarray
        策略收益率序列
    underlying_returns : np.ndarray
        标的资产收益率序列
    current_price : float
        当前价格
    strategy_name : str, optional
        策略名称，默认为'策略'
    ax : matplotlib.axes.Axes, optional
        坐标轴对象，如果为None则创建新的
    config : Any, optional
        配置对象，默认为default_config
    title : str, optional
        图表标题，如果为None则自动生成
    show_zero_line : bool, optional
        是否显示零线，默认为True
    show_grid : bool, optional
        是否显示网格，默认为True
    y_limits : List[float], optional
        Y轴限制

    返回:
    -------
    ax : matplotlib.axes.Axes
        坐标轴对象
    """
    config = config or default_config
    set_chinese_font()

    # 创建图形和坐标轴
    if ax is None:
        fig, ax = plt.subplots(dpi=config.visualization.figure_dpi,
                              figsize=config.visualization.figure_size)
    else:
        fig = ax.figure

    # 生成标题
    if title is None:
        title = f'{strategy_name}损益图'

    # 绘制数据
    ax.plot(stock_prices, underlying_returns,
            color=config.visualization.colors['underlying'],
            label='标的', linewidth=1, alpha=0.8)
    ax.plot(stock_prices, strategy_returns,
            color=config.visualization.colors['strategy'],
            label=f'{strategy_name}损益', linewidth=1.5)

    # 显示零线
    if show_zero_line:
        ax.axhline(0, color=config.visualization.colors['zero_line'],
                   linestyle='--', linewidth=0.8, alpha=0.7)

    # 设置标签
    ax.set_xlabel('股价')
    ax.set_ylabel('损益')

    # 显示网格
    if show_grid:
        ax.grid(color=config.visualization.colors['grid'],
                linestyle=config.visualization.grid_linestyle,
                linewidth=config.visualization.grid_linewidth,
                alpha=config.visualization.grid_alpha,
                zorder=0)

    # 设置十字坐标轴
    if config.visualization.cross_axes_enabled:
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['left'].set_position(('data', current_price))
        ax.xaxis.set_label_coords(1.03, 0.5)
        ax.yaxis.set_label_coords(0.5, 1.03)
    else:
        # 简单的坐标轴设置
        ax.axvline(current_price, color='gray', linestyle=':', alpha=0.5)

    # 设置Y轴限制
    if y_limits is not None:
        ax.set_ylim(y_limits)
    elif config.visualization.yaxis_limits:
        ax.set_ylim(config.visualization.yaxis_limits)

    # 设置Y轴为百分比形式
    ypercent(ax)

    # 显示图例
    ax.legend(loc='best')

    # 设置标题
    ax.set_title(title)

    return ax


def plot_strategy_comparison(
    stock_prices: np.ndarray,
    strategies: Dict[str, np.ndarray],
    underlying_returns: np.ndarray,
    current_price: float,
    ax=None,
    config: Optional[Any] = None,
    title: str = '策略对比',
    show_underlying: bool = True,
    show_grid: bool = True
):
    """
    绘制策略对比图

    参数:
    ----------
    stock_prices : np.ndarray
        股票价格序列
    strategies : Dict[str, np.ndarray]
        策略字典，键为策略名称，值为策略收益率序列
    underlying_returns : np.ndarray
        标的资产收益率序列
    current_price : float
        当前价格
    ax : matplotlib.axes.Axes, optional
        坐标轴对象
    config : Any, optional
        配置对象
    title : str, optional
        图表标题
    show_underlying : bool, optional
        是否显示标的资产
    show_grid : bool, optional
        是否显示网格

    返回:
    -------
    ax : matplotlib.axes.Axes
        坐标轴对象
    """
    config = config or default_config
    set_chinese_font()

    # 创建图形和坐标轴
    if ax is None:
        fig, ax = plt.subplots(dpi=config.visualization.figure_dpi,
                              figsize=config.visualization.figure_size)
    else:
        fig = ax.figure

    # 定义颜色循环
    colors = plt.cm.Set1(np.linspace(0, 1, len(strategies) + 1))

    # 绘制标的资产
    if show_underlying:
        ax.plot(stock_prices, underlying_returns,
                color=config.visualization.colors['underlying'],
                label='标的', linewidth=1.5, alpha=0.7)

    # 绘制各策略
    for idx, (strategy_name, strategy_returns) in enumerate(strategies.items()):
        color = colors[idx] if show_underlying else colors[idx + 1]
        ax.plot(stock_prices, strategy_returns,
                color=color, label=strategy_name, linewidth=1.5)

    # 设置标签
    ax.set_xlabel('股价')
    ax.set_ylabel('损益')

    # 显示网格
    if show_grid:
        ax.grid(color=config.visualization.colors['grid'],
                linestyle=config.visualization.grid_linestyle,
                linewidth=config.visualization.grid_linewidth,
                alpha=config.visualization.grid_alpha,
                zorder=0)

    # 显示零线
    ax.axhline(0, color=config.visualization.colors['zero_line'],
               linestyle='--', linewidth=0.8, alpha=0.7)

    # 显示当前价格线
    ax.axvline(current_price, color='gray', linestyle=':', alpha=0.5,
               label=f'当前价格: {current_price:.2f}')

    # 设置Y轴为百分比形式
    ypercent(ax)

    # 显示图例
    ax.legend(loc='best')

    # 设置标题
    ax.set_title(title)

    return ax


def plot_bull_bear_strategy_backtest(
    revenue_df: pd.DataFrame,
    underlying_revenue: pd.Series,
    ax=None,
    config: Optional[Any] = None,
    title: str = '牛熊价差套利策略'
):
    """
    绘制牛熊价差策略回测图

    参数:
    ----------
    revenue_df : pd.DataFrame
        收益数据框，包含'策略当月收益'和'策略累计收益'列
    underlying_revenue : pd.Series
        标的资产收益序列
    ax : matplotlib.axes.Axes, optional
        坐标轴对象
    config : Any, optional
        配置对象
    title : str, optional
        图表标题

    返回:
    -------
    ax : matplotlib.axes.Axes
        坐标轴对象
    """
    config = config or default_config
    set_chinese_font()

    # 创建图形和坐标轴
    if ax is None:
        fig, ax = plt.subplots(dpi=config.visualization.figure_dpi,
                              figsize=(10, 6))
    else:
        fig = ax.figure

    # 绘制策略收益
    if '策略当月收益' in revenue_df.columns and '策略累计收益' in revenue_df.columns:
        ax.plot(revenue_df.index, revenue_df['策略累计收益'],
                marker='o', label='策略累计收益', linewidth=1.5)
        # 可以同时显示当月收益
        # ax.bar(revenue_df.index, revenue_df['策略当月收益'],
        #        alpha=0.5, label='策略当月收益')

    # 绘制标的资产收益
    if underlying_revenue is not None:
        ax.plot(underlying_revenue.index, underlying_revenue.values,
                color='#f28c8c', label='10000份50etf', linewidth=1.5)

    # 设置标签
    ax.set_xlabel('日期')
    ax.set_ylabel('收益')

    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # 显示网格
    ax.grid(True, color='#c2c1c1', alpha=0.7)

    # 显示图例
    ax.legend(loc='best')

    # 设置标题
    ax.set_title(title)

    return ax


def save_plot(fig, filepath: str, dpi: int = 300, bbox_inches: str = 'tight'):
    """
    保存图表

    参数:
    ----------
    fig : matplotlib.figure.Figure
        图表对象
    filepath : str
        文件路径
    dpi : int, optional
        分辨率，默认为300
    bbox_inches : str, optional
        边界框，默认为'tight'
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    print(f"图表已保存至: {filepath}")