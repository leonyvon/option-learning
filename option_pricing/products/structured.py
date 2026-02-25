# -*- coding: utf-8 -*-
"""
结构化产品收益计算

提供各种保本型结构化产品的收益计算函数。
"""

import io
import sys
import numpy as np
import pandas as pd
from sympy import symbols, Eq, solve, Symbol
from typing import Union, Tuple, Optional, List
import warnings

# 设置默认编码
if sys.version_info[0] == 3 and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (ValueError, AttributeError):
        pass

# 导入验证工具
from ..utils.validation import validate_positive_number, validate_percentage


def cp_eu_call_1(
    asset: float,
    call_price: float,
    strike_price: float,
    participation_rate: float,
    capital_protection_rate: float,
    T: float
) -> float:
    """
    欧式看涨保本结构 - 已知期权数据和本金，求固收产品年化收益率

    参数:
    ----------
    asset : float
        本金
    call_price : float
        看涨期权每股价格
    strike_price : float
        行权价格
    participation_rate : float
        参与率
    capital_protection_rate : float
        保本率（如0.95表示95%保本）
    T : float
        期限（年）

    返回:
    -------
    r_fixed : float
        固收产品年化收益率

    示例:
    --------
    >>> cp_eu_call_1(asset=10000000, call_price=10, strike_price=100,
    ...              participation_rate=0.55, capital_protection_rate=1.0, T=1)
    0.0456

    公式:
    -----
    (asset - call_price * v) * r * T - call_price * v = (capital_protection_rate - 1) * asset
    其中 v = asset / strike_price * participation_rate
    """
    # 参数验证
    asset = validate_positive_number(asset, 'asset')
    call_price = validate_positive_number(call_price, 'call_price')
    strike_price = validate_positive_number(strike_price, 'strike_price')
    participation_rate = validate_percentage(
        participation_rate, 'participation_rate', min_value=0.0, max_value=5.0, allow_outside=True
    )
    capital_protection_rate = validate_percentage(
        capital_protection_rate, 'capital_protection_rate', min_value=0.0, max_value=1.5, allow_outside=True
    )
    T = validate_positive_number(T, 'T', allow_zero=True)

    # 计算头寸规模
    v = asset / strike_price * participation_rate

    # 建立方程并求解
    r = symbols('r')
    equation = Eq(
        (asset - call_price * v) * r * T - call_price * v,
        (capital_protection_rate - 1) * asset
    )

    solution = solve(equation, r)
    if not solution:
        raise ValueError("无法求解方程")

    return float(solution[0])


def cp_eu_call_2(
    r_fixed: float,
    asset: float,
    participation_rate: float,
    capital_protection_rate: float,
    T: float
) -> float:
    """
    欧式看涨保本结构 - 已知固收年化收益率和本金，倒求期权数据(每股权利金与执行价的比例)

    参数:
    ----------
    r_fixed : float
        固收产品年化收益率
    asset : float
        本金
    participation_rate : float
        参与率
    capital_protection_rate : float
        保本率
    T : float
        期限（年）

    返回:
    -------
    call_k_ratio : float
        期权价格与行权价的比例（call_price / strike_price）

    公式:
    -----
    (1 - C) * r * T - C = capital_protection_rate - 1
    其中 C = call_price * v / asset
        v = asset / strike_price * participation_rate
    推导得: call_price / strike_price = C / participation_rate
    """
    # 参数验证
    r_fixed = validate_percentage(r_fixed, 'r_fixed', min_value=-0.1, max_value=1.0, allow_outside=True)
    asset = validate_positive_number(asset, 'asset')
    participation_rate = validate_percentage(
        participation_rate, 'participation_rate', min_value=0.0, max_value=5.0, allow_outside=True
    )
    capital_protection_rate = validate_percentage(
        capital_protection_rate, 'capital_protection_rate', min_value=0.0, max_value=1.5, allow_outside=True
    )
    T = validate_positive_number(T, 'T', allow_zero=True)

    # 建立方程求解C（期权费用比例）
    C = symbols('C')
    equation_1 = Eq((1 - C) * r_fixed * T - C, capital_protection_rate - 1)
    solution_C = solve(equation_1, C)

    if not solution_C:
        raise ValueError("无法求解方程")

    C_value = float(solution_C[0])

    # 计算call_price / strike_price
    call_k_ratio = C_value / participation_rate

    return call_k_ratio


def cp_eu_call_3(
    asset: float,
    option_cost_ratio: float,
    capital_protection_rate: float,
    T: float
) -> float:
    """
    欧式看涨保本结构 - 已知期权费用比例，求固收产品年化收益率

    参数:
    ----------
    asset : float
        本金
    option_cost_ratio : float
        期权费用占本金的比例（如0.1表示10%）
    capital_protection_rate : float
        保本率
    T : float
        期限（年）

    返回:
    -------
    r_fixed : float
        固收产品年化收益率

    公式:
    -----
    cost_fixed * r * T - cost_call = (capital_protection_rate - 1) * asset
    其中 cost_call = asset * option_cost_ratio
        cost_fixed = asset * (1 - option_cost_ratio)
    """
    # 参数验证
    asset = validate_positive_number(asset, 'asset')
    option_cost_ratio = validate_percentage(
        option_cost_ratio, 'option_cost_ratio', min_value=0.0, max_value=1.0, allow_outside=True
    )
    capital_protection_rate = validate_percentage(
        capital_protection_rate, 'capital_protection_rate', min_value=0.0, max_value=1.5, allow_outside=True
    )
    T = validate_positive_number(T, 'T', allow_zero=True)

    # 计算成本
    cost_call = asset * option_cost_ratio
    cost_fixed = asset * (1 - option_cost_ratio)

    # 建立方程并求解
    r = symbols('r')
    equation = Eq(
        cost_fixed * r * T - cost_call,
        (capital_protection_rate - 1) * asset
    )

    solution = solve(equation, r)
    if not solution:
        raise ValueError("无法求解方程")

    return float(solution[0])


def cp_eu_diff(
    asset: float,
    call_buy_price: float,
    call_sell_price: float,
    strike_buy_price: float,
    participation_rate: float,
    capital_protection_rate: float,
    T: float
) -> float:
    """
    欧式看涨价差结构 - 已知期权数据和本金，求固收产品年化收益率

    参数:
    ----------
    asset : float
        本金
    call_buy_price : float
        买入看涨期权每股价格
    call_sell_price : float
        卖出看涨期权每股价格
    strike_buy_price : float
        买入看涨期权的行权价格
    participation_rate : float
        参与率
    capital_protection_rate : float
        保本率
    T : float
        期限（年）

    返回:
    -------
    r_fixed : float
        固收产品年化收益率

    公式:
    -----
    call_sell_price * v + (asset - call_buy_price * v) * r * T - call_buy_price * v
    = (capital_protection_rate - 1) * asset
    其中 v = asset / strike_buy_price * participation_rate
    """
    # 参数验证
    asset = validate_positive_number(asset, 'asset')
    call_buy_price = validate_positive_number(call_buy_price, 'call_buy_price')
    call_sell_price = validate_positive_number(call_sell_price, 'call_sell_price')
    strike_buy_price = validate_positive_number(strike_buy_price, 'strike_buy_price')
    participation_rate = validate_percentage(
        participation_rate, 'participation_rate', min_value=0.0, max_value=5.0, allow_outside=True
    )
    capital_protection_rate = validate_percentage(
        capital_protection_rate, 'capital_protection_rate', min_value=0.0, max_value=1.5, allow_outside=True
    )
    T = validate_positive_number(T, 'T', allow_zero=True)

    # 计算头寸规模
    v = asset / strike_buy_price * participation_rate

    # 建立方程并求解
    r = symbols('r')
    equation = Eq(
        call_sell_price * v + (asset - call_buy_price * v) * r * T - call_buy_price * v,
        (capital_protection_rate - 1) * asset
    )

    solution = solve(equation, r)
    if not solution:
        raise ValueError("无法求解方程")

    return float(solution[0])


def cp_eu_call_ps(
    asset: float,
    call_buy_price: float,
    call_sell_price: float,
    strike_buy_price: float,
    participation_rate: float,
    capital_protection_rate: float,
    T: float
) -> float:
    """
    欧式看涨价差结构（保本型） - 已知期权数据和本金，求固收产品年化收益率

    参数:
    ----------
    asset : float
        本金
    call_buy_price : float
        买入看涨期权每股价格
    call_sell_price : float
        卖出看涨期权每股价格
    strike_buy_price : float
        买入看涨期权的行权价格
    participation_rate : float
        参与率
    capital_protection_rate : float
        保本率
    T : float
        期限（年）

    返回:
    -------
    r_fixed : float
        固收产品年化收益率

    公式:
    -----
    call_sell_price * v + (asset - call_buy_price * v) * r * T - call_buy_price * v
    = (capital_protection_rate - 1) * asset
    其中 v = asset / strike_buy_price * participation_rate

    注意:
    -----
    此函数与cp_eu_diff公式相同，提供别名以保持与原notebook的兼容性
    """
    return cp_eu_diff(
        asset, call_buy_price, call_sell_price,
        strike_buy_price, participation_rate, capital_protection_rate, T
    )


def calculate_participation_from_cost_ratio(
    option_cost_ratio: float,
    call_price: float,
    strike_price: float,
    asset: float
) -> float:
    """
    根据期权费用比例计算参与率

    参数:
    ----------
    option_cost_ratio : float
        期权费用占本金的比例
    call_price : float
        看涨期权每股价格
    strike_price : float
        行权价格
    asset : float
        本金

    返回:
    -------
    participation_rate : float
        参与率
    """
    # 参数验证
    option_cost_ratio = validate_percentage(
        option_cost_ratio, 'option_cost_ratio', min_value=0.0, max_value=1.0, allow_outside=True
    )
    call_price = validate_positive_number(call_price, 'call_price')
    strike_price = validate_positive_number(strike_price, 'strike_price')
    asset = validate_positive_number(asset, 'asset')

    # 计算期权总成本
    cost_call = asset * option_cost_ratio

    # 计算头寸规模
    v = cost_call / call_price

    # 计算参与率
    participation_rate = v * strike_price / asset

    return participation_rate


def calculate_option_cost_from_participation(
    participation_rate: float,
    call_price: float,
    strike_price: float,
    asset: float
) -> Tuple[float, float]:
    """
    根据参与率计算期权成本和比例

    参数:
    ----------
    participation_rate : float
        参与率
    call_price : float
        看涨期权每股价格
    strike_price : float
        行权价格
    asset : float
        本金

    返回:
    -------
    option_cost : float
        期权成本
    option_cost_ratio : float
        期权成本占本金的比例
    """
    # 参数验证
    participation_rate = validate_percentage(
        participation_rate, 'participation_rate', min_value=0.0, max_value=5.0, allow_outside=True
    )
    call_price = validate_positive_number(call_price, 'call_price')
    strike_price = validate_positive_number(strike_price, 'strike_price')
    asset = validate_positive_number(asset, 'asset')

    # 计算头寸规模
    v = asset / strike_price * participation_rate

    # 计算期权成本
    option_cost = call_price * v
    option_cost_ratio = option_cost / asset

    return option_cost, option_cost_ratio


def calculate_revenue_eu_call(
    asset: float,
    call_price: float,
    strike_price: float,
    participation_rate: float,
    r_fixed: float,
    T: float,
    stock_prices: np.ndarray
) -> np.ndarray:
    """
    计算欧式看涨结构的收益序列

    参数:
    ----------
    asset : float
        本金
    call_price : float
        看涨期权每股价格
    strike_price : float
        行权价格
    participation_rate : float
        参与率
    r_fixed : float
        固收产品年化收益率
    T : float
        期限（年）
    stock_prices : np.ndarray
        股票价格序列

    返回:
    -------
    revenue : np.ndarray
        收益序列
    """
    # 参数验证
    asset = validate_positive_number(asset, 'asset')
    call_price = validate_positive_number(call_price, 'call_price')
    strike_price = validate_positive_number(strike_price, 'strike_price')
    participation_rate = validate_percentage(
        participation_rate, 'participation_rate', min_value=0.0, max_value=5.0, allow_outside=True
    )
    r_fixed = validate_percentage(r_fixed, 'r_fixed', min_value=-0.1, max_value=1.0, allow_outside=True)
    T = validate_positive_number(T, 'T', allow_zero=True)

    # 计算头寸规模和成本
    v = asset / strike_price * participation_rate
    cost_call = call_price * v
    cost_fixed = asset - cost_call

    # 计算收益
    revenue = np.where(
        strike_price <= stock_prices,
        (stock_prices - strike_price) * v + (cost_fixed * r_fixed * T - cost_call),
        cost_fixed * r_fixed * T - cost_call
    )

    return revenue


def calculate_revenue_eu_diff(
    asset: float,
    call_buy_price: float,
    call_sell_price: float,
    strike_buy_price: float,
    strike_sell_price: float,
    participation_rate: float,
    r_fixed: float,
    T: float,
    stock_prices: np.ndarray
) -> np.ndarray:
    """
    计算欧式看涨价差结构的收益序列

    参数:
    ----------
    asset : float
        本金
    call_buy_price : float
        买入看涨期权每股价格
    call_sell_price : float
        卖出看涨期权每股价格
    strike_buy_price : float
        买入看涨期权的行权价格
    strike_sell_price : float
        卖出看涨期权的行权价格
    participation_rate : float
        参与率
    r_fixed : float
        固收产品年化收益率
    T : float
        期限（年）
    stock_prices : np.ndarray
        股票价格序列

    返回:
    -------
    revenue : np.ndarray
        收益序列
    """
    # 参数验证
    asset = validate_positive_number(asset, 'asset')
    call_buy_price = validate_positive_number(call_buy_price, 'call_buy_price')
    call_sell_price = validate_positive_number(call_sell_price, 'call_sell_price')
    strike_buy_price = validate_positive_number(strike_buy_price, 'strike_buy_price')
    strike_sell_price = validate_positive_number(strike_sell_price, 'strike_sell_price')
    participation_rate = validate_percentage(
        participation_rate, 'participation_rate', min_value=0.0, max_value=5.0, allow_outside=True
    )
    r_fixed = validate_percentage(r_fixed, 'r_fixed', min_value=-0.1, max_value=1.0, allow_outside=True)
    T = validate_positive_number(T, 'T', allow_zero=True)

    # 计算头寸规模和成本
    v = asset / strike_buy_price * participation_rate
    cost_call = call_buy_price * v
    revenue_call = call_sell_price * v
    cost_fixed = asset - cost_call + revenue_call

    # 计算各部分收益
    revenue_1 = np.where(
        strike_buy_price <= stock_prices,
        (stock_prices - strike_buy_price) * v - cost_call,
        -cost_call
    )
    revenue_2 = np.where(
        strike_sell_price <= stock_prices,
        revenue_call + (-stock_prices + strike_sell_price) * v,
        revenue_call
    )
    revenue_3 = cost_fixed * r_fixed * T

    return revenue_1 + revenue_2 + revenue_3


class StructuredProductCalculator:
    """
    结构化产品计算器

    提供结构化产品收益计算的高级接口。
    """

    def __init__(self, asset: float, T: float = 1.0):
        """
        初始化计算器

        参数:
        ----------
        asset : float
            本金
        T : float, optional
            期限（年），默认为1.0
        """
        self.asset = validate_positive_number(asset, 'asset')
        self.T = validate_positive_number(T, 'T', allow_zero=True)
        self.results = {}

    def calculate_eu_call(
        self,
        call_price: float,
        strike_price: float,
        participation_rate: float,
        capital_protection_rate: float,
        stock_prices: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        计算欧式看涨保本结构

        参数:
        ----------
        call_price : float
            看涨期权每股价格
        strike_price : float
            行权价格
        participation_rate : float
            参与率
        capital_protection_rate : float
            保本率
        stock_prices : np.ndarray, optional
            股票价格序列，如果为None则自动生成

        返回:
        -------
        result : Dict[str, Any]
            计算结果字典
        """
        # 计算固收收益率
        r_fixed = cp_eu_call_1(
            self.asset, call_price, strike_price,
            participation_rate, capital_protection_rate, self.T
        )

        # 生成股票价格序列（如果未提供）
        if stock_prices is None:
            from ..utils.visualization import create_price_range
            stock_prices = create_price_range(
                strike_price,
                range_multiplier=[0.5, 1.5],
                step=1.0
            )

        # 计算收益
        revenue = calculate_revenue_eu_call(
            self.asset, call_price, strike_price,
            participation_rate, r_fixed, self.T, stock_prices
        )

        # 计算头寸规模和成本
        v = self.asset / strike_price * participation_rate
        cost_call = call_price * v
        cost_fixed = self.asset - cost_call
        option_ratio = cost_call / self.asset

        # 保存结果
        result = {
            'product_type': 'eu_call',
            'fixed_income_yield': r_fixed,
            'option_cost': cost_call,
            'fixed_income_cost': cost_fixed,
            'option_ratio': option_ratio,
            'position_size': v,
            'stock_prices': stock_prices,
            'revenue': revenue,
            'revenue_ratio': revenue / self.asset
        }
        self.results['eu_call'] = result

        return result

    def calculate_eu_diff(
        self,
        call_buy_price: float,
        call_sell_price: float,
        strike_buy_price: float,
        strike_sell_price: float,
        participation_rate: float,
        capital_protection_rate: float,
        stock_prices: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        计算欧式看涨价差结构

        参数:
        ----------
        call_buy_price : float
            买入看涨期权每股价格
        call_sell_price : float
            卖出看涨期权每股价格
        strike_buy_price : float
            买入看涨期权的行权价格
        strike_sell_price : float
            卖出看涨期权的行权价格
        participation_rate : float
            参与率
        capital_protection_rate : float
            保本率
        stock_prices : np.ndarray, optional
            股票价格序列，如果为None则自动生成

        返回:
        -------
        result : Dict[str, Any]
            计算结果字典
        """
        # 计算固收收益率
        r_fixed = cp_eu_diff(
            self.asset, call_buy_price, call_sell_price, strike_buy_price,
            participation_rate, capital_protection_rate, self.T
        )

        # 生成股票价格序列（如果未提供）
        if stock_prices is None:
            from ..utils.visualization import create_price_range
            stock_prices = create_price_range(
                strike_buy_price,
                range_multiplier=[0.5, 1.5],
                step=1.0
            )

        # 计算收益
        revenue = calculate_revenue_eu_diff(
            self.asset, call_buy_price, call_sell_price,
            strike_buy_price, strike_sell_price,
            participation_rate, r_fixed, self.T, stock_prices
        )

        # 计算头寸规模和成本
        v = self.asset / strike_buy_price * participation_rate
        cost_call = call_buy_price * v
        revenue_call = call_sell_price * v
        net_cost = cost_call - revenue_call
        cost_fixed = self.asset - net_cost
        net_option_ratio = net_cost / self.asset

        # 保存结果
        result = {
            'product_type': 'eu_diff',
            'fixed_income_yield': r_fixed,
            'buy_option_cost': cost_call,
            'sell_option_revenue': revenue_call,
            'net_option_cost': net_cost,
            'fixed_income_cost': cost_fixed,
            'net_option_ratio': net_option_ratio,
            'position_size': v,
            'stock_prices': stock_prices,
            'revenue': revenue,
            'revenue_ratio': revenue / self.asset
        }
        self.results['eu_diff'] = result

        return result