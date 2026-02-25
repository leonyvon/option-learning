# -*- coding: utf-8 -*-
"""
香草期权产品定价

提供香草期权产品的定价和收益计算，包括欧式看涨、价差等结构。
"""

import io
import sys
import numpy as np
import pandas as pd
from sympy import symbols, Eq, solve
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
from ..utils.validation import (
    validate_option_type, validate_positive_number, validate_percentage,
    validate_dataframe, validate_product_parameters
)


class VanillaProduct:
    """
    香草期权产品基类

    提供香草期权产品定价的通用功能，包括：
    - 欧式看涨保本结构
    - 欧式看涨价差结构
    """

    def __init__(
        self,
        data: pd.DataFrame,
        option_type: str,
        asset: float = None,
        participation_rate: float = None,
        capital_protection_rate: float = None,
        T: float = None,
        r: float = None,
        turnover_adjustment: bool = False,
        config: Optional[Any] = None
    ):
        """
        初始化香草产品

        参数:
        ----------
        data : pd.DataFrame
            包含股价、波动率等数据的DataFrame
        option_type : str
            期权类型: 'eu_call' 或 'eu_diff'
        asset : float, optional
            本金，默认为配置中的默认值
        participation_rate : float, optional
            参与率，默认为配置中的默认值
        capital_protection_rate : float, optional
            保本率，默认为配置中的默认值
        T : float, optional
            到期时间（年），默认为配置中的默认值
        r : float, optional
            年无风险利率，如果为None则从data中获取
        turnover_adjustment : bool, optional
            是否启用换手率调整，默认为False
        config : Any, optional
            配置对象，默认为default_config
        """
        self.config = config or default_config
        self.data = self._validate_data(data)
        self.option_type = self._validate_option_type(option_type)

        # 设置默认参数
        if asset is None:
            asset = self.config.strategy.default_asset
        if participation_rate is None:
            participation_rate = self.config.strategy.default_participation_rate
        if capital_protection_rate is None:
            capital_protection_rate = self.config.strategy.default_capital_protection_rate
        if T is None:
            T = 1.0  # 默认1年

        # 验证参数
        self.asset, self.participation_rate, self.capital_protection_rate, self.T = \
            validate_product_parameters(asset, participation_rate, capital_protection_rate, T)

        # 获取利率
        self.r = self._get_interest_rate(r)
        self.turnover_adjustment = turnover_adjustment

        # 计算当前价格和波动率
        self.S0 = self.data['close'].iloc[-1]
        self.sigma = self.data['sigma'].iloc[-1]

        # 初始化计算结果
        self.results = {}
        self.product_df = None
        self.revenue_df = None

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """验证数据"""
        required_columns = ['close', 'sigma']
        if self.config.turnover_adjustment.enabled:
            required_columns.append('turnoverRatio')

        return validate_dataframe(
            data, required_columns, allow_missing=False, index_type='datetime'
        )

    def _validate_option_type(self, option_type: str) -> str:
        """验证期权类型"""
        allowed_types = ['eu_call', 'eu_diff']
        if option_type not in allowed_types:
            raise ValueError(f"option_type必须是{allowed_types}之一，当前为: {option_type}")
        return option_type

    def _get_interest_rate(self, r: Optional[float]) -> float:
        """获取利率"""
        if r is not None:
            return r
        elif 'r' in self.data.columns:
            return self.data['r'].iloc[-1]
        else:
            raise ValueError("必须提供利率r，或确保data中包含'r'列")

    def _calculate_turnover_adjustment(self) -> float:
        """计算换手率调整系数"""
        if not self.turnover_adjustment:
            return 0.0

        if 'turnoverRatio' not in self.data.columns:
            warnings.warn("DataFrame中缺少'turnoverRatio'列，无法计算换手率调整", UserWarning)
            return 0.0

        tr_mean = self.data['turnoverRatio'].rolling(252).mean().iloc[-1]
        return self.config.turnover_adjustment.calculate_adjustment(tr_mean)

    def _calculate_option_price(
        self,
        option_type: str,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        adjustment: float = 0.0
    ) -> float:
        """计算期权价格"""
        from ..models.bsm import bsm_model

        price, _ = bsm_model(option_type, S0, K, T, r, sigma)
        return price * (1 + adjustment)

    def _create_price_simulation(self) -> pd.DataFrame:
        """创建价格模拟数据框"""
        price_min = self.S0 * self.config.strategy.price_simulation_range[0]
        price_max = self.S0 * self.config.strategy.price_simulation_range[1]
        step = self.config.strategy.price_simulation_step

        price_array = np.arange(round(price_min), round(price_max), step)
        revenue_bd = price_array / self.S0 - 1

        return pd.DataFrame({
            '股价': price_array,
            '标的': revenue_bd
        })

    def _calculate_eu_call_revenue(
        self,
        c: float,
        K: float,
        v: float,
        cost_call: float,
        cost_fixed: float,
        r_fixed: float,
        df: pd.DataFrame
    ) -> np.ndarray:
        """计算欧式看涨结构收益"""
        # 计算收益
        revenue = np.where(
            K <= df['股价'],
            (df['股价'] - K) * v + (cost_fixed * r_fixed * self.T - cost_call),
            cost_fixed * r_fixed * self.T - cost_call
        )
        return revenue

    def _calculate_eu_diff_revenue(
        self,
        c_buy: float,
        c_sell: float,
        K_buy: float,
        K_sell: float,
        v: float,
        cost_call: float,
        revenue_call: float,
        cost_fixed: float,
        r_fixed: float,
        df: pd.DataFrame
    ) -> np.ndarray:
        """计算欧式看涨价差结构收益"""
        # 计算各部分收益
        revenue_1 = np.where(
            K_buy <= df['股价'],
            (df['股价'] - K_buy) * v - cost_call,
            -cost_call
        )
        revenue_2 = np.where(
            K_sell <= df['股价'],
            revenue_call + (-df['股价'] + K_sell) * v,
            revenue_call
        )
        revenue_3 = cost_fixed * r_fixed * self.T

        return revenue_1 + revenue_2 + revenue_3

    def price_eu_call(
        self,
        strike_ratio: float = 1.0,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        欧式看涨保本结构定价

        参数:
        ----------
        strike_ratio : float, optional
            行权比价，默认为1.0（平值）
        verbose : bool, optional
            是否输出详细信息，默认为True

        返回:
        -------
        product_df : pd.DataFrame
            产品信息数据框
        """
        # 计算调整系数
        adj_params = self._calculate_turnover_adjustment()

        # 确定产品类型
        if self.participation_rate <= 1:
            product_type = '欧式保本型香草'
        else:
            product_type = '欧式杠杆型'

        # 计算行权价格和期权价格
        K = self.S0 * strike_ratio
        c = self._calculate_option_price(
            'call', self.S0, K, self.T, self.r, self.sigma, adj_params
        )

        # 计算固收产品年化收益率
        from .structured import cp_eu_call_1
        r_fixed = cp_eu_call_1(
            self.asset, c, K, self.participation_rate,
            self.capital_protection_rate, self.T
        )

        # 计算头寸规模
        v = self.asset / K * self.participation_rate
        cost_call = c * v
        cost_fixed = self.asset - cost_call
        C = cost_call / self.asset

        # 创建价格模拟
        df = self._create_price_simulation()

        # 计算收益
        revenue = self._calculate_eu_call_revenue(
            c, K, v, cost_call, cost_fixed, r_fixed, df
        )
        df['或有到期收益率'] = revenue / self.asset

        # 输出信息
        if verbose:
            print(f'已知有客户本金为{self.asset:.0f}元，可用期权数据为：c={c:.4f}，K={K:.2f}。'
                  f'参与度为{self.participation_rate:.2%},保本度为{self.capital_protection_rate:.2%}')
            print(f'购买期权成本为{cost_call:.2f}元')
            print(f'购买{cost_fixed:.2f}元年化收益率为{r_fixed:.2%}的固收产品')

        # 创建产品信息数据框
        way = '购买执行价为100%的看涨期权并购买固收产品'

        # 获取特定价格点的收益
        def get_revenue_at_price(price_multiplier: float) -> float:
            target_price = self.S0 * price_multiplier
            idx = (df['股价'] - target_price).abs().idxmin()
            return round(df.loc[idx]['或有到期收益率'], 4)

        product_df = pd.DataFrame({
            '类型': [product_type],
            '组合方法': [way],
            '期限': [f'{self.T:.1f}年'],
            '投资规模': [self.asset],
            '执行价': [self.S0],
            '报价': [c / self.S0],
            '购买固收比例': [1 - C],
            '购买期权比例': [C],
            '或有到期收益率(标的涨跌幅0%)': [get_revenue_at_price(1.0)],
            '或有到期收益率(标的涨跌幅10%)': [get_revenue_at_price(1.1)],
            '或有到期收益率(标的涨跌幅20%)': [get_revenue_at_price(1.2)],
            '收益范围': [f'{df["或有到期收益率"].min():.2%}~上不封顶'],
            '参与率': [self.participation_rate],
            '保本率': [self.capital_protection_rate],
            '必要固收年化': [round(r_fixed, 4)]
        })

        # 保存结果
        self.results['eu_call'] = {
            'strike_ratio': strike_ratio,
            'option_price': c,
            'strike_price': K,
            'fixed_income_yield': r_fixed,
            'option_cost': cost_call,
            'fixed_income_cost': cost_fixed,
            'option_ratio': C,
            'revenue_df': df
        }
        self.product_df = product_df
        self.revenue_df = df

        return product_df

    def price_eu_diff(
        self,
        strike_ratios: List[float] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        欧式看涨价差结构定价

        参数:
        ----------
        strike_ratios : List[float], optional
            行权比价列表，第一个为买入看涨期权的行权比价，
            第二个为卖出看涨期权的行权比价，默认为[1.0, 1.1]
        verbose : bool, optional
            是否输出详细信息，默认为True

        返回:
        -------
        product_df : pd.DataFrame
            产品信息数据框
        """
        if strike_ratios is None:
            strike_ratios = [1.0, 1.1]

        if len(strike_ratios) != 2:
            raise ValueError("strike_ratios必须包含两个值: [买入行权比价, 卖出行权比价]")

        # 计算调整系数
        adj_params = self._calculate_turnover_adjustment()

        # 计算行权价格和期权价格
        K_buy = self.S0 * strike_ratios[0]
        K_sell = self.S0 * strike_ratios[1]

        c_buy = self._calculate_option_price(
            'call', self.S0, K_buy, self.T, self.r, self.sigma, adj_params
        )
        c_sell = self._calculate_option_price(
            'call', self.S0, K_sell, self.T, self.r, self.sigma, adj_params
        )

        # 计算固收产品年化收益率
        from .structured import cp_eu_diff
        r_fixed = cp_eu_diff(
            self.asset, c_buy, c_sell, K_buy,
            self.participation_rate, self.capital_protection_rate, self.T
        )

        # 计算头寸规模
        v = self.asset / K_buy * self.participation_rate
        cost_call = c_buy * v
        revenue_call = c_sell * v
        C = (cost_call - revenue_call) / self.asset
        cost_fixed = self.asset - cost_call + revenue_call

        # 创建价格模拟
        df = self._create_price_simulation()

        # 计算收益
        revenue = self._calculate_eu_diff_revenue(
            c_buy, c_sell, K_buy, K_sell, v,
            cost_call, revenue_call, cost_fixed, r_fixed, df
        )
        df['或有到期收益率'] = revenue / self.asset

        # 输出信息
        if verbose:
            print('欧式看涨结构')
            print(f'已知有客户本金为1000万元，可用期权数据为：\n'
                  f'①：c={c_buy:.4f}，K={K_buy:.2f};②：c={c_sell:.4f},K={K_sell:.2f}。\n'
                  f'参与度为{self.participation_rate:.2%},保本度为{self.capital_protection_rate:.2%}')
            print(f'购买期权成本为{cost_call:.2f}元')
            print(f'卖出期权收入为{revenue_call:.2f}元')
            print(f'购买{cost_fixed:.2f}元年化收益率为{r_fixed:.2%}的固收产品')

        # 创建产品信息数据框
        way = '购买执行价为100%的看涨期权并购买固收产品，且卖出执行价为110%的看涨期权'

        # 获取特定价格点的收益
        def get_revenue_at_price(price_multiplier: float) -> float:
            target_price = self.S0 * price_multiplier
            idx = (df['股价'] - target_price).abs().idxmin()
            return round(df.loc[idx]['或有到期收益率'], 4)

        product_df = pd.DataFrame({
            '类型': ['欧式看涨价差'],
            '组合方法': [way],
            '投资规模': [self.asset],
            '期限': [f'{self.T:.1f}年'],
            '执行价': [f'{round(K_buy,2)},{round(K_sell,2)}'],
            '报价': [f'{(c_buy - c_sell) / self.S0:.4%}'],
            '购买固收比例': [1 - C],
            '净购买期权比例': [C],
            '或有到期收益率(标的涨跌幅0%)': [get_revenue_at_price(1.0)],
            '或有到期收益率(标的涨跌幅10%)': [get_revenue_at_price(1.1)],
            '或有到期收益率(标的涨跌幅20%)': [get_revenue_at_price(1.2)],
            '收益范围': [f'{df["或有到期收益率"].min():.2%}~{df["或有到期收益率"].max():.2%}'],
            '参与率': [self.participation_rate],
            '保本率': [self.capital_protection_rate],
            '必要固收年化': [round(r_fixed, 4)]
        })

        # 保存结果
        self.results['eu_diff'] = {
            'strike_ratios': strike_ratios,
            'buy_option_price': c_buy,
            'sell_option_price': c_sell,
            'buy_strike_price': K_buy,
            'sell_strike_price': K_sell,
            'fixed_income_yield': r_fixed,
            'option_cost': cost_call,
            'option_revenue': revenue_call,
            'net_option_ratio': C,
            'fixed_income_cost': cost_fixed,
            'revenue_df': df
        }
        self.product_df = product_df
        self.revenue_df = df

        return product_df

    def price(self, **kwargs) -> pd.DataFrame:
        """
        定价主函数

        参数:
        ----------
        **kwargs : dict
            传递给具体定价函数的参数

        返回:
        -------
        product_df : pd.DataFrame
            产品信息数据框
        """
        if self.option_type == 'eu_call':
            strike_ratio = kwargs.get('strike_ratio', 1.0)
            return self.price_eu_call(strike_ratio, **kwargs)
        elif self.option_type == 'eu_diff':
            strike_ratios = kwargs.get('strike_ratios', [1.0, 1.1])
            return self.price_eu_diff(strike_ratios, **kwargs)
        else:
            raise ValueError(f"不支持的期权类型: {self.option_type}")

    def plot_payoff(self, ax=None, **plot_kwargs):
        """
        绘制损益图

        参数:
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib坐标轴对象，如果为None则创建新的
        **plot_kwargs : dict
            绘图参数

        返回:
        -------
        ax : matplotlib.axes.Axes
            绘图坐标轴
        """
        if self.revenue_df is None:
            raise ValueError("请先调用price()方法计算收益")

        import matplotlib.pyplot as plt
        from ..utils.visualization import plot_payoff_diagram

        # 准备数据
        prices = self.revenue_df['股价'].values
        underlying_returns = self.revenue_df['标的'].values
        product_returns = self.revenue_df['或有到期收益率'].values

        # 绘制
        ax = plot_payoff_diagram(
            prices=prices,
            underlying_returns=underlying_returns,
            product_returns=product_returns,
            current_price=self.S0,
            ax=ax,
            config=self.config,
            **plot_kwargs
        )

        return ax


# 与原notebook兼容的函数
def vanillatype(
    data: pd.DataFrame,
    Type: str,
    asset: float = 10000000,
    KS: Union[float, List[float]] = 1.0,
    p_rate: float = 1.0,
    cp_rate: float = 1.0,
    T: float = 1.0,
    r: Optional[float] = None,
    tr_adj: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    香草产品定价函数（与原notebook兼容的接口）

    参数:
    ----------
    data : pd.DataFrame
        包含股价、波动率数据的df
    Type : str
        期权类型: 'eu_call' 或 'eu_diff'
    asset : float, optional
        本金，直接填1也可以，默认为10000000
    KS : Union[float, List[float]], optional
        执行价与现价的比例。如果是价差结构则传入一个列表，
        列表中第一个值是买购的执行价，第二个是卖购的执行价。默认为1.0
    p_rate : float, optional
        需要定制的参与度，默认为1.0
    cp_rate : float, optional
        需要定制的保本度，默认为1.0
    T : float, optional
        到期时间（年），默认为1.0
    r : float, optional
        年无风险利率，默认为None（从data中获取）
    tr_adj : bool, optional
        是否计算换手率调整系数，默认为False
    **kwargs : dict
        其他参数

    返回:
    -------
    product_df : pd.DataFrame
        产品信息数据框

    示例:
    --------
    >>> vanillatype(data, 'eu_call', asset=10000000, KS=1.0, p_rate=1.0, cp_rate=1.0, T=1, r=0.05)
    """
    # 创建产品实例
    product = VanillaProduct(
        data=data,
        option_type=Type,
        asset=asset,
        participation_rate=p_rate,
        capital_protection_rate=cp_rate,
        T=T,
        r=r,
        turnover_adjustment=tr_adj
    )

    # 调用定价方法
    if Type == 'eu_call':
        return product.price_eu_call(
            strike_ratio=KS if isinstance(KS, (int, float)) else KS[0],
            **kwargs
        )
    elif Type == 'eu_diff':
        if isinstance(KS, (int, float)):
            # 如果传入单个值，使用默认的价差结构
            strike_ratios = [KS, KS * 1.1]
        else:
            strike_ratios = KS
        return product.price_eu_diff(strike_ratios=strike_ratios, **kwargs)
    else:
        raise ValueError(f"不支持的期权类型: {Type}")