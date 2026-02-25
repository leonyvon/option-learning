# -*- coding: utf-8 -*-
"""
产品定价函数测试
"""

import io
import sys
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

# 设置默认编码
if sys.version_info[0] == 3 and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (ValueError, AttributeError):
        pass

# 导入要测试的模块
from option_pricing.products.structured import (
    cp_eu_call_1,
    cp_eu_call_2,
    cp_eu_call_3,
    cp_eu_diff,
    cp_eu_call_ps,
    calculate_participation_from_cost_ratio,
    calculate_option_cost_from_participation,
    calculate_revenue_eu_call,
    calculate_revenue_eu_diff
)
from option_pricing.products.vanilla import VanillaProduct


class TestStructuredProductFunctions:
    """结构化产品函数测试"""

    @pytest.fixture
    def structured_product_params(self) -> dict:
        """结构化产品测试参数"""
        return {
            'asset': 10000000.0,  # 1000万本金
            'call_price': 10.0,   # 看涨期权价格
            'strike_price': 100.0,  # 行权价格
            'participation_rate': 0.55,  # 参与率
            'capital_protection_rate': 1.0,  # 保本率
            'T': 1.0  # 1年
        }

    def test_cp_eu_call_1(self, structured_product_params):
        """测试欧式看涨保本结构收益率计算"""
        r_fixed = cp_eu_call_1(**structured_product_params)

        assert isinstance(r_fixed, float)
        # 收益率应在合理范围内
        assert -0.1 < r_fixed < 0.5

        # 验证公式: (asset - call*v)*r*T - call*v = (cp_rate-1)*asset
        asset = structured_product_params['asset']
        call_price = structured_product_params['call_price']
        strike_price = structured_product_params['strike_price']
        participation_rate = structured_product_params['participation_rate']
        T = structured_product_params['T']

        v = asset / strike_price * participation_rate
        cost_call = call_price * v
        cost_fixed = asset - cost_call

        left_side = cost_fixed * r_fixed * T - cost_call
        right_side = (structured_product_params['capital_protection_rate'] - 1) * asset

        assert abs(left_side - right_side) < 1e-10

    def test_cp_eu_call_2(self):
        """测试根据固收收益率倒求期权数据"""
        r_fixed = 0.05
        asset = 10000000.0
        participation_rate = 0.55
        capital_protection_rate = 1.0
        T = 1.0

        call_k_ratio = cp_eu_call_2(r_fixed, asset, participation_rate, capital_protection_rate, T)

        assert isinstance(call_k_ratio, float)
        assert call_k_ratio > 0

        # 验证反向计算
        call_price = call_k_ratio * 100  # 假设行权价100
        r_calculated = cp_eu_call_1(asset, call_price, 100.0, participation_rate, capital_protection_rate, T)

        # 应接近原始r_fixed
        assert abs(r_calculated - r_fixed) < 1e-10

    def test_cp_eu_call_3(self):
        """测试根据期权费用比例求固收收益率"""
        asset = 10000000.0
        option_cost_ratio = 0.1  # 10%期权费用
        capital_protection_rate = 1.0
        T = 1.0

        r_fixed = cp_eu_call_3(asset, option_cost_ratio, capital_protection_rate, T)

        assert isinstance(r_fixed, float)
        # 验证计算
        cost_call = asset * option_cost_ratio
        cost_fixed = asset - cost_call

        left_side = cost_fixed * r_fixed * T - cost_call
        right_side = (capital_protection_rate - 1) * asset

        assert abs(left_side - right_side) < 1e-10

    def test_cp_eu_diff(self):
        """测试欧式看涨价差结构收益率计算"""
        asset = 10000000.0
        call_buy_price = 10.0
        call_sell_price = 5.0
        strike_buy_price = 100.0
        participation_rate = 1.0
        capital_protection_rate = 1.0
        T = 1.0

        r_fixed = cp_eu_diff(
            asset, call_buy_price, call_sell_price,
            strike_buy_price, participation_rate, capital_protection_rate, T
        )

        assert isinstance(r_fixed, float)

        # 验证公式
        v = asset / strike_buy_price * participation_rate
        cost_call = call_buy_price * v
        revenue_call = call_sell_price * v

        left_side = revenue_call + (asset - cost_call) * r_fixed * T - cost_call
        right_side = (capital_protection_rate - 1) * asset

        assert abs(left_side - right_side) < 1e-10

    def test_cp_eu_call_ps_alias(self):
        """测试cp_eu_call_ps别名函数"""
        # cp_eu_call_ps应该是cp_eu_diff的别名
        params = {
            'asset': 10000000.0,
            'call_buy_price': 10.0,
            'call_sell_price': 5.0,
            'strike_buy_price': 100.0,
            'participation_rate': 1.0,
            'capital_protection_rate': 1.0,
            'T': 1.0
        }

        r1 = cp_eu_diff(**params)
        r2 = cp_eu_call_ps(**params)

        assert r1 == r2

    def test_calculate_participation_from_cost_ratio(self):
        """测试根据期权费用比例计算参与率"""
        option_cost_ratio = 0.1
        call_price = 10.0
        strike_price = 100.0
        asset = 10000000.0

        participation_rate = calculate_participation_from_cost_ratio(
            option_cost_ratio, call_price, strike_price, asset
        )

        assert isinstance(participation_rate, float)
        assert participation_rate > 0

        # 验证反向计算
        cost_call = asset * option_cost_ratio
        v = cost_call / call_price
        calculated_rate = v * strike_price / asset

        assert abs(participation_rate - calculated_rate) < 1e-10

    def test_calculate_option_cost_from_participation(self):
        """测试根据参与率计算期权成本"""
        participation_rate = 0.55
        call_price = 10.0
        strike_price = 100.0
        asset = 10000000.0

        option_cost, option_cost_ratio = calculate_option_cost_from_participation(
            participation_rate, call_price, strike_price, asset
        )

        assert isinstance(option_cost, float)
        assert isinstance(option_cost_ratio, float)
        assert option_cost > 0
        assert 0 < option_cost_ratio < 1

        # 验证计算
        v = asset / strike_price * participation_rate
        expected_cost = call_price * v
        expected_ratio = expected_cost / asset

        assert abs(option_cost - expected_cost) < 1e-10
        assert abs(option_cost_ratio - expected_ratio) < 1e-10

    def test_calculate_revenue_eu_call(self):
        """测试欧式看涨结构收益计算"""
        asset = 10000000.0
        call_price = 10.0
        strike_price = 100.0
        participation_rate = 0.55
        r_fixed = 0.05
        T = 1.0

        # 创建测试股票价格序列
        stock_prices = np.array([80, 90, 100, 110, 120])

        revenue = calculate_revenue_eu_call(
            asset, call_price, strike_price,
            participation_rate, r_fixed, T, stock_prices
        )

        assert isinstance(revenue, np.ndarray)
        assert revenue.shape == stock_prices.shape

        # 验证计算结果
        v = asset / strike_price * participation_rate
        cost_call = call_price * v
        cost_fixed = asset - cost_call

        # 手动计算第一个价格点（低于行权价）
        expected_revenue_0 = cost_fixed * r_fixed * T - cost_call
        assert abs(revenue[0] - expected_revenue_0) < 1e-10

        # 手动计算最后一个价格点（高于行权价）
        expected_revenue_4 = (stock_prices[4] - strike_price) * v + (cost_fixed * r_fixed * T - cost_call)
        assert abs(revenue[4] - expected_revenue_4) < 1e-10

    def test_calculate_revenue_eu_diff(self):
        """测试欧式看涨价差结构收益计算"""
        asset = 10000000.0
        call_buy_price = 10.0
        call_sell_price = 5.0
        strike_buy_price = 100.0
        strike_sell_price = 110.0
        participation_rate = 1.0
        r_fixed = 0.05
        T = 1.0

        stock_prices = np.array([90, 100, 105, 110, 120])

        revenue = calculate_revenue_eu_diff(
            asset, call_buy_price, call_sell_price,
            strike_buy_price, strike_sell_price,
            participation_rate, r_fixed, T, stock_prices
        )

        assert isinstance(revenue, np.ndarray)
        assert revenue.shape == stock_prices.shape

        # 验证计算逻辑
        v = asset / strike_buy_price * participation_rate
        cost_call = call_buy_price * v
        revenue_call = call_sell_price * v
        cost_fixed = asset - cost_call + revenue_call

        # 手动计算一个价格点
        price = stock_prices[2]  # 105

        if price >= strike_buy_price:
            revenue_1 = (price - strike_buy_price) * v - cost_call
        else:
            revenue_1 = -cost_call

        if price >= strike_sell_price:
            revenue_2 = revenue_call + (-price + strike_sell_price) * v
        else:
            revenue_2 = revenue_call

        revenue_3 = cost_fixed * r_fixed * T
        expected_revenue = revenue_1 + revenue_2 + revenue_3

        assert abs(revenue[2] - expected_revenue) < 1e-10


class TestVanillaProduct:
    """香草产品定价测试"""

    @pytest.fixture
    def mock_data(self) -> pd.DataFrame:
        """创建模拟数据"""
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        n_days = len(dates)

        # 生成价格序列
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = 100 * np.exp(np.cumsum(returns))

        # 创建DataFrame
        data = pd.DataFrame({
            'close': prices,
            'changeRatio': returns,
            'sigma': 0.2,
            'turnoverRatio': np.random.uniform(0.5, 2.0, n_days),
            'r': 0.03  # 固定利率
        }, index=dates)

        # 计算滚动波动率
        data['sigma'] = data['changeRatio'].rolling(126).std() * np.sqrt(252)
        data = data.dropna()

        return data

    def test_vanilla_product_init(self, mock_data):
        """测试VanillaProduct初始化"""
        product = VanillaProduct(
            data=mock_data,
            option_type='eu_call',
            asset=10000000,
            participation_rate=0.8,
            capital_protection_rate=0.95,
            T=1.0,
            r=0.03,
            turnover_adjustment=False
        )

        assert product.S0 == mock_data['close'].iloc[-1]
        assert product.sigma == mock_data['sigma'].iloc[-1]
        assert product.r == 0.03
        assert product.asset == 10000000
        assert product.participation_rate == 0.8
        assert product.capital_protection_rate == 0.95
        assert product.T == 1.0

    def test_vanilla_product_eu_call_pricing(self, mock_data):
        """测试欧式看涨保本结构定价"""
        product = VanillaProduct(
            data=mock_data,
            option_type='eu_call',
            asset=10000000,
            participation_rate=0.8,
            capital_protection_rate=1.0,
            T=1.0,
            r=0.03,
            turnover_adjustment=False
        )

        product_df = product.price_eu_call(strike_ratio=1.0, verbose=False)

        assert isinstance(product_df, pd.DataFrame)
        assert len(product_df) == 1
        assert '类型' in product_df.columns
        assert '报价' in product_df.columns
        assert '购买固收比例' in product_df.columns
        assert '或有到期收益率(标的涨跌幅0%)' in product_df.columns
        assert '必要固收年化' in product_df.columns

        # 验证必要固收年化是数值
        assert isinstance(product_df['必要固收年化'].iloc[0], (int, float, np.number))

        # 验证有收益数据框
        assert product.revenue_df is not None
        assert '股价' in product.revenue_df.columns
        assert '标的' in product.revenue_df.columns
        assert '或有到期收益率' in product.revenue_df.columns

    def test_vanilla_product_eu_diff_pricing(self, mock_data):
        """测试欧式看涨价差结构定价"""
        product = VanillaProduct(
            data=mock_data,
            option_type='eu_diff',
            asset=10000000,
            participation_rate=1.0,
            capital_protection_rate=1.0,
            T=1.0,
            r=0.03,
            turnover_adjustment=False
        )

        product_df = product.price_eu_diff(strike_ratios=[1.0, 1.1], verbose=False)

        assert isinstance(product_df, pd.DataFrame)
        assert len(product_df) == 1
        assert '类型' in product_df.columns
        assert '执行价' in product_df.columns
        assert '报价' in product_df.columns
        assert '净购买期权比例' in product_df.columns

        # 验证有收益数据框
        assert product.revenue_df is not None
        assert '或有到期收益率' in product.revenue_df.columns

    def test_vanilla_product_turnover_adjustment(self, mock_data):
        """测试换手率调整"""
        product_with_adj = VanillaProduct(
            data=mock_data,
            option_type='eu_call',
            asset=10000000,
            participation_rate=0.8,
            capital_protection_rate=1.0,
            T=1.0,
            r=0.03,
            turnover_adjustment=True
        )

        product_without_adj = VanillaProduct(
            data=mock_data,
            option_type='eu_call',
            asset=10000000,
            participation_rate=0.8,
            capital_protection_rate=1.0,
            T=1.0,
            r=0.03,
            turnover_adjustment=False
        )

        df_with_adj = product_with_adj.price_eu_call(strike_ratio=1.0, verbose=False)
        df_without_adj = product_without_adj.price_eu_call(strike_ratio=1.0, verbose=False)

        # 价格可能不同（由于调整）
        assert df_with_adj['报价'].iloc[0] != df_without_adj['报价'].iloc[0] or \
               df_with_adj['必要固收年化'].iloc[0] != df_without_adj['必要固收年化'].iloc[0]

    def test_vanilla_product_invalid_option_type(self, mock_data):
        """测试无效期权类型"""
        with pytest.raises(ValueError):
            VanillaProduct(
                data=mock_data,
                option_type='invalid_type',
                asset=10000000,
                participation_rate=0.8,
                capital_protection_rate=1.0,
                T=1.0,
                r=0.03
            )

    def test_vanilla_product_missing_data_columns(self, mock_data):
        """测试缺失数据列"""
        # 删除必需列
        invalid_data = mock_data.drop(columns=['sigma'])

        with pytest.raises(ValueError):
            VanillaProduct(
                data=invalid_data,
                option_type='eu_call',
                asset=10000000,
                participation_rate=0.8,
                capital_protection_rate=1.0,
                T=1.0,
                r=0.03
            )

    def test_vanilla_product_price_method(self, mock_data):
        """测试通用price方法"""
        product = VanillaProduct(
            data=mock_data,
            option_type='eu_call',
            asset=10000000,
            participation_rate=0.8,
            capital_protection_rate=1.0,
            T=1.0,
            r=0.03
        )

        # 使用通用price方法
        product_df = product.price(strike_ratio=1.0, verbose=False)

        assert isinstance(product_df, pd.DataFrame)
        assert len(product_df) == 1

        # 使用eu_diff类型
        product_diff = VanillaProduct(
            data=mock_data,
            option_type='eu_diff',
            asset=10000000,
            participation_rate=1.0,
            capital_protection_rate=1.0,
            T=1.0,
            r=0.03
        )

        product_df_diff = product_diff.price(strike_ratios=[1.0, 1.1], verbose=False)
        assert isinstance(product_df_diff, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])