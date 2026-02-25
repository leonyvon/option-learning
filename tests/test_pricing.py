# -*- coding: utf-8 -*-
"""
核心定价函数测试
"""

import io
import sys
import numpy as np
import pytest
from typing import Tuple

# 设置默认编码
if sys.version_info[0] == 3 and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (ValueError, AttributeError):
        pass

# 导入要测试的模块
from option_pricing.core.pricing import bsm_price, bsm_price_with_strike_ratio
from option_pricing.core.volatility import (
    calculate_implied_volatility,
    calculate_historical_volatility,
    find_nearest_value
)
from option_pricing.models.bsm import bsm_model, bsm_model_with_strike_ratio
from option_pricing.models.binomial import binary_tree_model


class TestBSMPricing:
    """BSM定价模型测试"""

    @pytest.fixture
    def bsm_params(self) -> dict:
        """BSM测试参数"""
        return {
            'S0': 100.0,
            'K': 100.0,
            'T': 1.0,
            'r': 0.05,
            'sigma': 0.2,
            'q': 0.0
        }

    def test_bsm_price_call(self, bsm_params):
        """测试看涨期权定价"""
        price = bsm_price('call', **bsm_params)
        assert isinstance(price, float)
        assert price > 0
        # 验证价格合理性（平值期权大约10-15）
        assert 8 < price < 20

    def test_bsm_price_put(self, bsm_params):
        """测试看跌期权定价"""
        price = bsm_price('put', **bsm_params)
        assert isinstance(price, float)
        assert price > 0
        # 平值看跌期权价格应低于看涨（无分红时）
        call_price = bsm_price('call', **bsm_params)
        assert price < call_price

    def test_bsm_price_with_greeks(self, bsm_params):
        """测试带希腊字母的定价"""
        price, greeks = bsm_price('call', **bsm_params, return_greeks=True)
        assert isinstance(price, float)
        assert isinstance(greeks, dict)
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'vega' in greeks
        assert 'theta' in greeks
        assert 'rho' in greeks

        # 验证希腊字母范围
        assert -1 <= greeks['delta'] <= 1
        assert greeks['gamma'] >= 0
        assert greeks['vega'] >= 0

    def test_bsm_price_vectorization(self):
        """测试向量化计算"""
        S0 = np.array([95, 100, 105])
        K = np.array([100, 100, 100])
        T = 1.0
        r = 0.05
        sigma = 0.2

        prices = bsm_price('call', S0, K, T, r, sigma)
        assert isinstance(prices, np.ndarray)
        assert prices.shape == (3,)
        # 价格应随S0增加而增加
        assert prices[0] < prices[1] < prices[2]

    def test_bsm_price_with_strike_ratio(self, bsm_params):
        """测试行权比价接口"""
        S0 = bsm_params['S0']
        strike_ratio = 1.0  # 平值
        price = bsm_price_with_strike_ratio(
            'call', S0, strike_ratio,
            bsm_params['T'], bsm_params['r'], bsm_params['sigma']
        )
        # 应与直接使用K=100得到相同结果
        direct_price = bsm_price('call', **bsm_params)
        assert abs(price - direct_price) < 1e-10

    def test_bsm_model_compatibility(self, bsm_params):
        """测试与原notebook的兼容性"""
        price, vega = bsm_model('call', **bsm_params)
        assert isinstance(price, (float, np.ndarray))
        assert isinstance(vega, float)
        # 价格应四舍五入到4位小数
        assert price == round(price, 4)

    def test_bsm_model_with_strike_ratio_compatibility(self):
        """测试行权比价兼容接口"""
        S0 = 100.0
        KS = 1.0
        T = 1.0
        r = 0.05
        sigma = 0.2

        price, vega = bsm_model_with_strike_ratio('call', S0, KS, T, r, sigma)
        assert price > 0
        assert vega > 0


class TestBinomialPricing:
    """二叉树定价模型测试"""

    @pytest.fixture
    def binomial_params(self) -> dict:
        """二叉树测试参数"""
        return {
            'S': 100.0,
            'K': 100.0,
            'T': 1.0,
            'r': 0.05,
            'sigma': 0.2,
            'N': 100
        }

    def test_binomial_model_european_call(self, binomial_params):
        """测试欧式看涨期权"""
        price = binary_tree_model(
            option_type='call',
            is_american=False,
            **binomial_params
        )
        assert isinstance(price, float)
        assert price > 0
        # 应与BSM价格接近
        bsm_price_val = bsm_price('call',
            S0=binomial_params['S'],
            K=binomial_params['K'],
            T=binomial_params['T'],
            r=binomial_params['r'],
            sigma=binomial_params['sigma']
        )
        # 允许一定误差
        assert abs(price - bsm_price_val) < 0.5

    def test_binomial_model_american_call(self, binomial_params):
        """测试美式看涨期权（无分红时应等于欧式）"""
        european_price = binary_tree_model(
            option_type='call',
            is_american=False,
            **binomial_params
        )
        american_price = binary_tree_model(
            option_type='call',
            is_american=True,
            **binomial_params
        )
        # 无分红时，美式看涨应等于欧式看涨
        assert abs(american_price - european_price) < 1e-10

    def test_binomial_model_american_put(self, binomial_params):
        """测试美式看跌期权（应高于欧式）"""
        european_price = binary_tree_model(
            option_type='put',
            is_american=False,
            **binomial_params
        )
        american_price = binary_tree_model(
            option_type='put',
            is_american=True,
            **binomial_params
        )
        # 美式看跌应不低于欧式看跌
        assert american_price >= european_price

    def test_binomial_model_vectorization(self):
        """测试向量化计算"""
        S = np.array([95, 100, 105])
        K = np.array([100, 100, 100])
        T = 1.0
        r = 0.05
        sigma = 0.2
        N = 100

        prices = binary_tree_model(S, K, T, r, sigma, N, option_type='call', is_american=False)
        assert isinstance(prices, np.ndarray)
        assert prices.shape == (3,)
        # 价格应随S增加而增加
        assert prices[0] < prices[1] < prices[2]


class TestVolatilityFunctions:
    """波动率函数测试"""

    def test_calculate_implied_volatility(self):
        """测试隐含波动率计算"""
        # 已知BSM价格，计算隐含波动率
        market_price = 10.4506  # 使用BSM计算的价格
        S0 = 100.0
        K = 100.0
        T = 1.0
        r = 0.05

        implied_vol = calculate_implied_volatility(
            market_price, 'call', S0, K, T, r,
            method='newton', verbose=False
        )

        assert isinstance(implied_vol, float)
        assert implied_vol > 0
        # 应接近原始波动率0.2
        assert abs(implied_vol - 0.2) < 0.01

    def test_calculate_historical_volatility(self):
        """测试历史波动率计算"""
        # 创建测试价格序列
        np.random.seed(42)
        n = 252  # 一年交易日
        returns = np.random.normal(0.0005, 0.02, n)
        prices = 100 * np.exp(np.cumsum(returns))

        vol = calculate_historical_volatility(
            prices, period=126, annualization_factor=252.0, method='simple'
        )

        assert isinstance(vol, float)
        assert vol > 0
        # 应接近输入的标准差年化值
        expected_vol = 0.02 * np.sqrt(252)
        assert abs(vol - expected_vol) < 0.05

    def test_find_nearest_value(self):
        """测试寻找最接近的值（修复bug）"""
        numbers = [95, 100, 105, 110, 115]

        # 测试正好在中间
        assert find_nearest_value(102.5, numbers) == 100
        assert find_nearest_value(107.5, numbers) == 110

        # 测试边界情况
        assert find_nearest_value(90, numbers) == 95  # 低于最小值
        assert find_nearest_value(120, numbers) == 115  # 高于最大值

        # 测试正好匹配
        assert find_nearest_value(100, numbers) == 100
        assert find_nearest_value(105, numbers) == 105

    def test_find_nearest_value_edge_cases(self):
        """测试边缘情况"""
        # 空数组
        with pytest.raises(ValueError):
            find_nearest_value(100, [])

        # 单个元素
        assert find_nearest_value(100, [95]) == 95
        assert find_nearest_value(100, [105]) == 105

        # 相等元素
        assert find_nearest_value(100, [95, 95, 100, 100]) == 100


class TestInputValidation:
    """输入验证测试"""

    def test_bsm_price_invalid_input(self):
        """测试BSM无效输入"""
        # 负的标的价格
        with pytest.raises(ValueError):
            bsm_price('call', S0=-100, K=100, T=1, r=0.05, sigma=0.2)

        # 负的波动率
        with pytest.raises(ValueError):
            bsm_price('call', S0=100, K=100, T=1, r=0.05, sigma=-0.2)

        # 无效的期权类型
        with pytest.raises(ValueError):
            bsm_price('invalid', S0=100, K=100, T=1, r=0.05, sigma=0.2)

        # 负的到期时间
        with pytest.raises(ValueError):
            bsm_price('call', S0=100, K=100, T=-1, r=0.05, sigma=0.2)

    def test_binomial_model_invalid_input(self):
        """测试二叉树无效输入"""
        # 负的步数
        with pytest.raises(ValueError):
            binary_tree_model(S=100, K=100, T=1, r=0.05, sigma=0.2, N=-100)

        # 无效的期权类型
        with pytest.raises(ValueError):
            binary_tree_model(S=100, K=100, T=1, r=0.05, sigma=0.2, N=100,
                             option_type='invalid', is_american=True)

    def test_implied_volatility_invalid_input(self):
        """测试隐含波动率无效输入"""
        # 负的市场价格
        with pytest.raises(ValueError):
            calculate_implied_volatility(-10, 'call', 100, 100, 1, 0.05)

        # 无效的计算方法
        with pytest.raises(ValueError):
            calculate_implied_volatility(10, 'call', 100, 100, 1, 0.05,
                                         method='invalid')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])