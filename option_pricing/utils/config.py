# -*- coding: utf-8 -*-
"""
配置管理系统

提取和管理硬编码参数，提供灵活的配置选项。
"""

import io
import sys
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any
import warnings

# 设置默认编码
if sys.version_info[0] == 3 and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (ValueError, AttributeError):
        pass


@dataclass
class TurnoverAdjustmentConfig:
    """换手率调整系数配置"""

    # 换手率调整系数相关参数（从原notebook中提取）
    mean_turnover_threshold: float = 1.0  # 换手率均值阈值
    adjustment_factor: float = 0.5  # 调整系数 (原: (1-tr_mean/1)*0.5)

    # 是否启用调整
    enabled: bool = True

    def calculate_adjustment(self, turnover_mean: float) -> float:
        """
        计算换手率调整系数

        参数:
        ----------
        turnover_mean : float
            换手率均值

        返回:
        -------
        adjustment : float
            调整系数
        """
        if not self.enabled or turnover_mean >= self.mean_turnover_threshold:
            return 0.0
        else:
            return (1 - turnover_mean / self.mean_turnover_threshold) * self.adjustment_factor


@dataclass
class PricingConfig:
    """定价相关配置"""

    # BSM定价相关
    bsm_round_decimals: int = 4  # BSM价格小数位数
    vega_round_decimals: int = 4  # Vega值小数位数

    # 二叉树定价相关
    default_tree_steps: int = 100  # 默认二叉树步数

    # 隐含波动率计算
    implied_vol_initial_guess: float = 0.5  # 初始猜测值
    implied_vol_max_iterations: int = 100  # 最大迭代次数
    implied_vol_tolerance: float = 1e-5  # 容差
    implied_vol_method: str = 'newton'  # 计算方法

    # 历史波动率计算
    historical_vol_window: int = 126  # 计算窗口（交易日）
    historical_vol_annualization: float = 252.0  # 年化因子
    historical_vol_method: str = 'simple'  # 计算方法


@dataclass
class StrategyConfig:
    """策略相关配置"""

    # 牛市/熊市价差策略参数
    bull_bear_spread_percentage: float = 0.05  # 5% 的价差比例
    option_volume: int = 10000  # 期权交易量

    # MA策略参数
    ma_period: int = 5  # 移动平均周期

    # 产品定价参数
    default_asset: float = 10000000  # 默认资产规模（1000万）
    default_participation_rate: float = 1.0  # 默认参与率
    default_capital_protection_rate: float = 1.0  # 默认保本率

    # 价格模拟范围
    price_simulation_range: List[float] = field(default_factory=lambda: [0.5, 1.5])  # 价格模拟范围比例
    price_simulation_step: float = 1.0  # 价格模拟步长（原为1，可根据需要调整）


@dataclass
class VisualizationConfig:
    """可视化配置"""

    # 图表样式
    figure_dpi: int = 100
    figure_size: tuple = (8, 6)

    # 颜色配置
    colors: Dict[str, str] = field(default_factory=lambda: {
        'underlying': '#002aff',  # 标的资产颜色
        'strategy': '#900000',    # 策略颜色
        'product': '#900000',     # 产品颜色
        'grid': 'k',              # 网格颜色
        'zero_line': 'grey',      # 零线颜色
        'target_return': 'red',   # 目标收益率颜色
    })

    # 网格样式
    grid_linestyle: str = '-.'
    grid_linewidth: float = 0.5
    grid_alpha: float = 0.5

    # 坐标轴设置
    yaxis_limits: List[float] = field(default_factory=lambda: [-0.5, 0.5])  # Y轴限制

    # 十字坐标轴设置
    cross_axes_enabled: bool = True  # 是否启用十字坐标轴


@dataclass
class DataConfig:
    """数据相关配置"""

    # 数据源配置
    default_start_date: str = '20200101'
    default_end_date: str = '20230714'

    # 缓存配置
    cache_enabled: bool = True
    cache_days: int = 7  # 缓存天数

    # 数据列名映射
    column_mapping: Dict[str, str] = field(default_factory=lambda: {
        'close': 'close',
        'changeRatio': 'changeRatio',
        'turnoverRatio': 'turnoverRatio',
        'sigma': 'sigma',
        'r': 'r',
        't': 't',
    })


class OptionPricingConfig:
    """
    期权定价配置管理器

    集中管理所有配置参数，提供默认值和自定义选项。
    """

    def __init__(
        self,
        turnover_adjustment: Optional[TurnoverAdjustmentConfig] = None,
        pricing: Optional[PricingConfig] = None,
        strategy: Optional[StrategyConfig] = None,
        visualization: Optional[VisualizationConfig] = None,
        data: Optional[DataConfig] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化配置

        参数:
        ----------
        turnover_adjustment : TurnoverAdjustmentConfig, optional
            换手率调整配置
        pricing : PricingConfig, optional
            定价配置
        strategy : StrategyConfig, optional
            策略配置
        visualization : VisualizationConfig, optional
            可视化配置
        data : DataConfig, optional
            数据配置
        custom_config : Dict[str, Any], optional
            自定义配置字典
        """
        self.turnover_adjustment = turnover_adjustment or TurnoverAdjustmentConfig()
        self.pricing = pricing or PricingConfig()
        self.strategy = strategy or StrategyConfig()
        self.visualization = visualization or VisualizationConfig()
        self.data = data or DataConfig()
        self.custom_config = custom_config or {}

        # 预计算一些常用值
        self._precomputed_values = {}
        self._precompute()

    def _precompute(self):
        """预计算一些常用值"""
        # 价格模拟步长转换为实际值（根据策略配置）
        self._precomputed_values['price_simulation_step_actual'] = (
            self.strategy.price_simulation_step
        )

    def update_config(self, **kwargs) -> None:
        """
        更新配置

        参数:
        ----------
        **kwargs : dict
            配置更新项，格式为: section_key.sub_key=value
            例如: pricing.bsm_round_decimals=6
        """
        for key, value in kwargs.items():
            if '.' in key:
                section, sub_key = key.split('.', 1)
                section_obj = getattr(self, section, None)
                if section_obj and hasattr(section_obj, sub_key):
                    setattr(section_obj, sub_key, value)
                else:
                    warnings.warn(f"配置项不存在: {key}")
            else:
                # 尝试更新自定义配置
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    self.custom_config[key] = value

        # 重新预计算
        self._precompute()

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        参数:
        ----------
        key : str
            配置键，格式为: section_key.sub_key
        default : Any, optional
            默认值

        返回:
        -------
        value : Any
            配置值
        """
        if '.' in key:
            section, sub_key = key.split('.', 1)
            section_obj = getattr(self, section, None)
            if section_obj and hasattr(section_obj, sub_key):
                return getattr(section_obj, sub_key)
        elif hasattr(self, key):
            return getattr(self, key)
        elif key in self.custom_config:
            return self.custom_config[key]

        return default

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'turnover_adjustment': asdict(self.turnover_adjustment),
            'pricing': asdict(self.pricing),
            'strategy': asdict(self.strategy),
            'visualization': asdict(self.visualization),
            'data': asdict(self.data),
            'custom_config': self.custom_config,
        }

    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """从字典加载配置"""
        if 'turnover_adjustment' in config_dict:
            self.turnover_adjustment = TurnoverAdjustmentConfig(**config_dict['turnover_adjustment'])
        if 'pricing' in config_dict:
            self.pricing = PricingConfig(**config_dict['pricing'])
        if 'strategy' in config_dict:
            self.strategy = StrategyConfig(**config_dict['strategy'])
        if 'visualization' in config_dict:
            self.visualization = VisualizationConfig(**config_dict['visualization'])
        if 'data' in config_dict:
            self.data = DataConfig(**config_dict['data'])
        if 'custom_config' in config_dict:
            self.custom_config = config_dict['custom_config']

        self._precompute()

    def save_to_file(self, filepath: str) -> None:
        """保存配置到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def load_from_file(self, filepath: str) -> None:
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        self.from_dict(config_dict)


# 默认配置实例
default_config = OptionPricingConfig()