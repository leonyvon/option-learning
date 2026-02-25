"""
期权定价函数库

提供专业的期权定价、产品设计和策略分析工具。
主要模块：
- models: 定价模型（BSM、二叉树等）
- products: 期权产品定价
- strategies: 交易策略
- utils: 工具函数和可视化
- data: 数据获取和处理
"""

__version__ = "1.0.0"
__author__ = "Option Pricing Library"

from . import core
from . import models
from . import products
from . import strategies
from . import utils
from . import data