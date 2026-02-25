# 期权定价函数库

专业的期权定价、产品设计和策略分析工具库。

## 主要模块

### 核心定价模块 (`option_pricing/core/`)
- `pricing.py`: BSM定价模型和二叉树模型的核心实现
- `volatility.py`: 隐含波动率计算、历史波动率计算

### 定价模型 (`option_pricing/models/`)
- `bsm.py`: Black-Scholes-Merton模型（兼容原notebook接口）
- `binomial.py`: 二叉树定价模型

### 产品定价 (`option_pricing/products/`)
- `vanilla.py`: 香草期权产品定价（欧式看涨、价差结构）
- `structured.py`: 结构化产品收益计算函数

### 交易策略 (`option_pricing/strategies/`)
- `spreads.py`: 价差策略（牛市价差、熊市价差、盒式价差、蝶式价差）

### 工具函数 (`option_pricing/utils/`)
- `config.py`: 配置管理系统
- `validation.py`: 输入验证和错误处理
- `visualization.py`: 可视化工具
- `helpers.py`: 辅助函数（修复的find_closest_number等）

### 数据获取 (`option_pricing/data/`)
- `fetchers.py`: 数据获取接口（封装akshare）

## 安装

### 从源码安装

```bash
git clone https://github.com/yourusername/option-pricing.git
cd option-pricing
pip install -e .
```

### 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

```python
import numpy as np
import pandas as pd
from option_pricing.models.bsm import bsm_model
from option_pricing.core.volatility import calculate_implied_volatility

# BSM定价
call_price, vega = bsm_model('call', S0=100, K=100, T=1, r=0.05, sigma=0.2)
print(f"看涨期权价格: {call_price}")

# 隐含波动率计算
market_price = 10.45
implied_vol = calculate_implied_volatility(
    market_price, 'call', S0=100, K=100, T=1, r=0.05
)
print(f"隐含波动率: {implied_vol:.4f}")
```

更多示例请查看 `examples/basic_usage.py`。

## 配置系统

```python
from option_pricing.utils.config import default_config

# 查看默认配置
print(f"默认资产规模: {default_config.strategy.default_asset}")

# 动态更新配置
default_config.update_config(
    pricing.bsm_round_decimals=6,
    strategy.default_asset=5000000
)
```

`

## 目录结构

```
option_pricing/
├── __init__.py
├── core/                    # 核心定价函数
│   ├── __init__.py
│   ├── pricing.py
│   └── volatility.py
├── models/                  # 定价模型
│   ├── __init__.py
│   ├── bsm.py
│   └── binomial.py
├── products/                # 期权产品
│   ├── __init__.py
│   ├── vanilla.py
│   └── structured.py
├── strategies/              # 交易策略
│   ├── __init__.py
│   └── spreads.py
├── utils/                   # 工具函数
│   ├── __init__.py
│   ├── config.py
│   ├── validation.py
│   ├── visualization.py
│   └── helpers.py
└── data/                    # 数据获取
    ├── __init__.py
    └── fetchers.py
```

## 测试

运行测试：

```bash
pytest tests/
```

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License


## 联系方式

如有问题，请通过GitHub Issues联系我们。