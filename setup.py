#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
期权定价函数库安装配置
"""

import io
import os
import sys
from setuptools import setup, find_packages

# 读取README
with io.open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# 读取版本信息
about = {}
with io.open(os.path.join('option_pricing', '__init__.py'), 'r', encoding='utf-8') as f:
    exec(f.read(), about)

setup(
    name='option-pricing',
    version=about['__version__'],
    description='专业的期权定价、产品设计和策略分析工具库',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=about['__author__'],
    author_email='',
    url='https://github.com/yourusername/option-pricing',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'sympy>=1.9.0',
        'matplotlib>=3.4.0',
        'akshare>=1.10.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.900.0',
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0',
            'seaborn>=0.11.0',
        ],
        'full': [
            'seaborn>=0.11.0',
            'talib>=0.4.0',  # 注意：可能需要通过conda安装
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Office/Business :: Financial',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    python_requires='>=3.8',
    keywords='option pricing, derivatives, financial engineering, quant finance',
    project_urls={
        'Documentation': 'https://github.com/yourusername/option-pricing',
        'Source': 'https://github.com/yourusername/option-pricing',
        'Tracker': 'https://github.com/yourusername/option-pricing/issues',
    },
)