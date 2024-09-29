# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/9/29
# User      : WuY
# File      : base_mods.py
# 文件中包含：
# 1、节点动力学基础模块
# 2、数值模拟算法
# 3、突触动力学基础模块

import os
import sys
import copy
import numpy as np
from numba import njit, prange


# ================================= 神经元模型的基类 =================================


