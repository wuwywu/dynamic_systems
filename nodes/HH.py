# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/10/20
# User      : WuY
# File      : HH.py
# Hodgkin-Huxley(HH) 模型

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# seed = 0
# np.random.seed(seed)                # 给numpy设置随机种子

# 网络参数
params_net = {
    "N"    : 10     # 节点个数
}

# 节点参数
params_nodes = {
    "g_Na" : 120,   # 钠离子通道的最大电导(mS/cm2)
    "g_K"  : 36,    # 钾离子通道的最大电导(mS/cm2)
    "g_L"  : 0.3,   # 漏离子电导(mS/cm2)
    "E_Na" : 50,    # 钠离子的平衡电位(mV)
    "E_K"  : -77,   # 钾离子的平衡电位(mV)
    "E_L"  : -54.4, # 漏离子的平衡电位(mV)
    "Cm"   : 1.0,   # 比膜电容(uF/cm2)
    "temperature" : 6.3, # 温度
}
temperature = params_nodes["temperature"]
params_nodes["phi"] = 3.0 ** ((temperature - 6.3) / 10)

# 放点参数
N = params_nodes["N"]
params_spikes_eval = {
    "th_up"         : 0,                         # 放电阈值
    "th_down"       : -10,                       # 放电阈下值
    "flag"          : np.zeros(N, dtype=int),    # 模型放电标志(>0, 放电)
    "flaglaunch"    : np.zeros(N, dtype=int),    # 模型开始放电标志(==1, 放电刚刚开始)
    "firingTime"    : np.zeros(N)                # 记录放电时间(上次放电)
}

# 运行变量
vars_run = {
    "t" : 0     # 运行时间
}

# 状态变量
N = params_nodes["N"]
mem = np.random.uniform(-.3, .3, N)
m = 0.5 * np.random.rand(N)
n = 1 * np.random.rand(N)
h = 0.6 * np.random.rand(N)
zhuang_vars_HH = np.array([mem, m, n, h])