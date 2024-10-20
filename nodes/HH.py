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
    "g_Na" : 120.,   # 钠离子通道的最大电导(mS/cm2)
    "g_K"  : 36.,    # 钾离子通道的最大电导(mS/cm2)
    "g_L"  : 0.3,   # 漏离子电导(mS/cm2)
    "E_Na" : 50.,    # 钠离子的平衡电位(mV)
    "E_K"  : -77.,   # 钾离子的平衡电位(mV)
    "E_L"  : -54.4, # 漏离子的平衡电位(mV)
    "Cm"   : 1.,   # 比膜电容(uF/cm2)
    "temperature" : 6.3, # 温度
    "Iex"  : 10.     # 恒定的外部激励
}
temperature = params_nodes["temperature"]
params_nodes["phi"] = 3.0 ** ((temperature - 6.3) / 10)

# 运行变量
vars_run = {
    "t" : 0     # 运行时间
}

# 节点状态变量
N = params_nodes["N"]
mem = np.random.uniform(-.3, .3, N)
m = 0.5 * np.random.rand(N)
n = 1 * np.random.rand(N)
h = 0.6 * np.random.rand(N)
zhuang_vars_HH = np.array([mem, m, n, h])

# 放电参数
N = params_nodes["N"]
params_spikes_eval = {
    "th_up"         : 0,                         # 放电阈值
    "th_down"       : -10,                       # 放电阈下值
    "flag"          : np.zeros(N, dtype=int),    # 模型放电标志(>0, 放电)
    "flaglaunch"    : np.zeros(N, dtype=int),    # 模型开始放电标志(==1, 放电刚刚开始)
    "firingTime"    : np.zeros(N)                # 记录放电时间(上次放电)
}

@njit
def model_nodes(vars, I, params, vars_run, *args):
    """
    :param vars:        状态变量 [N_vars, N_nodes]
    :param I:           外部的刺激电流 [N_vars, N_nodes]
    :param params:      节点参数
    :param vars_run:    运行中的变量
    :param args:        其他量
    :return:            每个状态变量的速度
    """
    # 变量
    mem, m, n, h = vars[0], vars[1], vars[2], vars[3]
    # 参数
    g_Na, g_K = params["g_Na"], params["g_K"]
    dmem_dt = (-g_Na * np.power(m, 3) * h * (mem - self._E_Na) \
               - self._g_K * np.power(n, 4) * (mem - self._E_K) \
               - self._g_L * (mem - self._E_L) + I[0]) / self._Cm
    dm_dt = 0.1 * (mem + 40.0) / (1.0 - np.exp(-(mem + 40) / 10.0)) * (1.0 - m) \
            - 4.0 * np.exp(-(mem + 65.0) / 18.0) * m + I[1]
    dn_dt = 0.01 * (mem + 55.0) / (1 - np.exp(-(mem + 55) / 10)) * (1 - n) \
            - 0.125 * np.exp(-(mem + 65.0) / 80) * n + I[2]
    dh_dt = 0.07 * np.exp(-(mem + 65.0) / 20.0) * (1.0 - h) \
            - 1.0 / (1.0 + np.exp(-(mem + 35.0) / 10.0)) * h + I[3]
    if self.temperature is not None:
        dm_dt *= self.phi
        dn_dt *= self.phi
        dh_dt *= self.phi

    return np.array([dmem_dt, dm_dt, dn_dt, dh_dt])

@njit
def HH():
    pass

if __name__ == "__main__":
    pass