#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Jun 19 13:13:51 2024

@author: jorrit
"""

import numpy as np
import matplotlib.pyplot as plt
import OneDim as D1

# Time units in days
# Distance units in meters
# Weight units in grams

method = 5

x_min = 0
x_max = 200
x_nstep = 101
t_min = 0
t_max = 1500
t_nstep = 101

q = 0.05
n = 0.33
al = 0.1
m0 = 100
c0 = 10
c1 = 5
R = 1
mu = 0
D_eff = 1e-9 * 3600 * 24

plt_nx = 5
plt_nt = 5
fontsize = 12

x = np.linspace(x_min,x_max,x_nstep)
t = np.linspace(t_min,t_max,t_nstep)

oneD = D1.OneDim(method = method, q = q, n = n, x = x, t = t, al = al, m0 = m0, c0 = c0, c1 = c1, R = R, mu = mu, D_eff = D_eff)
results = oneD.transport()

plt.figure(dpi = 300)

for i in range(plt_nx):
    time_step = int(((t_nstep - 1) / plt_nx) * (i + 1))
    plt.plot(x, results[time_step, :], lw = 3, label = f"t = {np.round(t[time_step], 0)} days")

plt.xlabel("Distance [m]", fontsize = fontsize)
plt.ylabel("Concentration $[g m^{-3}]$", fontsize = fontsize)
plt.title("Concentration profile", fontsize = fontsize)
plt.legend()
plt.show()

plt.figure(dpi = 300)

for i in range(plt_nt):
    dist_step = int(((x_nstep - 1) / plt_nt) * (i + 1))
    plt.plot(t, results[:, dist_step], lw = 3, label = f"x = {np.round(x[dist_step], 0)} m")

plt.xlabel("Time [days]", fontsize = fontsize)
plt.ylabel("Concentration $[g m^{-3}]$", fontsize = fontsize)
plt.title("Breakthrough curve", fontsize = fontsize)
plt.legend()
plt.show()

#sum_array = np.zeros(len(t))
#for i in range(len(sum_array)):
#    sum_array[i] = np.sum(results[i,:])
#plt.plot(t, sum_array)