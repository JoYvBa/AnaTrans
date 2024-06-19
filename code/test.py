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

method = 3
x = np.linspace(0,100,1001)
t = np.linspace(0,400,1001)
q = 0.05
n = 0.33
al = 0.1
m0 = 100
c0 = 10
R = 1
mu = 0
D_eff = 1e-9 * 3600 * 24

transport = D1.OneDim(method = method, q = q, n = n, x = x, t = t, al = al, m0 = m0, c0 = c0, R = R, mu = mu, D_eff = D_eff)

results = transport.compute()
plt.plot(x, results[100,:], label = f"t = {np.round(t[100], 0)} days")
plt.plot(x, results[300,:], label = f"t = {np.round(t[300], 0)} days")
plt.plot(x, results[500,:], label = f"t = {np.round(t[500], 0)} days")
plt.plot(x, results[700,:], label = f"t = {np.round(t[700], 0)} days")
plt.xlabel("Distance [m]")
plt.ylabel("Concentration [g/m3]")
plt.legend()
plt.show()

plt.plot(t, results[:,100], label = f"x = {np.round(x[100], 0)} m")
plt.plot(t, results[:,300], label = f"x = {np.round(x[300], 0)} m")
plt.plot(t, results[:,500], label = f"x = {np.round(x[500], 0)} m")
plt.plot(t, results[:,700], label = f"x = {np.round(x[700], 0)} m")
plt.xlabel("Time [d]")
plt.ylabel("Concentration [g/m3]")
plt.legend()
plt.show()

sum_array = np.zeros(len(t))
for i in range(len(sum_array)):
    sum_array[i] = np.sum(results[i,:])
plt.plot(t, sum_array)