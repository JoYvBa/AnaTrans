#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:53:01 2024

@author: jorrit
"""
import numpy as np


class OneDim:
    def __init__(self, method, q, n, x, t, al, D_eff = 1e-9, R = 1, mu = 0):
        self.method = method
        self.q      = q
        self.n      = n
        self.v      = q/n
        self.x      = x
        self.t      = t
        self.D      = D_eff + al * self.v
        self.R      = R
        self.mu     = mu
    
    def inf_flow_pulse(self):         # Method 1
        result = np.zeros((len(self.x), len(self.t)))
        