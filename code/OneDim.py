#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:53:01 2024

@author: jorrit
"""
import numpy as np
import scipy.special as sci


class OneDim:
    def __init__(self, method, q, n, x, t, al, m0 = 0, c0 = 0, D_eff = 0, R = 1, mu = 0):
        self.method = method
        self.q      = q
        self.n      = n
        self.v      = q/n
        self.x      = x
        self.t      = t[ : , np.newaxis]
        self.m0     = m0
        self.c0     = c0
        self.D      = D_eff + al * self.v
        self.R      = R
        self.mu     = mu
        
    # Method 1
    def inf_flow_pulse(self):         
        initial = self.m0 / (self.R * self.n * np.sqrt((4 * np.pi * self.D * self.t) / self.R))
        exponent = np.exp((-((self.x - (self.q * self.t / (self.R * self.n)))**2) / (4*np.pi*self.D* self.t / self.R)) - ((self.mu * self.t) / self.R))
        results = initial * exponent
        return(results)
        
    # Method 3
    def inf_flow_half(self):
        results = 0.5 * self.c0 * sci.erfc((self.x - (self.q * self.t / (self.R * self.n))) / (4*self.D * self.t / self.R)) * np.exp((self.mu * self.t) / self.R)
        return(results)
    
    def compute(self):
        if self.method == 1:
            return self.inf_flow_pulse()
        elif self.method == 3:
            return self.inf_flow_half()
        else:
            raise ValueError("Unknown method")