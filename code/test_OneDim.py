#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:41:10 2024

@author: jorrit
"""

import OneDim as D1
import numpy as np
import pytest

class TestClassOneDim():
    
    def setup_method(self):
        x_min = 0
        x_max = 200
        x_nstep = 101
        t_min = 0
        t_max = 1500
        t_nstep = 101
    
        self.q = 0.05
        self.n = 0.33
        self.al = 1
        self.m0 = 100
        self.c0 = 10
        self.c1 = 5
        self.R = 1
        self.mu = 0
        self.D_eff = 1e-4
        self.h = 1
        
        self.x = np.linspace(x_min,x_max,x_nstep)
        self.t = np.linspace(t_min,t_max,t_nstep)
    '''
    def test_inf_flow_pulse(self):
        method = 1
        oneD = D1.OneDim(method = method, q = self.q, n = self.n, x = self.x, t = self.t, al = self.al, D_eff = self.D_eff, m0 = self.m0, c0 = self.c0, c1 = self.c1, R = self.R, mu = self.mu, h = self.h)
        results = oneD.transport()
    '''
    def test_method_ID(self):
        method = 33
        oneD = D1.OneDim(method = method, q = self.q, n = self.n, x = self.x, t = self.t, al = self.al, D_eff = self.D_eff, m0 = self.m0, c0 = self.c0, c1 = self.c1, R = self.R, mu = self.mu, h = self.h)
        with pytest.raises(ValueError):
            oneD.transport()

