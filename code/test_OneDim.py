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
    '''
    Testing class for the OneDim class
    '''
    
    def setup_method(self):
        self.x_min = 0
        self.x_max = 200
        self.x_nstep = 51
        self.t_min = 0
        self.t_max = 1500
        self.t_nstep = 51
    
        self.q = 0.05
        self.n = 0.33
        self.al = 1
        self.m0 = 100
        self.c0 = 10
        self.c1 = 5
        self.R = 1
        self.mu = 0
        self.D_eff = 1e-9 * 3600 * 24
        self.h = 1
        
        self.x = np.linspace(self.x_min,self.x_max,self.x_nstep)
        self.t = np.linspace(self.t_min,self.t_max,self.t_nstep)

    def test_inf_flow_pulse(self):
        method = 6
        self.x_min = 0
        self.x_max = 2
        self.x_nstep = 101
        self.t_min = 0
        self.t_max = 10000
        self.t_nstep = 101
        
        self.x = np.linspace(self.x_min,self.x_max,self.x_nstep)
        self.t = np.linspace(self.t_min,self.t_max,self.t_nstep)
        
        results = D1.transport(method = method, x = self.x, t = self.t, D_eff = self.D_eff, c0 = self.c0)
        assert all(results[1:,0] == self.c0)
        print(results[50,25])
        assert results[50,25] == pytest.approx(5.90636)

    def test_method_ID(self):
        method = 33
        with pytest.raises(ValueError):
            D1.transport(method = method, q = self.q, n = self.n, x = self.x, t = self.t, al = self.al, D_eff = self.D_eff, m0 = self.m0, c0 = self.c0, c1 = self.c1, R = self.R, mu = self.mu, h = self.h)


