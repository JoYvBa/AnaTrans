#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:41:10 2024

@author: jorrit
"""

import ADE_eq as ade
import numpy as np
import pytest


class TestClassOneDim:
    """Testing class for the cxt_1D class."""

    def setup_method(self):
        self.x_min = 0
        self.x_max = 200
        self.x_nstep = 51
        self.t_min = 0
        self.t_max = 1500
        self.t_nstep = 51

        self.x = np.linspace(self.x_min, self.x_max, self.x_nstep)
        self.t = np.linspace(self.t_min, self.t_max, self.t_nstep)

    def test_inf_flow_pulse(self) -> None:

        results = D1.transport(method = method, x = self.x, t = self.t, D_eff = self.D_eff, c0 = self.c0)
        assert all(results[1:,0] == self.c0)
        print(results[50,25])
        assert results[50,25] == pytest.approx(5.90636)

    def test_value_type(self):
        adv = ade.cxt_1D(self.x, self.t)
        v = 'This should not work'
        with pytest.raises(ValueError):
            adv.ade(v)