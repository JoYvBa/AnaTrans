#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:53:01 2024

@author: jorrit
"""
import numpy as np
import scipy.special as sci


class OneDim:
    def __init__(self, method, q, n, x, t, al, m0 = 0, c0 = 0, D_eff = 0, R = 1, mu = 0, c1 = 0, rel_conc = True):
        self.method = method
        self.q        = q
        self.n        = n
        self.v        = q/n
        self.x        = x
        self.t        = t[ : , np.newaxis]
        self.m0       = m0
        self.c0       = c0
        self.D_eff    = D_eff
        self.D        = D_eff + al * self.v
        self.R        = R
        self.mu       = mu
        self.c1       = c1
        self.rel_conc = rel_conc
    
    # Replace q/n in methods by v
        
    # Method 1; Infinite column with steady flow in the x-direction and with pulse injection at t = 0
    def inf_flow_pulse(self):         
        initial = self.m0 / (self.R * self.n * np.sqrt((4 * np.pi * self.D * self.t) / self.R))
        exponent = np.exp((-((self.x - (self.q * self.t / (self.R * self.n)))**2) / (4*np.pi*self.D* self.t / self.R)) - ((self.mu * self.t) / self.R))
        results = initial * exponent
        return(results)
    
    # Method 2; Infinite column with no flow, and with pulse injection at t = 0
    def inf_noflow_pulse(self):
        initial = self.m0 / (self.R * self.n * np.sqrt((4 * np.pi * self.D_eff * self.t) / self.R))
        exponent = np.exp((-(self.x**2) / (4*np.pi*self.D_eff* self.t / self.R)) - ((self.mu * self.t) / self.R))
        results = initial * exponent
        return(results)
    
    # Method 3; Infinite column with steady flow, the solute is present with c0 at the left half of the infinite domain at t = 0
    def inf_flow_half(self):
        results = 0.5 * self.c0 * sci.erfc((self.x - (self.q * self.t / (self.R * self.n))) / (4*self.D * self.t / self.R)) * np.exp((self.mu * self.t) / self.R)
        return(results)
    
    # Method 4: Infinite column with steady flow, the solute is present with c0 at the left half of the infinite domain and with c1 at the right half at t = 0
    def inf_flow_halfinit(self):
        results = (self.c1 + 0.5 * (self.c0 - self.c1) * sci.erfc((self.x - (self.q * self.t / (self.R * self.n))) / (4*self.D * self.t / self.R)) * np.exp((self.mu * self.t)) / self.R)
        return(results)        
    
    # Method 5: Semi-infinite column with steady flow and constant concentration at x = 0
    def semi_flow_const(self):
        # Gives problems due to large values in the exponent and small values in the erfc!!!
        term_1 = 0.5 * self.c0
        term_2 = self. x - (self.t * self.v / self.R)
        term_3 = np.sqrt(4 * self.D * self.t / self.R)
        print(term_3)
        term_4 = np.exp(self.x * self.v / self.D)
        print(term_4)
        term_5 = self. x + (self.t * self.v / self.R)
        print(term_5[-1])
        term_23 = sci.erfc(term_2 / term_3)
        print(term_23[-1])
        term_53 = sci.erfc(term_5 / term_3)
        print(term_53[-1])
        term_453 = term_4 * term_53
        print(term_453[-1])
        results = term_1 * (term_23 + term_453)
        #results = 0.5 * self.c0 * (sci.erfc((self. x - (self.t * self.v / self.R)) / (np.sqrt(4 * self.D * self.t / self.R))) + np.exp(self.x * self.v / self.D) * sci.erfc((self. x + (self.t * self.v / self.R)) / (np.sqrt(4 * self.D * self.t / self.R))))
        return(results)
    
    # Streamline this with a dictionary!
    def transport(self):
        if self.method == 1:
            return self.inf_flow_pulse()
        elif self.method == 2:
            return self.inf_noflow_pulse()
        elif self.method == 3:
            return self.inf_flow_half()
        elif self.method == 4:
            return self.inf_flow_halfinit()
        elif self.method == 5:
            return self.semi_flow_const()
        else:
            raise ValueError("Unknown method")