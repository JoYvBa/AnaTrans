#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Jun 19 13:53:01 2024

@author: jorrit
"""

import numpy as np
import scipy.special as sci

class OneDim:
    """
    Calculates one-dimensional analytical solutions of the Advection Dispersion Equation (ADE) for different scenarios.
    Any units for distance, time and mass can be used, as long as they are consistant across all parameters. Output will have the same units.
     
    Parameters
    ----------
    method : Int
        Integer representing the initial and boundary condition scenarios. See methods section for which scenarios are available and corresponding values for method.
    q : Float
        Darcy velocity (specific discharge), in unit distance per unit time [L/T]
    n : Float
        Porosity of porous medium [-]
    x : Array
        Distance in the x-direction from the origin [L]. Concentration will be calculated for each distance listed in the array.
    t : Array
        Time [T]. Concentration will be calculated for each time listed in the array.
    al : Float, optional
        Longitudinal dispersitivty of the porous medium, in unit distance [L], for steady flow scenarios. Default is 0.
    D_eff : Float, optional
        Effective molecular diffusion coefficient [L^2/T]. The default is 0.
    m0 : Float, optional
        Initial mass per unit area [M/L^2] for pulse injection scenarios. The default is 0.
    c0 : Float, optional
        Initial concentration [M/L^3] for initial or constant concentration scenarios. The default is 0.
    c1 : Float, optional
        Initial concentration [M/L^3] in right half of the domain for scenario 4. The default is 0.
    R : Float, optional
        Linear retardation factor [-]. The default is 1.
    mu : Float, optional
        Linear decay rate [1/T] for scenarios with decay. The default is 0.
    h : Float, optional
        Width [L] of block concentration for scenario 7. The default is 0.
   
    Methods
    -------
    inf_flow_pulse() :    method = 1 
        Infinite column of porous medium with steady flow in the x-direction and a pulse input of mass per unit area m0 at t = 0.
    inf_noflow_pulse() :  method = 2 
        Infinite column of porous medium with no flow and a pulse input of m0 at t = 0.
    inf_flow_half() :     method = 3
        Infinite column of porous medium with steady flow in the x-direction and solute is present in the left half of the domain with concentration c0, at t = 0.
    inf_flow_halfinit() : method = 4
        Infinite column of porous medium with steady flow in the x-direction and solute with concentration c0 in the left half of the domain and concentration c1 in the right half of the domain at t = 0.
    semi_flow_const() :   method = 5
        Semi-infinite column of porous medium with steady flow in the x-direction and constant concentration c0 at x = 0.
    semi_noflow_const() : method = 6
        Semi-inifinte column of porous medium with no flow and constant concentration c0 at x = 0.
    inf_noflow_block() :  method = 7
        Infinite column of porous medium with no flow and a block concentration of width h and concentration c0 centered around x = 0.
    transport() :
        Uses the appropiate scenario depending on the value of method.

    Returns
    -------
    results : Array
        Two-dimensional array containing concentration values for each combination of x and t.
    
    Example 
    -------
    
    >>> import OneDim as D1
    >>> import numpy as np
    >>>
    >>> x = np.linspace(0,200,101)
    >>> t = np.linspace(0,1500,101)
    >>>
    >>> oneD = D1.OneDim(method = 1, q = 0.05, n = 0.33, x = x, t = t, al = 1, D_eff = 1e-4, m0 = 100, R = 1, mu = 0)
    >>> results = oneD.transport()
    """
    
    def __init__(self, method, x, t, q = 0, n = 0, al = 0, D_eff = 0, m0 = 0, c0 = 0, c1 = 0, R = 1, mu = 0, h = 0):
        self.method = method
        self.q        = q
        self.n        = n
        # Calculate flow velocity
        try:
            self.v    = q/n
        except:
            self.v    = 0
        self.x        = x
        # Transpose time array to the shape [n,1] for proper broadcasting
        self.t        = t[ : , np.newaxis]
        self.D_eff    = D_eff
        # Calculate the dispersion coefficient
        self.D        = D_eff + al * self.v
        self.m0       = m0
        self.c0       = c0
        self.c1       = c1
        self.R        = R
        self.mu       = mu
        self.h        = h
    
    # Replace q/n in methods by v
        
    # Method 1; Infinite column with steady flow in the x-direction and with pulse injection at t = 0
    def inf_flow_pulse(self):         
        initial = self.m0 / (self.R * self.n * np.sqrt((4 * np.pi * self.D * self.t) / self.R))
        exponent = np.exp((-((self.x - (self.v * self.t / self.R))**2) / (4 * np.pi * self.D * self.t / self.R)) - ((self.mu * self.t) / self.R))
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
        results = 0.5 * self.c0 * sci.erfc((self.x - (self.v * self.t / self.R)) / (4*self.D * self.t / self.R)) * np.exp((-self.mu * self.t) / self.R)
        return(results)
    
    # Method 4: Infinite column with steady flow, the solute is present with c0 at the left half of the infinite domain and with c1 at the right half at t = 0
    def inf_flow_halfinit(self):
        results = (self.c1 + 0.5 * (self.c0 - self.c1) * sci.erfc((self.x - (self.v * self.t / self.R)) / (4*self.D * self.t / self.R)) * np.exp((-self.mu * self.t)) / self.R)
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
    
    # Method 6: Semi-inifinte column with no flow and constant concentration at x = 0
    def semi_noflow_const(self):
        results = self.c0 * sci.erfc(self.x / (np.sqrt(4 * self.D_eff * self.t / self.R))) * np.exp(- self.mu * self.t / self.R)
        return(results)
    
    # Method 7: Infinite column with no flow and a block concentration of width h centered around x = 0
    def inf_noflow_block(self):
        results = 0.5 * self.c0 * (sci.erf((self.h / 2 - self.x) / np.sqrt(4 * self.D_eff * self.t)) + sci.erf((self.h / 2 + self.x) / np.sqrt(4 * self.D_eff * self.t)))
        return(results)
    
    # Streamline this with a dictionary!
    def methods(self):
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
        elif self.method == 6:
            return self.semi_noflow_const()
        elif self.method == 7:
            return self.inf_noflow_block()
        else:
            raise ValueError("Unknown method")
            
def transport(method, x, t, q = 0, n = 0, al = 0, D_eff = 0, m0 = 0, c0 = 0, c1 = 0, R = 1, mu = 0, h = 0):
    """
    Calculates one-dimensional analytical solutions of the Advection Dispersion Equation (ADE) for different scenarios.
    Any units for distance, time and mass can be used, as long as they are consistant across all parameters. Output will have the same units.
     
    Parameters
    ----------
    method : Int
        Integer representing the initial and boundary condition scenarios. See methods section for which scenarios are available and corresponding values for method.
    q : Float
        Darcy velocity (specific discharge), in unit distance per unit time [L/T]
    n : Float
        Porosity of porous medium [-]
    x : Array
        Distance in the x-direction from the origin [L]. Concentration will be calculated for each distance listed in the array.
    t : Array
        Time [T]. Concentration will be calculated for each time listed in the array.
    al : Float, optional
        Longitudinal dispersitivty of the porous medium, in unit distance [L], for steady flow scenarios. Default is 0.
    D_eff : Float, optional
        Effective molecular diffusion coefficient [L^2/T]. The default is 0.
    m0 : Float, optional
        Initial mass per unit area [M/L^2] for pulse injection scenarios. The default is 0.
    c0 : Float, optional
        Initial concentration [M/L^3] for initial or constant concentration scenarios. The default is 0.
    c1 : Float, optional
        Initial concentration [M/L^3] in right half of the domain for scenario 4. The default is 0.
    R : Float, optional
        Linear retardation factor [-]. The default is 1.
    mu : Float, optional
        Linear decay rate [1/T] for scenarios with decay. The default is 0.
    h : Float, optional
        Width [L] of block concentration for scenario 7. The default is 0.
   
    Methods
    -------
    inf_flow_pulse() :    method = 1 
        Infinite column of porous medium with steady flow in the x-direction and a pulse input of mass per unit area m0 at t = 0.
    inf_noflow_pulse() :  method = 2 
        Infinite column of porous medium with no flow and a pulse input of m0 at t = 0.
    inf_flow_half() :     method = 3
        Infinite column of porous medium with steady flow in the x-direction and solute is present in the left half of the domain with concentration c0, at t = 0.
    inf_flow_halfinit() : method = 4
        Infinite column of porous medium with steady flow in the x-direction and solute with concentration c0 in the left half of the domain and concentration c1 in the right half of the domain at t = 0.
    semi_flow_const() :   method = 5
        Semi-infinite column of porous medium with steady flow in the x-direction and constant concentration c0 at x = 0.
    semi_noflow_const() : method = 6
        Semi-inifinte column of porous medium with no flow and constant concentration c0 at x = 0.
    inf_noflow_block() :  method = 7
        Infinite column of porous medium with no flow and a block concentration of width h and concentration c0 centered around x = 0.
    transport() :
        Uses the appropiate scenario depending on the value of method.

    Returns
    -------
    results : Array
        Two-dimensional array containing concentration values for each combination of x and t.
    
    Example 
    -------
    
    >>> import OneDim as D1
    >>> import numpy as np
    >>>
    >>> x = np.linspace(0,200,101)
    >>> t = np.linspace(0,1500,101)
    >>>
    >>> oneD = D1.OneDim(method = 1, q = 0.05, n = 0.33, x = x, t = t, al = 1, D_eff = 1e-4, m0 = 100, R = 1, mu = 0)
    >>> results = oneD.transport()
    """
    return(OneDim(method = method, q = q, n = n, x = x, t = t, al = al, D_eff = D_eff, m0 = m0, c0 = c0, c1 = c1, R = R, mu = mu, h = h).methods())
            
            