#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 2019

Script containing analytical solutions for specific settings of ADE+BC
according to solution hand-out for Lecture "Hydrogeologic Transport Phenomena"

Governing equation: 1D ADE with linear adsorption and decay
    R dC/dt = D d²C/dx² - q/n dC/dx - mu C

Target variable:
    - C(x,t) = concentration distribution in space and time
Parameters:   
     - R = Retardation
     - D = diffusion rate
     - q = flow rate
     - n = porosity
     - mu = decay rate

* the equation needs to be modified when adsoved solutes decays at the same rate as aqueous solute with 'mu' being replaced by ' R mu'

@author: zech
"""

import numpy as np
import copy
from scipy.special import erfc,erf

DEF_SET = dict(
        v=1.,               # flow velocity
        c0=1.,              # initial concentration
        por=1.,             # porosity
        m0 = 1.,             # initial mass
        deff = 1e-9,        # effective diffusion
        alphaL = 0.1 ,       # dispersivity
        alphaT = 0.01 ,       # dispersivity
        alphaV = 0.001 ,       # dispersivity
        ret_factor = 1.,    # retardation rate
        mu = 0,             # decay rate
        t0 = 0,             # initial time
        t1 = 1,             # first time sill (e.g. end of block injection)
        t2 = 2,             # first time sill 
        sigma2 = 1,         # broadness of gaussian input for advection
        h_y = 1.,           # thickness of source strip in y-direction
        h_z = 1.,           # thickness of source strip in z-direction    
#    mean = 0,
#    var = 1,
#    dim = 2,
#    n_network = 4,
#    n_ens=1000,
#    pressure_in=1,
#    pressure_out=0,
#    ens_seed=False,
#    dir_ens = './',
#    file_settings='settings.csv',
#    file_results='results.csv',
#    delimiter=',',
    )


class Concentration:

    """ Class to calculate concentration distribution for advective transport
        in groundwater with optional other processes involved:
            
    """
   
    def __init__(self, x, t,**settings):

        self.settings=copy.copy(DEF_SET) 
        self.settings.update(settings)   
         
        self.x = np.array(x, ndmin=1, dtype=float)
        self.t = np.array(t, ndmin=1, dtype=float)

        self.tt = np.tile(self.t,(len(self.x),1)).T
        self.xx = np.tile(self.x,(len(self.t),1))
        
        self.v = self.settings['v']         # constant flow velocity
        self.c0 = self.settings['c0']       # initial concentration
        
        self.cxt = np.zeros((len(self.x), len(self.t)), dtype=float)

#    def normalize(self):
#
#        """ Normalizing data according to area under curve (e.g. normalizing pdf)  """
#
#        return self.norm

    def advection(
            self,
            bc = 'constant',
            adsorption = False,
            decay = False,
            decay_adsorbed = False,
            verbose = False,
            **settings
            ):

        """
        Analytical solutions for pure advective transport for different 
        boundary conditions (input of mass at left boundary)
        
        Implicitly assuming an initial mass of 0 in entire domain
        """

        self.settings.update(settings)   

        if adsorption is True:
            ### retardation due to adsorption
            velo= self.v/self.settings['ret_factor']
        else:
            velo = self.v

        arg_x0 = self.tt - self.xx/velo
        arg_t0 = self.xx - velo*self.tt

        if bc == 'constant':
            self.cxt=np.where(arg_t0 <= 0,self.c0,0)
        elif bc == 'pulse':
            self.cxt=np.where(arg_t0 == 0 ,self.c0,0)
        elif bc == 'step':
            self.cxt=self.step_advection(arg_x0)
        elif bc == 'triangle':
            self.cxt=self.triangle_advection(arg_x0)
        elif bc == 'gaussian':
            self.cxt=self.gaussian_advection(arg_t0)
        else:
            print('WARNING: Boundary conditions not implemented')

        if decay:
            self.cxt = self.cxt*self.decay(decay_adsorbed)
            
        if verbose:
            self.print_setting(process = 'Advection', bc = bc)

        return self.cxt

    def diffusion(
            self,
            bc='constant',
            adsorption = False,
            decay = False,
            decay_adsorbed = False,
            verbose = False,
            **settings
            ):

        """
        Analytical solutions for diffusive transport for different 
        intial and boundary conditions 
            - pulse input
            - constant input
            
        Potentially additional processes:
            - Linear equilibrium adsorption (retardation by factor R)
            - Linear decay (exponential decay term)        
        """

        self.settings.update(settings)   

        if adsorption:
            ### retardation due to adsorption
            tt = self.tt/self.settings['ret_factor']                    ### effective time
            m0=self.settings['m0']/self.settings['ret_factor']          ### initial mass per unit area of the sample m0
        else:
            tt = self.tt                    ### effective time
            m0=self.settings['m0']          ### initial mass per unit area of the sample m0 
  
        if bc == 'pulse' :              
            """
            - infinite column
            - slug of tracer at t=0 injected at x=0 (delta distribution)
            - m0 is inital mass per unit area
            - no flow (zero velocity)
            """            
            self.cxt = 0.5* m0/self.settings['por']*self.pulse_diffusion(self.xx,self.settings['deff']*tt)
#            self.cxt = 0.5* m0/self.settings['por']/np.sqrt(np.pi*deff*self.tt)*np.exp( - 0.25 * self.xx*self.xx /deff/self.tt)

        elif bc == 'constant' : 
            """
            - infinite column
            - initial mass distribution: 
                C(x<0) = C_0   (left domain)
                C(x>0)  = 0     (right domain)
            """            
            self.cxt = 0.5* self.c0 *self.constant_diffusion(self.xx,self.settings['deff']*tt)

        elif bc == 'constant_semi' :              
            """
            - semi-infinite column x>0
            - initial mass distribution:
                C(x,t)  = 0
            - boundary condition: (constant input from left)
                C(x=0) = C_0    (left boundary)
            """            
            self.cxt = self.c0 *self.constant_diffusion(self.xx,self.settings['deff']*tt)
#            self.cxt = 0.5* self.c0*erfc(0.5*self.xx /np.sqrt(deff*self.tt))

        elif bc == 'step':   
            cxt_1 = self.constant_diffusion(self.xx,self.settings['deff']*tt)

            arg_tt = self.tt -self.settings['t1']     
            arg_tt[arg_tt<=0] = 1
            cxt_2 = self.constant_diffusion(self.xx,self.settings['deff']*arg_tt)           

            self.cxt = self.c0* np.where(self.tt -self.settings['t1']>0,cxt_1-cxt_2,cxt_1)
          

#            cxt_1 = erfc(0.5*self.xx /np.sqrt(self.settings['deff']*self.tt))
#            arg = self.tt-self.settings['t1']      
#            cxt_2= erfc(0.5*self.xx/np.sqrt(self.settings['deff']*np.where(arg>0,arg,1)))
#            self.cxt = 0.5*self.c0* np.where(arg>0,cxt_1-cxt_2,cxt_1)

        else:
            print('WARNING: Input conditions not implemented')

        if decay:
            self.cxt = self.cxt*self.decay(decay_adsorbed)

        if verbose:
            self.print_setting(process = 'Diffusion', bc = bc)

        return self.cxt

    def ade(
            self,
            bc='constant',
            dispersion = False,
            adsorption = False,
            decay = False,
            decay_adsorbed = False,
            verbose = False,
            **settings
            ):

        """
        Analytical solutions for diffusive transport for different 
        intial and boundary conditions       
        """

        self.settings.update(settings)   
        
        if dispersion:
            d_disp = self.settings['deff'] + self.v*self.settings['alphaL']
            self.settings.update(d_disp = d_disp)
#            print(self.v*self.settings['alphaL'],self.settings['deff'],d_disp)
        else:
            d_disp = self.settings['deff']
        if adsorption:
            ### retardation due to adsorption
            tt = self.tt/self.settings['ret_factor']                    ### effective time
            m0= self.settings['m0']/self.settings['ret_factor']         ### initial mass per unit area of the sample m0
        else:
            tt = self.tt                    ### effective diffusion coefficient
            m0= self.settings['m0']          ### initial mass per unit area of the sample m0 
  
        ### moving coordinate in flow direction
        arg=self.xx - self.v *tt
        spread = d_disp*tt

        if bc == 'pulse' :  
            """
            - infinite column
            - steady velocity v = v (already divided by porosity)
            - slug of tracer at t=0 injected at x=0 (delta distribution)
            - m0 is inital mass per unit area
            """            
            
            self.cxt = 0.5* m0/self.settings['por']*self.pulse_diffusion(arg,spread)

        elif bc == 'constant' : 
            """
            - infinite column
            - steady velocity v = v (already divided by porosity)
            - initial mass distribution: 
                C(x<+0) = C_0   (left domain)
                C(x>0)  = 0     (right domain)
            """            
            self.cxt = 0.5* self.settings['c0'] *self.constant_diffusion(arg,spread)

        elif bc == 'constant_semi' : 
            """
            - semi-infinite column x>0
            - initial mass distribution:
                C(x,t)  = 0
            - boundary condition: (constant input from left)
                C(x=0) = C_0    (left boundary)
            - solution not valid for decay (specific solution in case of decay available, but not implemented)                
            """            
            arg2 = self.xx + self.v *tt
            term = np.exp(self.xx*self.v/d_disp) * self.constant_diffusion(arg2,spread)
            coeff2 = np.where(arg2<30,term,0)
#            if arg2 > 3:
#                coeff2 = 0
#            else:
#                coeff2 = 
            self.cxt = 0.5* self.settings['c0'] *(self.constant_diffusion(arg,spread) + coeff2)
#            self.cxt = 0.5* self.c0 *(self.constant_diffusion(arg,spread)) 
#             self.cxt = 0.5* self.c0 *(self.constant_diffusion(arg,spread) + self.constant_diffusion(arg2,spread)*np.exp(self.xx*self.v/d_disp)) 
#            self.cxt = 0.5* self.c0 *(self.constant_diffusion(arg,tt) + self.constant_diffusion(self.xx + self.v *tt,tt)*np.exp(self.xx*self.v/self.settings['deff'])) 
            decay = False

        else:
            print('WARNING: Input conditions not implemented')

        if decay:
            self.cxt = self.cxt*self.decay(decay_adsorbed)

        if verbose:
            self.print_setting(process = 'Advection and Diffusion', bc = bc)


        return self.cxt

    ###########################################################################
    ### Auxiliary functions 
    ###########################################################################

    def step_advection(self,arg): #,t0=0,t1=1,c0=1.):
        
        cx0=np.zeros_like(arg)
        cx0[(self.settings['t0']<=arg)*(arg<=self.settings['t1'])]=self.c0
        
        return cx0    

    def gaussian_advection(self,arg): 
        
        #c0=np.zeros_like(arg)
        #c0[(0<arg)]= 1/np.sqrt(2.*np.pi*sigma2)*np.exp(-0.5*(arg-t0)*(arg-t0)/sigma2)
        cx0= self.c0/np.sqrt(2.*np.pi*self.settings['sigma2'])*np.exp(-0.5*(arg-self.settings['t0'])**2/self.settings['sigma2'])
        
        return cx0    

    def triangle_advection(self,arg):

#        c0=np.zeros_like(arg)
#        c0[(x0<=arg)*(arg<=x1)]=1 #(arg-x0)/(x1-x0)
#        c0[(x1<arg)*(arg<=x2)]=.2 #(x2-arg) /(x2-x1)
        c01 = np.where((self.settings['t0']<=arg)*(arg<=self.settings['t1']), (arg-self.settings['t0'])*self.c0/(self.settings['t1']-self.settings['t0']),0)
        c02 = np.where((self.settings['t1']<arg)*(arg<=self.settings['t2']), self.c0- (arg - self.settings['t1'])*self.c0/(self.settings['t2']-self.settings['t1']),0)

        return c01 + c02     

    def pulse_diffusion(self,arg,spread):
        """ solution of diffusive transport in 1D for pulse injection"""
        cxt = 1/np.sqrt(np.pi*spread)*np.exp(-0.25 * arg**2 /spread)
#        cxt = 1/np.sqrt(np.pi*self.settings['deff']*tt)*np.exp(-0.25 * arg**2 /self.settings['deff']/tt)
   
        return cxt     
        
    def constant_diffusion(self,arg,spread):
        """ solution of diffusive transport in 1D for constant injection"""

        cxt = erfc(0.5*arg /np.sqrt(spread))
#        cxt = erfc(0.5*arg /np.sqrt(self.settings['deff']*tt))
   
        return cxt     


    def decay(self,decay_adsorbed = False):
        
        if decay_adsorbed:
            """ adsorbed solute also decays at the same rate """
            mu = self.settings['mu']
        else:
            """ adsorbed solute does not decays, thus retarded dacay """
            mu = self.settings['mu']/self.settings['ret_factor']
            
        decay_factor=np.exp(-mu*self.tt)
        
        return decay_factor

    def print_setting(self,process='',bc=''):
        
        print("Analytical solution of ADE:")
        print("Process: {}".format(process))
        print("Input: {}".format(bc))
        print('### Parameters: ###')
        print("Velocity v = {}".format(self.settings['v']))
        print("Porosity: theta = {}".format(self.settings['por']))
        print("Diffusion: D = {}".format(self.settings['deff']))
        print("Dispersivity: alpha = {}".format(self.settings['alphaL']))
        print("Retardation factor: R = {}".format(self.settings['ret_factor']))
        print("decay rate: mu = {}".format(self.settings['mu']))

###############################################################################
###############################################################################
###############################################################################

class Concentration2D:

    """ Class to calculate concentration distribution for ADE in 2D
            
    """
   
    def __init__(self, x, y, t,**settings):

        self.settings=copy.copy(DEF_SET) 
        self.settings.update(settings)   
         
        self.x = np.array(x, ndmin=1, dtype=float)
        self.y = np.array(y, ndmin=1, dtype=float)
        self.t = np.array(t, ndmin=1, dtype=float)

        self.xx,self.yy,self.tt = np.meshgrid(self.x, self.y, self.t, sparse=True, indexing='ij') #sparse=False, indexing='xy')
       
        self.v = self.settings['v']         # constant flow velocity
        
        self.cxyt = np.zeros_like(self.xx, dtype=float)

#        self.cxyt = self.tt*np.sin(self.xx**2 + self.yy**2) / (self.xx**2 + self.yy**2)
        
    def ade(self,
            bc = 'constant',
            dispersion = False,
            adsorption = False,
            decay = False,
            decay_adsorbed = False,
#            verbose = False,
            **settings
            ):
        self.settings.update(settings)   

        if dispersion:
            disp_x = self.v*self.settings['alphaL'] + self.settings['deff']
            disp_y = self.v*self.settings['alphaT'] + self.settings['deff']
 
            self.settings.update(disp_x = disp_x,disp_y=disp_y)   
        else:
            disp_x = self.settings['deff']
            disp_y = self.settings['deff']

        if adsorption:
            ### retardation due to adsorption
            tt = self.tt/self.settings['ret_factor']                    ### effective time
            m0= self.settings['m0']/self.settings['ret_factor']         ### initial mass per unit area of the sample m0
        else:
            tt = self.tt                    ### effective diffusion coefficient
            m0=self.settings['m0']          ### initial mass per unit area of the sample m0 
  
        ### moving coordinate in flow direction
        arg=self.xx - self.v *tt

        if bc == 'pulse' :              
            """
            - infinite column
            - slug of tracer at t=0 injected at x=0 (delta distribution)
            - m0 is inital mass per unit area
            - no flow (zero velocity)
            """          
            term1 = self.settings['por']*np.pi*tt*np.sqrt(disp_x*disp_y)
            term2 = np.exp(-0.25 * arg**2 /(disp_x*tt) - 0.25 * self.yy**2 /(disp_y*tt))     

            self.cxyt = 0.25*m0/term1 * term2

            if decay:
                self.cxyt = self.cxyt*self.decay(decay_adsorbed)

        elif bc == 'constant' : 
            """
            """            

            if decay == True:
                raise ValueError('Solution for constant input in 2D with decay not implemented.')

            term1 = erfc(0.5*arg/np.sqrt(disp_x*tt))
#            term2 = erfc(0.5*(self.yy + 0.5*self.settings['h_y'])/np.sqrt(disp_y*tt))     
#            term3 = erfc(0.5*(self.yy - 0.5*self.settings['h_y'])/np.sqrt(disp_y*tt))     
            term2 = erf(0.5*(self.yy + 0.5*self.settings['h_y'])/np.sqrt(disp_y*tt))     
            term3 = erf(0.5*(self.yy - 0.5*self.settings['h_y'])/np.sqrt(disp_y*tt))     
            print(erf(0.25*self.settings['h_y']/np.sqrt(disp_y*tt)))
            print(erf(-0.25*self.settings['h_y']/np.sqrt(disp_y*tt)))
#            print(0.5*self.settings['h_y']/np.sqrt(disp_y*tt))

            self.cxyt = 0.25*self.settings['c0']*term1 * (term2 - term3)
#            self.cxyt = 0.25*self.settings['c0']*term1 * (term2 + term3)
#            print(np.max(term1),np.max(term2),np.max(term3))
        return self.cxyt

    def decay(self,decay_adsorbed = False):
        
        if decay_adsorbed:
            """ adsorbed solute also decays at the same rate """
            mu = self.settings['mu']
        else:
            """ adsorbed solute does not decays, thus retarded dacay """
            mu = self.settings['mu']/self.settings['ret_factor']
            
        decay_factor=np.exp(-mu*self.tt)
        
        return decay_factor

###############################################################################
###############################################################################
###############################################################################

class Concentration3D:

    """ Class to calculate concentration distribution for ADE in 2D
            
    """
   
    def __init__(self, x, y, z, t,**settings):

        self.settings=copy.copy(DEF_SET) 
        self.settings.update(settings)   
         
        self.x = np.array(x, ndmin=1, dtype=float)
        self.y = np.array(y, ndmin=1, dtype=float)
        self.z = np.array(z, ndmin=1, dtype=float)
        self.t = np.array(t, ndmin=1, dtype=float)

        self.xx,self.yy,self.zz,self.tt = np.meshgrid(self.x, self.y, self.z, self.t, sparse=True, indexing='ij') #sparse=False, indexing='xy')
       
        self.v = self.settings['v']         # constant flow velocity        
        self.cxyzt = np.zeros_like(self.xx, dtype=float)
        
    def ade(self,
            bc = 'constant',
            dispersion = False,
            adsorption = False,
            decay = False,
            decay_adsorbed = False,
#            verbose = False,
            **settings
            ):
        self.settings.update(settings)   

        if dispersion:
            disp_x = self.v*self.settings['alphaL'] + self.settings['deff']
            disp_y = self.v*self.settings['alphaT'] + self.settings['deff']
            disp_z = self.v*self.settings['alphaV'] + self.settings['deff']

            if self.v*self.settings['alphaT']<self.settings['deff']:
                print('Transverse horizontal dispersion smaller than diffusion')
#            print(self.v*self.settings['alphaT'],self.settings['deff'],disp_y)
 
            if self.v*self.settings['alphaV']<self.settings['deff']:
                print('Transverse vertical dispersion smaller than diffusion')
#            print(self.v*self.settings['alphaV'],self.settings['deff'],disp_z)

            self.settings.update(disp_x = disp_x,disp_y=disp_y,disp_z=disp_z)   
        else:
            disp_x = self.settings['deff']
            disp_y = self.settings['deff']
            disp_z = self.settings['deff']

        if adsorption:
            ### retardation due to adsorption
            tt = self.tt/self.settings['ret_factor']                    ### effective time
            m0= self.settings['m0']/self.settings['ret_factor']         ### initial mass per unit area of the sample m0
        else:
            tt = self.tt                    ### effective diffusion coefficient
            m0=self.settings['m0']          ### initial mass per unit area of the sample m0 
  
        ### moving coordinate in flow direction
        arg=self.xx - self.v *tt

        if bc == 'pulse' :              
            """
            - infinite column
            - slug of tracer at t=0 injected at x=0 (delta distribution)
            - m0 is inital mass per unit area
            - no flow (zero velocity)
            """          
            term1 = self.settings['por']*4*np.pi*tt*np.sqrt(4.*np.pi*tt)*np.sqrt(disp_x*disp_y*disp_z)
            term2 = np.exp(-0.25 * arg**2 /(disp_x*tt) - 0.25 * self.yy**2 /(disp_y*tt) - 0.25 * self.zz**2 /(disp_z*tt))     

            self.cxyzt = m0/term1 * term2

            if decay:
                self.cxyzt = self.cxt*self.decay(decay_adsorbed)

        elif bc == 'constant' : 
            """
            """            

            if decay == True:
                raise ValueError('Solution for constant input in 3D with decay not implemented.')
                

            term1 = erfc( 0.5*arg/np.sqrt(disp_x*tt))
            term2 = erf(0.5*(self.yy + 0.5*self.settings['h_y'])/np.sqrt(disp_y*tt))     
            term3 = erf(0.5*(self.yy - 0.5*self.settings['h_y'])/np.sqrt(disp_y*tt))     
            term4 = erf(0.5*(self.zz + 0.5*self.settings['h_z'])/np.sqrt(disp_z*tt))     
            term5 = erf(0.5*(self.zz - 0.5*self.settings['h_z'])/np.sqrt(disp_z*tt))     

#            print(self.settings['h_y'],self.settings['h_z'])
            self.cxyzt = 0.125*self.settings['c0']*term1 * (term2 - term3)* (term4 - term5)

            print(self.settings['c0'])
        return self.cxyzt

    def decay(self,decay_adsorbed = False):
        
        if decay_adsorbed:
            """ adsorbed solute also decays at the same rate """
            mu = self.settings['mu']
        else:
            """ adsorbed solute does not decays, thus retarded dacay """
            mu = self.settings['mu']/self.settings['ret_factor']
            
        decay_factor=np.exp(-mu*self.tt)
        
        return decay_factor

###############################################################################
###############################################################################
###############################################################################


class Diffusion_steady_1D:
    """
    Calculation of steady state Diffusion in layered material:
        xl      -   array with spatial locations of top, bottom and 
                    potentially locations of layers inbetween
        diff    -   array of diffusion values in layers
        ct      -   concentration at top/left BC (float)
        cb      -   concentration at bottom/right BC (float)
        xb      -   array of layer boundary locations                
    """
     
    def __init__(self,xl,diff,ct=1,cb=0,xb=False):
 
        self.xl = np.array(xl,ndmin=1)
        self.diff = np.array(diff,ndmin=1)
        self.ct = ct
        self.cb = cb
                
        self.check()
                
    def check(self):

        ### Number of Layers
        self.nl = len(self.diff)

        if self.nl + 1 != len(self.xl):
            raise ValueError('Number of Diffusion Values does not match number of layers')

        self.a1 = None
        self.b1 = None

    def boundary_concentrations(self):
        
        self.cl = np.zeros(self.nl+1)
        self.cl[0] = self.ct
        self.cl[-1] = self.cb

        diff_x = np.diff(self.xl)

        ### version Alraune :
        self.cl[1] = (self.diff[0]*diff_x[1]*self.cl[0] + self.diff[1]*diff_x[0]*self.cl[-1])/(self.diff[0]*diff_x[1]+self.diff[1]*diff_x[0])
        
        ### version Majid (i):
#        diff_c_total = (self.ct-self.cb)
#        term2 = (diff_x[0]*self.diff[1] + diff_x[1]*self.diff[0])
#        self.cl[1]=self.ct - (diff_x[0]*self.diff[1])/term1*diff_c_total            
#        self.cl[1]=self.cb + (diff_x[1]*self.diff[0])/term2*diff_c_total
#        print((diff_x[0]*self.diff[1])/term2)
#        print((diff_x[1]*self.diff[0])/term2)

        return self.cl        

    def coefficients(self):
        
        ### single layer
        self.a1 = np.zeros_like(self.diff)
        self.b1 = np.zeros_like(self.diff)

        diff_x = np.diff(self.xl)
        diff_c_total = -(self.ct-self.cb)

        if self.nl == 1 :
            self.a1[0] = diff_c_total/(diff_x[0])
            self.b1[0] = self.ct - self.a1 * self.xl[0]

        elif self.nl == 2:
            ### vector containing layer boundary concentrations 
            self.boundary_concentrations()
            
            diff_cl = np.diff(self.cl)
            for i in range(self.nl):
                self.a1[i] = diff_cl[i]/(diff_x[i])
                self.b1[i] = self.cl[i] - self.a1[i] * self.xl[i]

        else: 
            raise ValueError('Solution for more than 2 Layers not implemented')
                           

    def concentration_distribution(self,x):

        if np.any(self.a1) == None:
            self.coefficients()
            
        self.cx = self.a1[0]*x + self.b1[0]

        for i in range(1,self.nl):
            self.cx=np.where(x>self.xl[i],self.a1[i]*x + self.b1[i],self.cx)

        return self.cx

###############################################################################
###############################################################################
###############################################################################

def retardation_factor(kd,rho_b,por):
    """
    calculates retardation factor for linear adsorption

    ret = retardation_factor(kd,rho_b,por)

    """        
    return  1. + kd * rho_b /por

def ctr_dimless(tr,pe = 1):
    """
    calculates concentration in dimensionless time
    
    ctr = ctr_dimless(tr,pe)
    """        
    
    ctr = 0.5*erfc( (0.25*pe/tr)**2*(1-tr))
    
    return ctr

def alpha_plume_witdh(dl,dc):

    """ Calculate Dispersivity from total plume width method:
        99.7% percent of mass within 6 standard deviations around the mean        
    """
    sigma = dl/6.       
    alpha = 0.5*sigma*sigma/dc       
    return alpha

###############################################################################
###############################################################################
###############################################################################
        
    
def Transport1D_const_x(xx,t,q,n,C0):
 
    """ Concentration profile C(x) for linear transport with 
        - constant flow velocity q/n
        - constant input of tracer at t=0
        - concentration C0
    """
        
    Cx=np.zeros(len(xx))

    ###polution front
    x_front=q/n*t
    
    Cx=np.where(xx<=x_front,C0,0)
    
    return Cx

def Transport1D_const_t(x,tt,q,n,C0):
    
    """ BTC C(t) for linear transport with 
        - constant flow velocity q/n
        - constant input of tracer at t=0
        - concentration C0
    """

    Ct=np.zeros(len(tt))

    ###polution front
    t_front=x/q*n
    
    Ct=np.where(tt>=t_front,C0,0)
    
    return Ct

def ADE1D_infinite_slug(x,t,R=1.,D=1.,q=0.,n=1.,mu=0.,M0A=1.):

    """Analytical solution for 1D ADE 
        - in infinite column 
            - BC:  lim x -> pm infinity C(x,t) = 0 $
        - with steady state flow (q=const.)
        - slug of tracer at t=0:
            - IC: C(x,0) = M_{0/A} / R_n delta(x)       
            - M_{0/A} = initial mass per unit area
       

        Parameters
        ---------
        
        
        Return
        ------
    """
    
    fac1=4.*D*t/R
    
    Cxt=M0A/(R*n)/np.sqrt(fac1*np.pi)*np.exp( -(x-q*t/(R*n))**2/fac1 - mu*t/R)
    
    return Cxt

def erfc_approx(b):

    """Approximation of complementary error function (erfc) according to Abramowitz&Stegun, 1972
        works sufficiently well for b>-1 

        Parameter
        ---------
        
        b               :   array
                            positions of evaluation
        
        Return
        ------
        erfc_approx     :   array
                            values of approximate omplementary error function at locations b
    """
    
    a1=0.3480242
    a2=-0.0958798
    a3 =0.7478556
    w=1./(1.+0.47047*b)
    
    erfc_approx=(a1*w+a2*w*w+a3*w*w*w)*np.exp(-b*b)
    
    return erfc_approx