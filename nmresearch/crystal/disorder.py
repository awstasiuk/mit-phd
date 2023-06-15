from sympy import inverse_fourier_transform, cos, exp, sqrt, pi
from sympy.abc import x, k
import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.linalg import expm

import crystal
import atom

import pickle


class Disorder:
    r"""
    This class should accept a :type:`Crystal`, and be able to compute the mean field at a given
    site.


    """

    def __init__(self, crystal):
        self._crystal = crystal
        
    @staticmethod
    def fp_coupling(atpt1, atpt2):
    # compute coupling coefficients in terms of p1, p2 in cell basis
    # coupling returned in units of rad/s
        p1 = atpt1.pos()
        p2 = atpt2.pos()
        hbar = 1.05457*10**(-34) # J s / rad
        r = (p2-p1)*10**-10
        dx = np.linalg.norm(r)
        rhat = r/dx
        cos = rhat[2]
        return (10**-7) * hbar * atpt1.get_gamma() * atpt2.get_gamma() * (1-3*cos**2) / dx**3
   
    def cpl_xtal(self, orig_atom, atomlis, shell_rad=1):
        #generates crystal, should be bottleneck
        #specify mean field point as an ordered pair (atom, position), crystal, and list of atoms with which we want to couple
        xtal = self._crystal.generate_crystal(shell_rad)
        new_xtal =[]
        for atompos in xtal:
            if atompos.name() in atomlis:
                atompos.set_cpl(Disorder.fp_coupling(orig_atom, atompos))
            new_xtal.append(atompos)
        return new_xtal
    @staticmethod
    def get_rand_config(cpl_xtal):
    # get a random configuration of n_spins
    # each with spin-dimension s, so spin-1/2 is s=2,
    # spin-2 is s=5, etc (j=(s-1)/2)
        rng = np.random.default_rng()
        config = []
        for atompos in cpl_xtal:
            s= atompos.get_dim()
            config.append(rng.integers(low=0, high=s)-(s-1)/2)
        return config
    #@staticmethod
    def mean_field_calc(self, orig_atom, atomlis, shell_rad=1):
        crystal = self.cpl_xtal(orig_atom, atomlis, shell_rad)
        config = Disorder.get_rand_config(crystal)
        return sum([spin.cpl()*orien for spin, orien in zip(crystal, config)])
        
        
    def variance_estimate(self, orig_atom, atomlis, shell_rad=1):
        crystal = self.cpl_xtal(orig_atom, atomlis, shell_rad)
        return sum([((spin.get_dim()**2-1)/12)*(spin.cpl()**2) for spin in crystal])
    
d1 = .36853
d2 = .39785
fl = atom.Atom(dim_s = 2, gamma = 251.662*10**6, name = "flourine")
#print(fl.get_gamma())
ph = atom.Atom(dim_s = 2, gamma = 108.291*10**6, name = "phosphorous")
unit_cell = {fl: np.array([ [0,0,1/4], [0,0,3/4] ]), ph: np.array([[d1, d2, .25], [-d2, d1 - d2, .25], [d2 - d1, -d1, .25],
               [-d1,-d2, .75], [d2, d2 - d1, .75], [d1-d2, d1, .75]])}
# =============================================================================
#     unit_cell = [ (fl, [0,0,1/4]), (fl,[0,0,3/4]), (ph, [d1, d2, .25]), (ph, [-d2, d1 - d2, .25]), (ph, [d2 - d1, -d1, .25]), (ph, [-d1,-d2, .75]), (ph, [d2, d2 - d1, .75]), (ph, [d1-d2, d1, .75]) ]
# =============================================================================
fp_lat = np.array([[9.375, 9.375*np.cos(120*np.pi/180),0], [0, 9.375 *np.sin(120*np.pi/180),0], [0,0,6.887]])
fp_xtal = crystal.Crystal(unit_cell, fp_lat)
mycalc = Disorder(fp_xtal)
orig_atom = atom.AtomPos(dim_s=2, gamma = 251.662*10**6, pos= [0,0,0.25*6.887], name = "flourine")
#orig_atom = atom.AtomPos(atom =fl, pos=[0,0,0.25])
atomlis = ["phosphorous"]

print(mycalc.variance_estimate(orig_atom, atomlis, 30))
print(mycalc.mean_field_calc(orig_atom, atomlis))

    
    
    
