#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 00:04:35 2023

@author: zhang
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
import pandas as pd



class Emulator:
    
    def __init__(self,equiv,flowrate,elec,solv):
        self.equiv = equiv
        self.flowrate = flowrate
        self.elec = elec
        self.solv = solv
        
    def setParameters(self):
        v = 2E-3*self.flowrate/60
        c0 = np.array([self.equiv*0.30, 0.30, 0, 0]) 
        
        if self.elec=="Acetic_anhydride":
            k0 = np.array([40,0.2])
        else:
            k0 = np.array([30,0.3])

        
        if self.solv=="Toluene":
            phi = 0.35
            k0[0] = 40
            
        if self.solv=="EtOAc":
            phi = 0.35
            k0[0] = 100
        
        if self.solv=="MeCN":
            phi = 0.09*self.flowrate + 0.02
            k0[0] = k0[0] * 1
            
        if self.solv=="THF":

            if self.elec=="Acetic_anhydride":
                phi = 0.19*self.flowrate -0.11
                k0[0] = k0[0] * 1
                
            else:
                phi = 0.05*self.flowrate + 0.06
                k0[0] = k0[0] * 1
        
            
        return c0,v,phi,k0



    def calculate(self):
        
        Nu = np.array([[-1, -1],    # Reaction stoichimetry     
                        [-1, 0],
                        [1, 0],
                        [0, 1]])
                        

        Vr = 0.5 * 1E-3  # Reactor volume, L  

        Nocomp = 4
        Noreac = 2

        order = np.array([[1, 1],     # Reaction order    
                          [1, 0],
                          [0, 0],
                          [0, 0]])
        
        
        def f(tau,c):
            
            
            dcdtau = np.zeros(Nocomp)
            k = np.zeros(Noreac)
            Rate = np.zeros(Noreac)
         
         
            for i in range(0,Noreac):
                Rate[i] = k0[i] * np.prod((c*phi)**order[:,i]) # mixing index phi
            
            for i in range(0,Nocomp):
                dcdtau[i] = np.sum(Rate * Nu[i,:])
                
            return np.array(dcdtau)


        tau_span = np.array([0,Vr/v])
        spaces = np.linspace(tau_span[0],tau_span[1],100)

        soln = solve_ivp(f, tau_span, c0, t_eval=spaces)
        
        tau = soln.t
        cA = soln.y[0]
        cB = soln.y[1]
        cC = soln.y[2]
        cD = soln.y[3]
        
        return tau,cA,cB,cC,cD
    
    def plot(self):
        
        plt.figure()
        plt.plot(tau,cA,'-', label='cA')
        plt.plot(tau,cB,'-', label='cB')
        plt.plot(tau,cC,'-', label='cC')
        plt.plot(tau,cD,'-', label='cD')


        plt.xlabel("res_time/s")
        plt.ylabel("conc/(mol/L)")
        plt.legend()
        plt.show()
        

    
    
test = Emulator(1.4,6,"Acetic_chloride","THF")
# test = Emulator(1.2,4.45,"Acetic_chloride","EtOAc")
# test = Emulator(1.0,2.27,"Acetic_chloride","EtOAc")
c0,v,phi,k0 = test.setParameters()
tau,cA,cB,cC,cD = test.calculate()
yield_pred = round(100*(cC[-1])/0.3,2)
print("yield = ", yield_pred)
test.plot()





# Load data from excel
# data = pd.read_excel('/Users/zhang/Desktop/Reaction opt/#DATA/20230630-TrainingSet-4vars-MaxPro.xlsx', sheet_name='Sheet1')

# pred = []
# for i in range(2,29):
#     test2 = data.iloc[i-2,0:4]
#     test3 = Emulator(test2[0],test2[1],test2[2],test2[3])
#     c0,v,phi,k0 = test3.setParameters()
#     tau,cA,cB,cC,cD = test3.calculate()
#     yield_pred = round(100*(cC[-1])/0.3,2)
#     # test3.plot()
#     print('\nentry =',i)
#     print("yield = ", yield_pred)
    
#     pred.append(yield_pred)




















