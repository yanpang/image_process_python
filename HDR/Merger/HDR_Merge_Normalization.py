# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 12:36:18 2022

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math

matplotlib.use('TkAgg')
ER = 4
CG = 1.9
short_exposure = 1
long_exposure = ER*short_exposure
medium_exposure =math.sqrt(ER)

short_to_long_norm_factor = ER
medium_to_long_norm_factor = math.sqrt(ER)

long_to_short_norm_factor = 1/ER
medium_to_short_norm_factor = 1/math.sqrt(ER)

print("L:M:S = %d\t%d\t%d" %(long_exposure,medium_exposure,short_exposure))
 
axis_X_lux = np.linspace(0,5000,num=1000)
axis_Y_S = np.array(axis_X_lux*(1/(CG)*short_exposure),dtype='uint32')
axis_Y_M = np.array(axis_X_lux*(1/(CG)*medium_exposure),dtype='uint32')
axis_Y_L = np.array(axis_X_lux*(1/(CG)*long_exposure),dtype='uint32')

#saturation clipping
axis_Y_S = np.where(axis_Y_S>1023,1023,axis_Y_S)
axis_Y_M = np.where(axis_Y_M>1023,1023,axis_Y_M)
axis_Y_L = np.where(axis_Y_L>1023,1023,axis_Y_L)
#plot 
plt.plot(axis_X_lux,axis_Y_S,label='short',linewidth=3)
plt.plot(axis_X_lux,axis_Y_M,label='medium',linewidth=3)
plt.plot(axis_X_lux,axis_Y_L,label='long',linewidth=3)



#normalization A : S,M->L
axis_Y_S_norm_A = axis_Y_S*short_to_long_norm_factor
axis_Y_M_norm_A = axis_Y_M*medium_to_long_norm_factor
axis_Y_L_norm_A = axis_Y_L
#plot 
plt.plot(axis_X_lux,axis_Y_S_norm_A,'--',label='short normA',)
plt.plot(axis_X_lux,axis_Y_M_norm_A,'--',label='medium normA')
plt.plot(axis_X_lux,axis_Y_L_norm_A,'--',label='long normA')

#normalization B : L,M->S
axis_Y_S_norm_B = axis_Y_S
axis_Y_M_norm_B = axis_Y_M*medium_to_short_norm_factor
axis_Y_L_norm_B = axis_Y_L*long_to_short_norm_factor
#plot 
plt.plot(axis_X_lux,axis_Y_S_norm_B,label='short normB',linewidth=1)
plt.plot(axis_X_lux,axis_Y_M_norm_B,label='medium normB',linewidth=1)
plt.plot(axis_X_lux,axis_Y_L_norm_B,label='long normB',linewidth=1)


plt.xlim([0,2000])
plt.ylim([0,5000])
plt.legend()
plt.grid()

plt.show(block='True')