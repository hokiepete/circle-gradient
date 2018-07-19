# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:28:33 2018

@author: pnolan86
"""
'''
from matplotlib import font_manager

font_manager.findfont('cmr10', rebuild_if_missing=True)
'''
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.close('all')
#from matplotlib import font_manager
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.rc('font', **{'family': 'serif', 'serif': ['cmr10']})#['Computer Modern']})
plt.plot([0,1],[-1,1],color='k',label='cow')
plt.ylabel("$\\alpha$")
plt.xlabel("$\dot{\\rho}$")
plt.legend()
plt.show()




#'''