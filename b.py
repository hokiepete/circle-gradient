# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:28:33 2018

@author: pnolan86
"""
'''
from matplotlib import font_manager

font_manager.findfont('cmr10', rebuild_if_missing=True)
'''
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
#from matplotlib import font_manager
import seaborn as sns
sns.set_style('ticks')
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.rc('font', **{'family': 'serif', 'serif': ['cmr10']})#['Computer Modern']})
plt.plot([0,1],[0,1],color='k',label='cow')
plt.ylabel("Banana")
plt.xlabel("$\dot{\rho}$")
plt.legend()
plt.show()




#'''