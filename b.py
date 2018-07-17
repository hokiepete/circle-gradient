# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:28:33 2018

@author: pnolan86
"""

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
#from matplotlib import font_manager
import seaborn as sns
sns.set_style('ticks')
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.usetex']=True
plt.rc('font', family='serif')
#plt.rc('mathtext', font='serif')
plt.plot([0,1],[0,1],color='k',label='cow')
plt.ylabel("Banana")
plt.xlabel("$\dot{\rho}$")
plt.legend()
plt.show()