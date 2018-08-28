# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 15:19:58 2018

@author: pnola
"""

from mpl_toolkits.basemap import Basemap
import functions as f

ross4= [-106.040772,37.780315]
schmale4 = [-106.0422978,37.78155488]
ground = [-106.03917,37.781644]

print(f.lonlat2km(ground[0],ground[1],ross4[0],ross4[1]))
print(f.lonlat2km(ground[0],ground[1],schmale4[0],schmale4[1]))
print(f.lonlat2km(schmale4[0],schmale4[1],ross4[0],ross4[1]))
import matplotlib.pyplot as plt
plt.scatter(ross4)
plt.scatter(schmale4)
plt.scatter(ground)
