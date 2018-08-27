# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 17:01:01 2018

@author: pnola
"""
import numpy as np
percent = 0
radius = 1000
ps=np.load('passing_files/passing_times_{0:02d}th_percentile_radius={1:04d}.np.npy'.format(percent,radius))+24
            