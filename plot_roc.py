# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 16:46:36 206

@author: pnola
"""

import numpy as np
import h5py as hp
import matplotlib.pyplot as plt


with hp.File('850mb_300m_10min_NAM_Rhodot_Origin_t=0-215hrs_Sept2017.hdf5','r') as data:
    rhodot = data['rhodot'][:].squeeze()
    s1 = data['s1'][:].squeeze()
    t = data['t'][:].squeeze()
    data.close()
for radius in np.linspace(100,10000,37):#[1,10,100,500,1000,5000,10000,15000]:
    rhodot_TPR = []
    rhodot_FPR = []
    s1_TPR = []
    s1_FPR = []
    for percent in np.arange(0,101,1):
        passing_times = np.load('passing_files/passing_times_{0:03d}th_percentile_radius={1:05d}.npy'.format(90,int(radius)))+24            
        #passing_times = np.load('passing_files/passing_times_{0:03d}th_percentile_radius={1:05d}.npy'.format(percent,radius))+24            
        thresh_rhodot = -np.percentile(-rhodot[rhodot<0],percent)
        thresh_s1 = -np.percentile(-s1[s1<0],percent)
        
        rhodot_true_positive = 0
        rhodot_false_positive = 0
        rhodot_true_negative = 0
        rhodot_false_negative = 0
        #for tt in range(1267):
        for tt in range(24,1291):#1267):
            if len([x for x in passing_times if x == tt])!=0:
                if rhodot[tt]<thresh_rhodot:    
                    rhodot_true_positive += 1
                else:
                    rhodot_false_negative += 1
            else:
                if rhodot[tt]<thresh_rhodot:    
                    rhodot_false_positive += 1
                else:
                    rhodot_true_negative += 1
        
        rhodot_total_true = rhodot_true_positive+rhodot_true_negative
        rhodot_total_false = rhodot_false_positive+rhodot_false_negative
        rhodot_total_positive = rhodot_true_positive+rhodot_false_positive
        rhodot_total_negative = rhodot_true_negative+rhodot_false_negative
        
        if rhodot_true_positive+rhodot_false_negative != 0:
            rhodot_TPR.append(rhodot_true_positive/(rhodot_true_positive+rhodot_false_negative))
        else:
            rhodot_TPR.append(np.nan)
        
        if rhodot_false_positive+rhodot_true_negative != 0:
            rhodot_FPR.append(rhodot_false_positive/(rhodot_false_positive+rhodot_true_negative))
        else:
            rhodot_FPR.append(np.nan)
        
        
        s1_true_positive = 0
        s1_false_positive = 0
        s1_true_negative = 0
        s1_false_negative = 0
        #for tt in range(1267):
        for tt in range(24,1291):#1267):
            if len([x for x in passing_times if x == tt])!=0:
                if s1[tt]<thresh_s1:    
                    s1_true_positive += 1
                else:
                    s1_false_negative += 1
            else:
                if s1[tt]<thresh_s1:    
                    s1_false_positive += 1
                else:
                    s1_true_negative += 1
        
        s1_total_true = s1_true_positive+s1_true_negative
        s1_total_false = s1_false_positive+s1_false_negative
        s1_total_positive = s1_true_positive+s1_false_positive
        s1_total_negative = s1_true_negative+s1_false_negative
        
        if s1_true_positive+s1_false_negative != 0:
            s1_TPR.append(s1_true_positive/(s1_true_positive+s1_false_negative))
        else:
            s1_TPR.append(np.nan)
        
        if s1_false_positive+s1_true_negative != 0:
            s1_FPR.append(s1_false_positive/(s1_false_positive+s1_true_negative))
        else:
            s1_FPR.append(np.nan)
        
        #s1_TPR.append(s1_true_positive/(s1_true_positive+s1_false_negative))
        #s1_FPR.append(s1_false_positive/(s1_false_positive+s1_true_negative))
        '''
        plt.close('all')
        plt.figure(3)
        plt.plot(s1,'r')
        plt.axhline(thresh_s1,color='b',linestyle='dashed')
        #plt.plot(rhodot,'r')
        #plt.axhline(thresh_rhodot,color='r',linestyle='dashed')
        for x in passing_times:
            plt.axvline(x,alpha=0.3)
        plt.xlabel('Time Step')
        plt.title('LCS Passing Times')
        plt.savefig('LCS_passing_times_radius={0:05d}_percentile={1:03d}.png'.format(radius,percent), transparent=False, bbox_inches='tight',pad_inches=0)
        '''
    
    plt.close('all')
    plt.figure(1)
    plt.plot([0,1],[0,1],'b--')
    plt.plot(rhodot_FPR,rhodot_TPR,'r-')
    #plt.scatter(rhodot_FPR,rhodot_TPR,color='r')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Rhodot, radius = {0} meters'.format(radius))
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.savefig('Rhodot_ROC_radius={0:05d}_no_ftle_thresh.png'.format(int(radius)), transparent=False, bbox_inches='tight',pad_inches=0)
    #plt.savefig('Rhodot_ROC_radius={0:05d}.png'.format(radius), transparent=False, bbox_inches='tight',pad_inches=0)
    
    plt.figure(2)
    plt.plot([0,1],[0,1],'b--')
    plt.plot(s1_FPR,s1_TPR,'r-')
    #plt.scatter(s1_FPR,s1_TPR,color='r')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('s1, radius = {0} meters'.format(radius))
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.savefig('S1_ROC_radius={0:05d}_no_ftle_thresh.png'.format(int(radius)), transparent=False, bbox_inches='tight',pad_inches=0)
    #plt.savefig('S1_ROC_radius={0:05d}.png'.format(radius), transparent=False, bbox_inches='tight',pad_inches=0)
    '''
    plt.figure(3)
    plt.plot(s1,'b')
    plt.axhline(thresh_s1,'b--')
    plt.plot(rhodot,'r')
    plt.axhline(thresh_rhodot,'r--')
    for x in passing_times:
        plt.axvline(x,alpha=0.3)
    plt.xlabel('Time Step')
    plt.title('LCS Passing Times')
    plt.savefig('LCS_passing_times_radius={0:05d}.png'.format(radius), transparent=False, bbox_inches='tight',pad_inches=0)
    '''