# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 16:46:36 206

@author: pnola
"""

import numpy as np
import h5py as hp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['lines.linewidth']=1
matplotlib.rcParams['lines.markersize']=2
plt.rc('font', **{'family': 'serif', 'serif': ['cmr10']})
titlefont = {'fontsize':12}
labelfont = {'fontsize':10}
tickfont = {'fontsize':8}
time_step_offset = 24
rhodot_TPR_all_int05 = []
rhodot_FPR_all_int05 = []
s1_TPR_all_int05 = []
s1_FPR_all_int05 = []
rhodot_TPR_all_int1 = []
rhodot_FPR_all_int1 = []
s1_TPR_all_int1 = []
s1_FPR_all_int1 = []
rhodot_TPR_all_int2 = []
rhodot_FPR_all_int2 = []
s1_TPR_all_int2 = []
s1_FPR_all_int2 = []
#with hp.File('850mb_300m_10min_NAM_Rhodot_Origin_t=0-215hrs_Sept2017.hdf5','r') as data:
with hp.File('hunterdata_r=02km_interpolated_2_cridges.hdf5','r') as data:
    rhodot = data['rhodot'][:].squeeze()
    s1 = data['s1'][:].squeeze()
    t = data['t'][:].squeeze()
    data.close()
for radius in [400,800,1200,1600,2000,3000,5000,7500,10000]:#[200,500,800,1000,2000,3500,5000,7500,10000]:#[200,300,400]:#np.append(np.linspace(100,10000,37),np.array([1,10,100,500,1000,5000,10000,15000])):
    radius=int(radius)
    rhodot_TPR_int1 = []
    rhodot_FPR_int1 = []
    s1_TPR_int1 = []
    s1_FPR_int1 = []
    rhodot_TPR_int2 = []
    rhodot_FPR_int2 = []
    s1_TPR_int2 = []
    s1_FPR_int2 = []
    rhodot_TPR_int05 = []
    rhodot_FPR_int05 = []
    s1_TPR_int05 = []
    s1_FPR_int05 = []
    for percent in np.linspace(0,100,2):#np.arange(0,101,1):
        print(radius,percent)
        passing_times = np.load('passing_files/passing_times_{0:03d}th_percentile_radius={1:05d}_int=-1.npy'.format(90,radius))+time_step_offset          
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
            rhodot_TPR_int1.append(rhodot_true_positive/(rhodot_true_positive+rhodot_false_negative))
        else:
            rhodot_TPR_int1.append(np.nan)
        
        if rhodot_false_positive+rhodot_true_negative != 0:
            rhodot_FPR_int1.append(rhodot_false_positive/(rhodot_false_positive+rhodot_true_negative))
        else:
            rhodot_FPR_int1.append(np.nan)
        
        
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
            s1_TPR_int1.append(s1_true_positive/(s1_true_positive+s1_false_negative))
        else:
            s1_TPR_int1.append(np.nan)
        
        if s1_false_positive+s1_true_negative != 0:
            s1_FPR_int1.append(s1_false_positive/(s1_false_positive+s1_true_negative))
        else:
            s1_FPR_int1.append(np.nan)

        passing_times = np.load('passing_files/passing_times_{0:03d}th_percentile_radius={1:05d}_int=-2.npy'.format(90,radius))+time_step_offset            
        
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
            rhodot_TPR_int2.append(rhodot_true_positive/(rhodot_true_positive+rhodot_false_negative))
        else:
            rhodot_TPR_int2.append(np.nan)
        
        if rhodot_false_positive+rhodot_true_negative != 0:
            rhodot_FPR_int2.append(rhodot_false_positive/(rhodot_false_positive+rhodot_true_negative))
        else:
            rhodot_FPR_int2.append(np.nan)
        
        
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
            s1_TPR_int2.append(s1_true_positive/(s1_true_positive+s1_false_negative))
        else:
            s1_TPR_int2.append(np.nan)
        
        if s1_false_positive+s1_true_negative != 0:
            s1_FPR_int2.append(s1_false_positive/(s1_false_positive+s1_true_negative))
        else:
            s1_FPR_int2.append(np.nan)
        
        passing_times = np.load('passing_files/passing_times_{0:03d}th_percentile_radius={1:05d}_int=-0,5.npy'.format(90,radius))+time_step_offset            
        
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
            rhodot_TPR_int05.append(rhodot_true_positive/(rhodot_true_positive+rhodot_false_negative))
        else:
            rhodot_TPR_int05.append(np.nan)
        
        if rhodot_false_positive+rhodot_true_negative != 0:
            rhodot_FPR_int05.append(rhodot_false_positive/(rhodot_false_positive+rhodot_true_negative))
        else:
            rhodot_FPR_int05.append(np.nan)
        
        
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
            s1_TPR_int05.append(s1_true_positive/(s1_true_positive+s1_false_negative))
        else:
            s1_TPR_int05.append(np.nan)
        
        if s1_false_positive+s1_true_negative != 0:
            s1_FPR_int05.append(s1_false_positive/(s1_false_positive+s1_true_negative))
        else:
            s1_FPR_int05.append(np.nan)

    rhodot_TPR_all_int1.append(rhodot_TPR_int1)
    rhodot_FPR_all_int1.append(rhodot_FPR_int1)
    s1_TPR_all_int1.append(s1_TPR_int1)
    s1_FPR_all_int1.append(s1_FPR_int1)
    rhodot_TPR_all_int2.append(rhodot_TPR_int2)
    rhodot_FPR_all_int2.append(rhodot_FPR_int2)
    s1_TPR_all_int2.append(s1_TPR_int2)
    s1_FPR_all_int2.append(s1_FPR_int2)
    rhodot_TPR_all_int05.append(rhodot_TPR_int05)
    rhodot_FPR_all_int05.append(rhodot_FPR_int05)
    s1_TPR_all_int05.append(s1_TPR_int05)
    s1_FPR_all_int05.append(s1_FPR_int05)

#"""
plt.close('all')
figwidth = 6
FigSize=(figwidth, figwidth)
plt.figure(1,figsize=FigSize)
gs = gridspec.GridSpec(3, 3)
gs.update(wspace=0.05, hspace=0.05)
plt.subplot(gs[0])
plt.plot([0,1],[0,1],'k:')
plt.plot(rhodot_FPR_all_int05[0],rhodot_TPR_all_int05[0],'g--')
plt.scatter(rhodot_FPR_all_int05[0][::20],rhodot_TPR_all_int05[0][::20],color='g')
plt.plot(rhodot_FPR_all_int1[0],rhodot_TPR_all_int1[0],'r-')
plt.scatter(rhodot_FPR_all_int1[0][::20],rhodot_TPR_all_int1[0][::20],color='r')
plt.plot(rhodot_FPR_all_int2[0],rhodot_TPR_all_int2[0],'b-.')
plt.scatter(rhodot_FPR_all_int2[0][::20],rhodot_TPR_all_int2[0][::20],color='b')
plt.annotate('0.4km', xy=(0.74, 0.02), xycoords='axes fraction')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.yticks(**tickfont)
plt.xticks([])
plt.axis('equal')

ax=plt.subplot(gs[1])
plt.plot([0,1],[0,1],'k:')
plt.plot(rhodot_FPR_all_int05[1],rhodot_TPR_all_int05[1],'g--',label='0.5hr')
plt.scatter(rhodot_FPR_all_int05[1][::20],rhodot_TPR_all_int05[1][::20],color='g')
plt.plot(rhodot_FPR_all_int1[1],rhodot_TPR_all_int1[1],'r-',label='1hr')
plt.scatter(rhodot_FPR_all_int1[1][::20],rhodot_TPR_all_int1[1][::20],color='r')
plt.plot(rhodot_FPR_all_int2[1],rhodot_TPR_all_int2[1],'b-.',label='2hr')
plt.scatter(rhodot_FPR_all_int2[1][::20],rhodot_TPR_all_int2[1][::20],color='b')
plt.annotate('0.8km', xy=(0.74, 0.02), xycoords='axes fraction')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.xticks([])
plt.yticks([])
plt.axis('equal')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),  shadow=False, ncol=3)

plt.subplot(gs[2])
plt.plot([0,1],[0,1],'k:')
plt.plot(rhodot_FPR_all_int05[2],rhodot_TPR_all_int05[2],'g--')
plt.scatter(rhodot_FPR_all_int05[2][::20],rhodot_TPR_all_int05[2][::20],color='g')
plt.plot(rhodot_FPR_all_int1[2],rhodot_TPR_all_int1[2],'r-')
plt.scatter(rhodot_FPR_all_int1[2][::20],rhodot_TPR_all_int1[2][::20],color='r')
plt.plot(rhodot_FPR_all_int2[2],rhodot_TPR_all_int2[2],'b-.')
plt.scatter(rhodot_FPR_all_int2[2][::20],rhodot_TPR_all_int2[2][::20],color='b')
plt.annotate('1.2km', xy=(0.74, 0.02), xycoords='axes fraction')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.xticks([])
plt.yticks([])
plt.axis('equal')

plt.subplot(gs[3])
plt.plot([0,1],[0,1],'k:')
plt.plot(rhodot_FPR_all_int05[3],rhodot_TPR_all_int05[3],'g--')
plt.scatter(rhodot_FPR_all_int05[3][::20],rhodot_TPR_all_int05[3][::20],color='g')
plt.plot(rhodot_FPR_all_int1[3],rhodot_TPR_all_int1[3],'r-')
plt.scatter(rhodot_FPR_all_int1[3][::20],rhodot_TPR_all_int1[3][::20],color='r')
plt.plot(rhodot_FPR_all_int2[3],rhodot_TPR_all_int2[3],'b-.')
plt.scatter(rhodot_FPR_all_int2[3][::20],rhodot_TPR_all_int2[3][::20],color='b')
plt.annotate('1.6km', xy=(0.74, 0.02), xycoords='axes fraction')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.yticks(**tickfont)
plt.xticks([])
plt.ylabel('True Positive Rate',**labelfont)
plt.axis('equal')

plt.subplot(gs[4])
plt.plot([0,1],[0,1],'k:')
plt.plot(rhodot_FPR_all_int05[4],rhodot_TPR_all_int05[4],'g--')
plt.scatter(rhodot_FPR_all_int05[4][::20],rhodot_TPR_all_int05[4][::20],color='g')
plt.plot(rhodot_FPR_all_int1[4],rhodot_TPR_all_int1[4],'r-')
plt.scatter(rhodot_FPR_all_int1[4][::20],rhodot_TPR_all_int1[4][::20],color='r')
plt.plot(rhodot_FPR_all_int2[4],rhodot_TPR_all_int2[4],'b-.')
plt.scatter(rhodot_FPR_all_int2[4][::20],rhodot_TPR_all_int2[4][::20],color='b')
plt.annotate('2.0km', xy=(0.74, 0.02), xycoords='axes fraction')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.xticks([])
plt.yticks([])
plt.axis('equal')

plt.subplot(gs[5])
plt.plot([0,1],[0,1],'k:')
plt.plot(rhodot_FPR_all_int05[5],rhodot_TPR_all_int05[5],'g--')
plt.scatter(rhodot_FPR_all_int05[5][::20],rhodot_TPR_all_int05[5][::20],color='g')
plt.plot(rhodot_FPR_all_int1[5],rhodot_TPR_all_int1[5],'r-')
plt.scatter(rhodot_FPR_all_int1[5][::20],rhodot_TPR_all_int1[5][::20],color='r')
plt.plot(rhodot_FPR_all_int2[5],rhodot_TPR_all_int2[5],'b-.')
plt.scatter(rhodot_FPR_all_int2[5][::20],rhodot_TPR_all_int2[5][::20],color='b')
plt.annotate('3.0km', xy=(0.74, 0.02), xycoords='axes fraction')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.xticks([])
plt.yticks([])
plt.axis('equal')

plt.subplot(gs[6])
plt.plot([0,1],[0,1],'k:')
plt.plot(rhodot_FPR_all_int05[6],rhodot_TPR_all_int05[6],'g--')
plt.scatter(rhodot_FPR_all_int05[6][::20],rhodot_TPR_all_int05[6][::20],color='g')
plt.plot(rhodot_FPR_all_int1[6],rhodot_TPR_all_int1[6],'r-')
plt.scatter(rhodot_FPR_all_int1[6][::20],rhodot_TPR_all_int1[6][::20],color='r')
plt.plot(rhodot_FPR_all_int2[6],rhodot_TPR_all_int2[6],'b-.')
plt.scatter(rhodot_FPR_all_int2[6][::20],rhodot_TPR_all_int2[6][::20],color='b')
plt.annotate('5.0km', xy=(0.74, 0.02), xycoords='axes fraction')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.axis('equal')

plt.subplot(gs[7])
plt.plot([0,1],[0,1],'k:')
plt.plot(rhodot_FPR_all_int05[7],rhodot_TPR_all_int05[7],'g--')
plt.scatter(rhodot_FPR_all_int05[7][::20],rhodot_TPR_all_int05[7][::20],color='g')
plt.plot(rhodot_FPR_all_int1[7],rhodot_TPR_all_int1[7],'r-')
plt.scatter(rhodot_FPR_all_int1[7][::20],rhodot_TPR_all_int1[7][::20],color='r')
plt.plot(rhodot_FPR_all_int2[7],rhodot_TPR_all_int2[7],'b-.')
plt.scatter(rhodot_FPR_all_int2[7][::20],rhodot_TPR_all_int2[7][::20],color='b')
plt.annotate('7.5km', xy=(0.74, 0.02), xycoords='axes fraction')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.xlabel('False Positive Rate',**labelfont)
plt.yticks([])
plt.xticks(**tickfont)
plt.axis('equal')
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),  shadow=True, ncol=3)

plt.subplot(gs[8])
plt.plot([0,1],[0,1],'k:')
plt.plot(rhodot_FPR_all_int05[8],rhodot_TPR_all_int05[8],'g--')
plt.scatter(rhodot_FPR_all_int05[8][::20],rhodot_TPR_all_int05[8][::20],color='g')
plt.plot(rhodot_FPR_all_int1[8],rhodot_TPR_all_int1[8],'r-')
plt.scatter(rhodot_FPR_all_int1[8][::20],rhodot_TPR_all_int1[8][::20],color='r')
plt.plot(rhodot_FPR_all_int2[8],rhodot_TPR_all_int2[8],'b-.')
plt.scatter(rhodot_FPR_all_int2[8][::20],rhodot_TPR_all_int2[8][::20],color='b')
plt.annotate('10.0km', xy=(0.70, 0.02), xycoords='axes fraction')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.yticks([])
plt.xticks(**tickfont)
plt.axis('equal')

#plt.savefig('Rhodot_subplots.eps'.format(radius), transparent=False, bbox_inches='tight',pad_inches=0)
#plt.savefig('Rhodot_subplots.png'.format(radius), transparent=False, bbox_inches='tight',pad_inches=0)
plt.savefig('Rhodot_subplots_hunterflight.eps'.format(radius), transparent=False, bbox_inches='tight',pad_inches=0)
plt.savefig('Rhodot_subplots_hunterflight.png'.format(radius), transparent=False, bbox_inches='tight',pad_inches=0)

plt.figure(2,figsize=FigSize)
gs = gridspec.GridSpec(3, 3)
gs.update(wspace=0.05, hspace=0.05)
plt.subplot(gs[0])
plt.plot([0,1],[0,1],'k:')
plt.plot(s1_FPR_all_int05[0],s1_TPR_all_int05[0],'g--')
plt.scatter(s1_FPR_all_int05[0][::20],s1_TPR_all_int05[0][::20],color='g')
plt.plot(s1_FPR_all_int1[0],s1_TPR_all_int1[0],'r-')
plt.scatter(s1_FPR_all_int1[0][::20],s1_TPR_all_int1[0][::20],color='r')
plt.plot(s1_FPR_all_int2[0],s1_TPR_all_int2[0],'b-.')
plt.scatter(s1_FPR_all_int2[0][::20],s1_TPR_all_int2[0][::20],color='b')
plt.annotate('0.4km', xy=(0.74, 0.02), xycoords='axes fraction')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.yticks(**tickfont)
plt.xticks([])
plt.axis('equal')

ax=plt.subplot(gs[1])
plt.plot([0,1],[0,1],'k:')
plt.plot(s1_FPR_all_int05[1],s1_TPR_all_int05[1],'g--',label='0.5hr')
plt.scatter(s1_FPR_all_int05[1][::20],s1_TPR_all_int05[1][::20],color='g')
plt.plot(s1_FPR_all_int1[1],s1_TPR_all_int1[1],'r-',label='1hr')
plt.scatter(s1_FPR_all_int1[1][::20],s1_TPR_all_int1[1][::20],color='r')
plt.plot(s1_FPR_all_int2[1],s1_TPR_all_int2[1],'b-.',label='2hr')
plt.scatter(s1_FPR_all_int2[1][::20],s1_TPR_all_int2[1][::20],color='b')
plt.annotate('0.8km', xy=(0.74, 0.02), xycoords='axes fraction')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.xticks([])
plt.yticks([])
plt.axis('equal')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),  shadow=False, ncol=3)

plt.subplot(gs[2])
plt.plot([0,1],[0,1],'k:')
plt.plot(s1_FPR_all_int05[2],s1_TPR_all_int05[2],'g--')
plt.scatter(s1_FPR_all_int05[2][::20],s1_TPR_all_int05[2][::20],color='g')
plt.plot(s1_FPR_all_int1[2],s1_TPR_all_int1[2],'r-')
plt.scatter(s1_FPR_all_int1[2][::20],s1_TPR_all_int1[2][::20],color='r')
plt.plot(s1_FPR_all_int2[2],s1_TPR_all_int2[2],'b-.')
plt.scatter(s1_FPR_all_int2[2][::20],s1_TPR_all_int2[2][::20],color='b')
plt.annotate('1.2km', xy=(0.74, 0.02), xycoords='axes fraction')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.xticks([])
plt.yticks([])
plt.axis('equal')

plt.subplot(gs[3])
plt.plot([0,1],[0,1],'k:')
plt.plot(s1_FPR_all_int05[3],s1_TPR_all_int05[3],'g--')
plt.scatter(s1_FPR_all_int05[3][::20],s1_TPR_all_int05[3][::20],color='g')
plt.plot(s1_FPR_all_int1[3],s1_TPR_all_int1[3],'r-')
plt.scatter(s1_FPR_all_int1[3][::20],s1_TPR_all_int1[3][::20],color='r')
plt.plot(s1_FPR_all_int2[3],s1_TPR_all_int2[3],'b-.')
plt.scatter(s1_FPR_all_int2[3][::20],s1_TPR_all_int2[3][::20],color='b')
plt.annotate('1.6km', xy=(0.74, 0.02), xycoords='axes fraction')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.yticks(**tickfont)
plt.xticks([])
plt.ylabel('True Positive Rate',**labelfont)
plt.axis('equal')

plt.subplot(gs[4])
plt.plot([0,1],[0,1],'k:')
plt.plot(s1_FPR_all_int05[4],s1_TPR_all_int05[4],'g--')
plt.scatter(s1_FPR_all_int05[4][::20],s1_TPR_all_int05[4][::20],color='g')
plt.plot(s1_FPR_all_int1[4],s1_TPR_all_int1[4],'r-')
plt.scatter(s1_FPR_all_int1[4][::20],s1_TPR_all_int1[4][::20],color='r')
plt.plot(s1_FPR_all_int2[4],s1_TPR_all_int2[4],'b-.')
plt.scatter(s1_FPR_all_int2[4][::20],s1_TPR_all_int2[4][::20],color='b')
plt.annotate('2.0km', xy=(0.74, 0.02), xycoords='axes fraction')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.xticks([])
plt.yticks([])
plt.axis('equal')

plt.subplot(gs[5])
plt.plot([0,1],[0,1],'k:')
plt.plot(s1_FPR_all_int05[5],s1_TPR_all_int05[5],'g--')
plt.scatter(s1_FPR_all_int05[5][::20],s1_TPR_all_int05[5][::20],color='g')
plt.plot(s1_FPR_all_int1[5],s1_TPR_all_int1[5],'r-')
plt.scatter(s1_FPR_all_int1[5][::20],s1_TPR_all_int1[5][::20],color='r')
plt.plot(s1_FPR_all_int2[5],s1_TPR_all_int2[5],'b-.')
plt.scatter(s1_FPR_all_int2[5][::20],s1_TPR_all_int2[5][::20],color='b')
plt.annotate('3.0km', xy=(0.74, 0.02), xycoords='axes fraction')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.xticks([])
plt.yticks([])
plt.axis('equal')

plt.subplot(gs[6])
plt.plot([0,1],[0,1],'k:')
plt.plot(s1_FPR_all_int05[6],s1_TPR_all_int05[6],'g--')
plt.scatter(s1_FPR_all_int05[6][::20],s1_TPR_all_int05[6][::20],color='g')
plt.plot(s1_FPR_all_int1[6],s1_TPR_all_int1[6],'r-')
plt.scatter(s1_FPR_all_int1[6][::20],s1_TPR_all_int1[6][::20],color='r')
plt.plot(s1_FPR_all_int2[6],s1_TPR_all_int2[6],'b-.')
plt.scatter(s1_FPR_all_int2[6][::20],s1_TPR_all_int2[6][::20],color='b')
plt.annotate('5.0km', xy=(0.74, 0.02), xycoords='axes fraction')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.axis('equal')

plt.subplot(gs[7])
plt.plot([0,1],[0,1],'k:')
plt.plot(s1_FPR_all_int05[7],s1_TPR_all_int05[7],'g--')
plt.scatter(s1_FPR_all_int05[7][::20],s1_TPR_all_int05[7][::20],color='g')
plt.plot(s1_FPR_all_int1[7],s1_TPR_all_int1[7],'r-')
plt.scatter(s1_FPR_all_int1[7][::20],s1_TPR_all_int1[7][::20],color='r')
plt.plot(s1_FPR_all_int2[7],s1_TPR_all_int2[7],'b-.')
plt.scatter(s1_FPR_all_int2[7][::20],s1_TPR_all_int2[7][::20],color='b')
plt.annotate('7.5km', xy=(0.74, 0.02), xycoords='axes fraction')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.xlabel('False Positive Rate',**labelfont)
plt.yticks([])
plt.xticks(**tickfont)
plt.axis('equal')

plt.subplot(gs[8])
plt.plot([0,1],[0,1],'k:')
plt.plot(s1_FPR_all_int05[8],s1_TPR_all_int05[8],'g--')
plt.scatter(s1_FPR_all_int05[8][::20],s1_TPR_all_int05[8][::20],color='g')
plt.plot(s1_FPR_all_int1[8],s1_TPR_all_int1[8],'r-')
plt.scatter(s1_FPR_all_int1[8][::20],s1_TPR_all_int1[8][::20],color='r')
plt.plot(s1_FPR_all_int2[8],s1_TPR_all_int2[8],'b-.')
plt.scatter(s1_FPR_all_int2[8][::20],s1_TPR_all_int2[8][::20],color='b')
plt.annotate('10.0km', xy=(0.70, 0.02), xycoords='axes fraction')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.yticks([])
plt.xticks(**tickfont)
plt.axis('equal')

#plt.savefig('s1_subplots.eps'.format(radius), transparent=False, bbox_inches='tight',pad_inches=0)
#plt.savefig('s1_subplots.png'.format(radius), transparent=False, bbox_inches='tight',pad_inches=0)
plt.savefig('s1_subplots_hunterflight.eps'.format(radius), transparent=False, bbox_inches='tight',pad_inches=0)
plt.savefig('s1_subplots_hunterflight.png'.format(radius), transparent=False, bbox_inches='tight',pad_inches=0)