# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 16:46:36 206

@author: pnola
"""

import numpy as np
import h5py as hp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import trapz
import sys
epsilon = sys.float_info.epsilon
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
with hp.File('850mb_300m_10min_NAM_Rhodot_Origin_t=0-215hrs_Sept2017.hdf5','r') as data:
#with hp.File('hunterdata_r=02km_interpolated_2_cridges.hdf5','r') as data:
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
    for percent in np.linspace(100,0,101):
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
    '''
    FPR_x_05 = s1_FPR_int05[0]
    FPR_x_1 = s1_FPR_int1[0]
    FPR_x_2 = s1_FPR_int2[0]
    for i in range(1,len(s1_FPR_int05)):
        if s1_FPR_int05[i] <= FPR_x_05:
            s1_FPR_int05[i] = FPR_x_05 + 1e-10
        if s1_FPR_int1[i] <= FPR_x_1:
            s1_FPR_int1[i] = FPR_x_1 + 1e-10
        if s1_FPR_int2[i] <= FPR_x_2:
            s1_FPR_int2[i] = FPR_x_2 + 1e-10
    '''     
            
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


for P in range(9):
    plt.subplot(gs[P])
    plt.plot([0,1],[0,1],'k:')
    plt.plot(rhodot_FPR_all_int05[P],rhodot_TPR_all_int05[P],'g--')
    plt.scatter(rhodot_FPR_all_int05[P][::20],rhodot_TPR_all_int05[P][::20],color='g',label='{0:1.3f}'.format(trapz(rhodot_TPR_all_int05[P],rhodot_FPR_all_int05[P])/rhodot_FPR_all_int05[P][-1]))
    plt.plot(rhodot_FPR_all_int1[P],rhodot_TPR_all_int1[P],'r-')
    plt.scatter(rhodot_FPR_all_int1[P][::20],rhodot_TPR_all_int1[P][::20],color='r',label='{0:1.3f}'.format(trapz(rhodot_TPR_all_int1[P],rhodot_FPR_all_int1[P])/rhodot_FPR_all_int1[P][-1]))
    plt.plot(rhodot_FPR_all_int2[P],rhodot_TPR_all_int2[P],'b-.')
    plt.scatter(rhodot_FPR_all_int2[P][::20],rhodot_TPR_all_int2[P][::20],color='b',label='{0:1.3f}'.format(trapz(rhodot_TPR_all_int2[P],rhodot_FPR_all_int2[P])/rhodot_FPR_all_int2[P][-1]))
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.legend(fontsize=8,loc='lower right')
    plt.axis('equal')
    if P == 0:
        plt.annotate('0.4km', xy=(0.03, 0.9), xycoords='axes fraction')
        plt.yticks(**tickfont)
        plt.xticks([])
    elif P==1:
        plt.annotate('0.8km', xy=(0.03, 0.9), xycoords='axes fraction')
        plt.yticks([])
        plt.xticks([])
    elif P==2:
        plt.annotate('1.2km', xy=(0.03, 0.9), xycoords='axes fraction')
        plt.yticks([])
        plt.xticks([])
    elif P==3:
        plt.annotate('1.6km', xy=(0.03, 0.9), xycoords='axes fraction')
        plt.ylabel('True Positive Rate',**labelfont)
        plt.yticks(**tickfont)
        plt.xticks([])
    elif P==4:
        plt.annotate('2.0km', xy=(0.03, 0.9), xycoords='axes fraction')
        plt.yticks([])
        plt.xticks([])
    elif P==5:
        plt.annotate('3.0km', xy=(0.03, 0.9), xycoords='axes fraction')
        plt.yticks([])
        plt.xticks([])
    elif P==6:
        plt.annotate('5.0km', xy=(0.03, 0.9), xycoords='axes fraction')
        plt.yticks(**tickfont)
        plt.xticks(**tickfont)
    elif P==7:
        plt.annotate('7.5km', xy=(0.03, 0.9), xycoords='axes fraction')
        plt.xlabel('False Positive Rate',**labelfont)
        plt.yticks([])
        plt.xticks(**tickfont)
    elif P==8:
        plt.annotate('10.0km', xy=(0.03, 0.9), xycoords='axes fraction')      
        plt.yticks([])
        plt.xticks(**tickfont)
        
plt.savefig('Rhodot_subplots_v2.eps'.format(radius), transparent=False, bbox_inches='tight',pad_inches=0)
plt.savefig('Rhodot_subplots_v2.png'.format(radius), transparent=False, bbox_inches='tight',pad_inches=0)
#plt.savefig('Rhodot_subplots_hunterflight_v2.eps'.format(radius), transparent=False, bbox_inches='tight',pad_inches=0)
#plt.savefig('Rhodot_subplots_hunterflight_v2.png'.format(radius), transparent=False, bbox_inches='tight',pad_inches=0)

plt.figure(2,figsize=FigSize)
gs = gridspec.GridSpec(3, 3)
gs.update(wspace=0.05, hspace=0.05)

for P in range(9):
    plt.subplot(gs[P])
    plt.plot([0,1],[0,1],'k:')
    plt.plot(s1_FPR_all_int05[P],s1_TPR_all_int05[P],'g--')
    plt.scatter(s1_FPR_all_int05[P][::20],s1_TPR_all_int05[P][::20],color='g',label='{0:1.3f}'.format(trapz(s1_TPR_all_int05[P],s1_FPR_all_int05[P])/s1_FPR_all_int05[P][-1]))
    plt.plot(s1_FPR_all_int1[P],s1_TPR_all_int1[P],'r-')
    plt.scatter(s1_FPR_all_int1[P][::20],s1_TPR_all_int1[P][::20],color='r',label='{0:1.3f}'.format(trapz(s1_TPR_all_int1[P],s1_FPR_all_int1[P])/s1_FPR_all_int1[P][-1]))
    plt.plot(s1_FPR_all_int2[P],s1_TPR_all_int2[P],'b-.')
    plt.scatter(s1_FPR_all_int2[P][::20],s1_TPR_all_int2[P][::20],color='b',label='{0:1.3f}'.format(trapz(s1_TPR_all_int2[P],s1_FPR_all_int2[P])/s1_FPR_all_int2[P][-1]))
    #        trapz(s1_TPR_all_int2[P],[x/s1_FPR_all_int2[P][-1] for x in s1_FPR_all_int2[P]])))
    #        trapz([y/s1_TPR_all_int2[P][-1] for y in s1_TPR_all_int2[P]],[x/s1_FPR_all_int2[P][-1] for x in s1_FPR_all_int2[P]])))
    #        trapz(s1_TPR_all_int2[P],s1_FPR_all_int2[P])/s1_FPR_all_int2[P][-1]))
    #        trapz(s1_TPR_all_int2[P],s1_FPR_all_int2[P])))
    #        trapz(s1_TPR_all_int2[P],s1_FPR_all_int2[P])))
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.legend(fontsize=8,loc='lower right')
    plt.axis('equal')
    if P == 0:
        plt.annotate('0.4km', xy=(0.03, 0.9), xycoords='axes fraction')
        plt.yticks(**tickfont)
        plt.xticks([])
    elif P==1:
        plt.annotate('0.8km', xy=(0.03, 0.9), xycoords='axes fraction')
        plt.yticks([])
        plt.xticks([])
    elif P==2:
        plt.annotate('1.2km', xy=(0.03, 0.9), xycoords='axes fraction')
        plt.yticks([])
        plt.xticks([])
    elif P==3:
        plt.annotate('1.6km', xy=(0.03, 0.9), xycoords='axes fraction')
        plt.ylabel('True Positive Rate',**labelfont)
        plt.yticks(**tickfont)
        plt.xticks([])
    elif P==4:
        plt.annotate('2.0km', xy=(0.03, 0.9), xycoords='axes fraction')
        plt.yticks([])
        plt.xticks([])
    elif P==5:
        plt.annotate('3.0km', xy=(0.03, 0.9), xycoords='axes fraction')
        plt.yticks([])
        plt.xticks([])
    elif P==6:
        plt.annotate('5.0km', xy=(0.03, 0.9), xycoords='axes fraction')
        plt.yticks(**tickfont)
        plt.xticks(**tickfont)
    elif P==7:
        plt.annotate('7.5km', xy=(0.03, 0.9), xycoords='axes fraction')
        plt.xlabel('False Positive Rate',**labelfont)
        plt.yticks([])
        plt.xticks(**tickfont)
    elif P==8:
        plt.annotate('10.0km', xy=(0.03, 0.9), xycoords='axes fraction')      
        plt.yticks([])
        plt.xticks(**tickfont)
    
plt.savefig('s1_subplots_v2.eps'.format(radius), transparent=False, bbox_inches='tight',pad_inches=0)
plt.savefig('s1_subplots_v2.png'.format(radius), transparent=False, bbox_inches='tight',pad_inches=0)
#plt.savefig('s1_subplots_hunterflight_v2.eps'.format(radius), transparent=False, bbox_inches='tight',pad_inches=0)
#plt.savefig('s1_subplots_hunterflight_v2.png'.format(radius), transparent=False, bbox_inches='tight',pad_inches=0)

#'''