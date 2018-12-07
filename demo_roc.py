import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['lines.linewidth']=1
matplotlib.rcParams['lines.markersize']=2
plt.rc('font', **{'family': 'serif', 'serif': ['cmr10']})
tickfont = {'fontsize':8}
labelfont = {'fontsize':10}
x = np.linspace(0,1,101)
#y = np.log(x)
#y=-y/y.min()
y=-(x-1)**32+1
y=(x-1)**9+1
plt.close('all')
plt.figure(1,(4,4))
plt.plot([0,1],[0,1],'k:')
plt.plot(x,y)
plt.scatter(x[::20],y[::20])
plt.annotate(s='100\%',xy=(x[0],y[0]),xytext=(x[0]-0.04,0.06),fontsize=10)
plt.annotate(s='80\%',xy=(x[19],y[19]),xytext=(x[19]-0.03,0.91),fontsize=10)
plt.annotate(s='60\%',xy=(x[39],y[39]),xytext=(x[39]-0.03,1.03),fontsize=10)
plt.annotate(s='40\%',xy=(x[59],y[59]),xytext=(x[59]-0.03,1.03),fontsize=10)
plt.annotate(s='20\%',xy=(x[79],y[79]),xytext=(x[79]-0.03,1.03),fontsize=10)
plt.annotate(s='0\%',xy=(x[100],y[100]),xytext=(x[100]-0.03,1.03),fontsize=10)
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.yticks(**tickfont)
plt.xticks(**tickfont)

plt.ylabel('True Positive Rate',**labelfont)
plt.xlabel('False Positive Rate',**labelfont)
#plt.axis('equal')
plt.savefig('demo_roc.eps', transparent=False, bbox_inches='tight',pad_inches=0)
#plt.scatter(x,y)