import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from sklearn.metrics import auc
import pandas as pd
import itertools

from task import Tasks
from provider import Provider 
from platforms import Platform
from functions import *
import multiprocessing

result = pd.read_pickle('alpha_result.pkl')
x_axis = result.index
#print(result.to_string())
#print(x_axis)
capacity = 300
#--------------------------------------------------------------theoretic social welfare------------------------------------------------------------------------------------------------------------------
plt.semilogx(x_axis, result['mere1'], 'k*--', label = 'NSW: b1')
plt.semilogx(x_axis, result['mere2'], 'k<--', label = 'NSW: proposed')
plt.semilogx(x_axis, result['expected1'], 'r8--', label = 'ESW: b1')
plt.semilogx(x_axis, result['expected2'], 'gs--', label = 'ESW: b2')
plt.semilogx(x_axis, result['expected3'], 'bo--', label = 'ESW: proposed')
plt.ylim(0,17000)
plt.xlim(0, 11)
plt.legend(loc='best', fontsize = 20)
plt.xlabel(r'Maximum Task Depreciation Coefficient, $\alpha_j$', fontsize = 20)
plt.ylabel('Theoretic Social Welfare', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.tight_layout()
plt.show()

#--------------------------------------------------------------real social welfare------------------------------------------------------------------------------------------------------------------
plt.semilogx(x_axis, result['real1'], 'r8--', label = 'RSW: b1')
plt.semilogx(x_axis, result['real2'], 'gs--', label = 'RSW: b2')
plt.semilogx(x_axis, result['real3'], 'bo--', label = 'RSW: proposed')
plt.ylim(0,17000)
plt.xlim(0, 11)
plt.legend(loc='best', fontsize = 20)
plt.xlabel(r'Maximum Task Depreciation Coefficient, $\alpha_j$', fontsize = 20)
plt.ylabel('Simulated Social Welfare', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.tight_layout()
plt.show()

#--------------------------------------------------------------budget balance check------------------------------------------------------------------------------------------------------------------
plt.semilogx(x_axis, result['before1'], 'r8--', label = 'before: b1')
plt.semilogx(x_axis, result['before2'], 'gs--', label = 'before: b2')
plt.semilogx(x_axis, result['before3'], 'bo--', label = 'before: proposed')
plt.semilogx(x_axis, result['after1'], 'r', label = 'after: b1')
plt.semilogx(x_axis, result['after2'], 'g', label = 'after: b2')
plt.semilogx(x_axis, result['after3'], 'b', label = 'after: proposed')
#plt.ylim(0,17000)
plt.xlim(0, 11)
plt.legend(loc='best', fontsize = 15)
plt.xlabel(r'Maximum Task Depreciation Coefficient, $\alpha_j$', fontsize = 20)
plt.ylabel('Platform Utility', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.tight_layout()
plt.show()

#--------------------------------------------------------------average requester utility------------------------------------------------------------------------------------------------------------------

#average requester utility:
av_r1 = (result['real1'] - result['f_after1']) / capacity
av_r2 = (result['real2'] - result['f_after2']) / capacity
av_r3 = (result['real3'] - result['f_after3']) / capacity


#average provider utility:
av_p1 = (result['p_after1'] - result['cost1']) / capacity
av_p2 = (result['p_after2'] - result['cost2']) / capacity
av_p3 = (result['p_after3'] - result['cost2']) / capacity

plt.semilogx(x_axis, av_r1, 'r8--', label = 'b1')
plt.semilogx(x_axis, av_r2, 'gs--', label = 'b2')
plt.semilogx(x_axis, av_r3, 'bo--', label = 'proposed')
plt.xlim(0, 11)
plt.legend(loc='best', fontsize = 15)
plt.xlabel(r'Maximum Task Depreciation Coefficient, $\alpha_j$', fontsize = 20)
plt.ylabel('Average Requester Utility', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.tight_layout()
plt.show()

#--------------------------------------------------------------average provider utility------------------------------------------------------------------------------------------------------------------
plt.semilogx(x_axis, av_p1, 'r', label = 'b1')
plt.semilogx(x_axis, av_p2, 'g', label = 'b2')
plt.semilogx(x_axis, av_p3, 'b', label = 'proposed')
#plt.ylim(0,17000)
plt.xlim(0, 11)
plt.legend(loc='best', fontsize = 15)
plt.xlabel(r'Maximum Task Depreciation Coefficient, $\alpha_j$', fontsize = 20)
plt.ylabel('Average provider Utility', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.tight_layout()
plt.show()

































