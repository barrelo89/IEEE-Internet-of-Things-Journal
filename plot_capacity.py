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

result = pd.read_pickle('data/result1.pkl')
x_axis = pd.Series(result.index, index = result.index)

#print(result.to_string())
#print(x_axis)
#--------------------------------------------------------------theoretic social welfare------------------------------------------------------------------------------------------------------------------
plt.plot(x_axis, result['mere1'],     'k*--', label = 'Naive: B1', markersize = 10)  
plt.plot(x_axis, result['mere2'],     'k>--', label = 'Naive: Proposed', markersize = 10)  
plt.plot(x_axis, result['expected1'], 'r8--', label = 'Expected: B1', markersize = 10)  
plt.plot(x_axis, result['expected2'], 'bs--', label = 'Expected: B2', markersize = 10)  
plt.plot(x_axis, result['expected3'], 'g^--', label = 'Expected: Proposed', markersize = 10)  
plt.xlabel('Platform Capacity', fontsize = 25)
plt.ylabel('Theoretic Social Welfare', fontsize = 25)
plt.xlim(90, 510)
#plt.ylim()
plt.legend(loc = 2, fontsize = 20)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.tight_layout()
plt.show()

#--------------------------------------------------------------real social welfare------------------------------------------------------------------------------------------------------------------
plt.plot(x_axis, result['real1'], 'r8--', label = 'B1', markersize = 10)  
plt.plot(x_axis, result['real2'], 'bs--', label = 'B2', markersize = 10)  
plt.plot(x_axis, result['real3'], 'g^--', label = 'Proposed', markersize = 10)  
plt.xlabel('Platform Capacity', fontsize = 25)
plt.ylabel('Simulated Social Welfare', fontsize = 25)
plt.xlim(90, 510)
plt.ylim(0, 30000)
plt.legend(loc = 2, fontsize = 25)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.tight_layout()
plt.show()

#--------------------------------------------------------------budget balance check------------------------------------------------------------------------------------------------------------------
plt.plot(x_axis, result['before1'], 'r8--', label = 'B1 before submission', markersize = 10) 
plt.plot(x_axis, result['before2'], 'bs--', label = 'B2 before submission', markersize = 10)  
plt.plot(x_axis, result['before3'], 'g^--', label = 'Proposed before submission', markersize = 10)  
plt.plot(x_axis, result['after1'], 'r', label = 'B1 after submission')
plt.plot(x_axis, result['after2'], 'b', label = 'B2 after submission')
plt.plot(x_axis, result['after3'], 'g', label = 'Proposed after submission')  
  
plt.xlabel('Platform Capacity', fontsize = 25)
plt.ylabel('Platform Utility', fontsize = 25)
plt.xlim(90, 510)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()

#average requester utility:
av_r1 = (result['real1'] - result['f_after1']) / x_axis # result['real1'] - result['f_after1'] : Series, x_axis : Series로 맞추어야 작동을 한다. 만약 series to dataframe이면 안 맞다.
av_r2 = (result['real2'] - result['f_after2']) / x_axis
av_r3 = (result['real3'] - result['f_after3']) / x_axis


#average provider utility:
av_p1 = (result['p_after1'] - result['cost1']) / x_axis
av_p2 = (result['p_after2'] - result['cost2']) / x_axis
av_p3 = (result['p_after3'] - result['cost2']) / x_axis

#--------------------------------------------------------------average requester utility------------------------------------------------------------------------------------------------------------------
plt.plot(x_axis, av_r1, 'r8--', label = 'b1')
plt.plot(x_axis, av_r2, 'gs--', label = 'b2')
plt.plot(x_axis, av_r3, 'bo--', label = 'proposed')
plt.xlim(90, 510)
plt.legend(loc='best', fontsize = 15)
plt.xlabel('Platform Capacity', fontsize = 25)
plt.ylabel('Average Requester Utility', fontsize = 25)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.tight_layout()
plt.show()

#--------------------------------------------------------------average provider utility------------------------------------------------------------------------------------------------------------------
plt.plot(x_axis, av_p1, 'r', label = 'b1')
plt.plot(x_axis, av_p2, 'g', label = 'b2')
plt.plot(x_axis, av_p3, 'b', label = 'proposed')
#plt.ylim(0,17000)
plt.xlim(90, 510)
plt.legend(loc='best', fontsize = 15)
plt.xlabel('Platform Capacity', fontsize = 25)
plt.ylabel('Average Provider Utility', fontsize = 25)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.tight_layout()
plt.show()























































