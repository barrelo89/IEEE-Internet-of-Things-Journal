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

from scipy.interpolate import interp1d

result = pd.read_pickle('data/provider_number_result_1.pkl')
x_axis = result.index
#print(result.to_string())

capacity = 300
#16개

targets = ['mere2', 'expected2', 'expected3', 'real2', 'real3', 'before2', 'before3', 'after2', 'after3', 'p_before2', 'f_before2', 'p_after2', 'p_after3', 'f_after2', 'f_after3', 'cost2']
  
result = extrapolate(targets, x_axis, result)  
result.to_pickle('provider_number_result2.pkl')
print(result.to_string())
print(x_axis)


#average requester utility:
av_r1 = (result['real1'] - result['f_after1']) / capacity
av_r2 = (result['real2'] - result['f_after2']) / capacity
av_r3 = (result['real3'] - result['f_after3']) / capacity


#average provider utility:
av_p1 = (result['p_after1'] - result['cost1']) / capacity
av_p2 = (result['p_after2'] - result['cost2']) / capacity
av_p3 = (result['p_after3'] - result['cost2']) / capacity

plt.plot(x_axis, av_r1, 'r8--', label = 'b1')
plt.plot(x_axis, av_r2, 'bs--', label = 'b2')
plt.plot(x_axis, av_r3, 'g^--', label = 'proposed')
plt.ylim(0,40)
plt.xlim(490, 3010)
plt.legend(loc='best', fontsize = 20)
plt.xlabel('Number of Requesters', fontsize = 20)
plt.ylabel('Average Requester Utility', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.tight_layout()
plt.show()

plt.plot(x_axis, av_p1, 'r8--', label = 'b1')
plt.plot(x_axis, av_p2, 'bs--', label = 'b2')
plt.plot(x_axis, av_p3, 'g^--', label = 'proposed')
plt.legend(loc='best', fontsize = 20)
plt.xlim(490, 3010)
plt.xlabel('Number of Providers', fontsize = 20)
plt.ylabel('Average Provider Utility', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.tight_layout()
plt.show()



#result의 구조
#result[0]: mere1
#result[1]: mere2
#result[2]: expected1
#result[3]: expected2
#result[4]: expected3
#result[5]: real1
#result[6]: real2
#result[7]: real3
#result[8]: before1
#result[9]: before2
#result[10]: before3
#result[11]: after1
#result[12]: after2
#result[13]: after3
#result[14]: p_before1
#result[15]: p_before2
#result[16]: f_before1
#result[17]: f_before2
#result[18]: p_after1
#result[19]: p_after2
#result[20]: p_after3
#result[21]: f_after1
#result[22]: f_after2
#result[23]: f_after3
#result[24]: cost1
#result[25]: cost2












































