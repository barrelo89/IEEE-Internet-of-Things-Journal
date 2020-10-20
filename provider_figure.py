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

#--------------------------------PROVIDER.EPS-------------------------------------------------------------------------------------------------------------------------------------------------------------

time_unit = 0.8
max_value = 100
max_deadline = 100
max_alpha = 100 #나중에 max_alpha를 변화시키면서 한 번 보자
max_task_size = 10

max_provider_bid = 10
max_mu = 1.5

task1 = Tasks(max_value, max_alpha, max_deadline, max_task_size)
task1.original_value_ = max_value
task1.deadline_ = max_deadline/4
task1.expiry_ = 1.5*max_deadline/4

values1, X_axis = task1.TaskValueFunction(time_unit)

provider1 = Provider(max_mu, max_provider_bid, time_unit)
provider2 = Provider(max_mu, max_provider_bid, time_unit)
provider3 = Provider(max_mu, max_provider_bid, time_unit)

provider1.mean_ = 0.5
provider1.sigma_ = 2*provider1.mean_

provider2.mean_ = 0.8
provider2.sigma_ = 2*provider2.mean_

provider3.mean_ = 1.2
provider3.sigma_ = 2*provider3.mean_

probability1, cdf1, x_axis1 = provider1.prob_distribution(task1)
probability2, cdf2, x_axis2 = provider2.prob_distribution(task1)
probability3, cdf3, x_axis3 = provider3.prob_distribution(task1)

plt.plot(x_axis1, probability1, 'r--*', markersize= 8, label = r'$\mu$ = %.1f' %provider1.mean_, lw = 1)
plt.plot(x_axis1, probability2, 'g--v', markersize= 8, label = r'$\mu$ = %.1f' %provider2.mean_, lw = 1)
plt.plot(x_axis1, probability3, 'b--^', markersize= 8, label = r'$\mu$ = %.1f' %provider3.mean_, lw = 1)

plt.vlines(x = task1.expiry_, ymin = 0, ymax = max_value+10, colors = 'k', linestyles = 'dashed', label = r'$t^{ex}=38$', lw = 2)
plt.vlines(x = task1.deadline_, ymin = 0, ymax = max_value+10, colors = 'k', linestyles = 'dotted', label = r'$t^d=25$', lw = 2)
plt.legend(loc = 'best', fontsize = 20)
plt.ylim(0,0.6)
plt.xlim(0,task1.expiry_+10)
plt.xlabel('Time Elapsed', fontsize = 20)
plt.ylabel('Task Completion Probability', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.tight_layout()
plt.show()





































#end
